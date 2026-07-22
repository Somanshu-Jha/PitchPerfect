/**
 * ProctorEngine.ts — head-pose-compensated gaze + violation fusion.
 * Pure logic: no React, no MediaPipe loading (caller feeds frames).
 */
import type {
  CalibrationModel,
  ProctorEvent,
  ProctorMetrics,
  ProctorPolicyConfig,
  ProctorViolationType,
} from './ProctorTypes';

// ── MediaPipe 478-landmark indices ────────────────────────────────────────────
const LEFT_IRIS = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const LEFT_EYE_CORNERS: [number, number] = [33, 133];
const RIGHT_EYE_CORNERS: [number, number] = [362, 263];

interface P3 { x: number; y: number; z: number }

const centroid = (pts: P3[]): P3 => ({
  x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
  y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
  z: pts.reduce((s, p) => s + p.z, 0) / pts.length,
});

function irisOffset(iris: P3, inner: P3, outer: P3): { x: number; y: number } {
  const cx = (inner.x + outer.x) / 2;
  const cy = (inner.y + outer.y) / 2;
  const w = Math.abs(outer.x - inner.x);
  if (w < 1e-3) return { x: 0, y: 0 };
  return { x: (iris.x - cx) / w, y: (iris.y - cy) / w };
}

/** Euler angles (degrees) from MediaPipe column-major 4×4 transformation matrix. */
export function headPoseFromMatrix(m: number[]): { yaw: number; pitch: number; roll: number } {
  const r = 180 / Math.PI;
  return {
    yaw:   Math.atan2(m[8], m[10]) * r,
    pitch: Math.asin(Math.max(-1, Math.min(1, -m[9]))) * r,
    roll:  Math.atan2(m[1], m[5]) * r,
  };
}

/** Least-squares fit: screen = a0 + a1*iris + a2*headAngle from calibration samples. */
export function fitCalibration(samples: Array<{
  irisX: number; irisY: number; yawDeg: number; pitchDeg: number;
  targetX: number; targetY: number;
}>): CalibrationModel {
  const solve = (rows: Array<[number, number, number]>, b: number[]): [number, number, number] => {
    let s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
    let t0 = 0, t1 = 0, t2 = 0;
    rows.forEach((r, i) => {
      s00 += r[0]*r[0]; s01 += r[0]*r[1]; s02 += r[0]*r[2];
      s11 += r[1]*r[1]; s12 += r[1]*r[2]; s22 += r[2]*r[2];
      t0 += r[0]*b[i]; t1 += r[1]*b[i]; t2 += r[2]*b[i];
    });
    const det = s00*(s11*s22 - s12*s12) - s01*(s01*s22 - s12*s02) + s02*(s01*s12 - s11*s02);
    if (Math.abs(det) < 1e-9) return [0, 0, 0];
    const dx = t0*(s11*s22 - s12*s12) - s01*(t1*s22 - s12*t2) + s02*(t1*s12 - s11*t2);
    const dy = s00*(t1*s22 - t2*s12) - t0*(s01*s22 - s02*s12) + s02*(s01*t2 - t1*s02);
    const dz = s00*(s11*t2 - s12*t1) - s01*(s01*t2 - s02*t1) + t0*(s01*s12 - s11*s02);
    return [dx/det, dy/det, dz/det];
  };
  const ax = solve(samples.map(s => [1, s.irisX, s.yawDeg]),   samples.map(s => s.targetX));
  const ay = solve(samples.map(s => [1, s.irisY, s.pitchDeg]), samples.map(s => s.targetY));
  const rms = Math.sqrt(samples.reduce((acc, s) => {
    const px = ax[0] + ax[1]*s.irisX + ax[2]*s.yawDeg;
    const py = ay[0] + ay[1]*s.irisY + ay[2]*s.pitchDeg;
    return acc + (px - s.targetX)**2 + (py - s.targetY)**2;
  }, 0) / Math.max(1, samples.length));
  const degenerate = ax.every(v => v === 0) || ay.every(v => v === 0);
  return {
    ax: ax as [number, number, number],
    ay: ay as [number, number, number],
    eyeOpenBaseline: 0,
    quality: degenerate || rms > 0.35 ? 'noisy' : 'good',
  };
}

// ── DwellFlag ─────────────────────────────────────────────────────────────────
/** Fires once after `active` held for dwellMs; re-arms cooldownMs after deactivation. */
export class DwellFlag {
  private activeSince: number | null = null;
  private firedAt: number | null = null;
  private inactiveSince: number | null = null;
  constructor(private dwellMs: number, private cooldownMs: number) {}
  update(active: boolean, now: number): boolean {
    if (!active) {
      if (this.activeSince !== null || this.firedAt !== null) {
        if (this.inactiveSince === null) this.inactiveSince = now;
        if (this.firedAt !== null && now - this.inactiveSince! >= this.cooldownMs) {
          this.firedAt = null;
        }
      }
      this.activeSince = null;
      return false;
    }
    this.inactiveSince = null;
    if (this.firedAt !== null) return false;
    if (this.activeSince === null) this.activeSince = now;
    if (now - this.activeSince >= this.dwellMs) {
      this.firedAt = now;
      return true;
    }
    return false;
  }
}

// ── Rolling buffer ────────────────────────────────────────────────────────────
class RollingBuf {
  private buf: number[] = [];
  constructor(private size: number) {}
  push(v: number) { this.buf.push(v); if (this.buf.length > this.size) this.buf.shift(); }
  get values() { return this.buf; }
  get length() { return this.buf.length; }
  avg(): number { return this.buf.length ? this.buf.reduce((a, b) => a + b, 0) / this.buf.length : 0; }
  variance(): number {
    if (this.buf.length < 2) return 0;
    const m = this.avg();
    return this.buf.reduce((a, b) => a + (b - m) ** 2, 0) / this.buf.length;
  }
  std(): number { return Math.sqrt(this.variance()); }
  masd(): number {
    if (this.buf.length < 2) return 0;
    let s = 0;
    for (let i = 1; i < this.buf.length; i++) s += Math.abs(this.buf[i] - this.buf[i-1]);
    return s / (this.buf.length - 1);
  }
  firstDiffVariance(): number {
    if (this.buf.length < 3) return 0;
    const diffs = [];
    for (let i = 1; i < this.buf.length; i++) diffs.push(this.buf[i] - this.buf[i-1]);
    const m = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    return diffs.reduce((a, b) => a + (b - m) ** 2, 0) / diffs.length;
  }
}

// ── FaceLandmarkerResult type (subset we use) ─────────────────────────────────
export interface FaceLandmarkerResult {
  faceLandmarks: Array<Array<{ x: number; y: number; z: number }>>;
  faceBlendshapes?: Array<{ categories: Array<{ categoryName: string; score: number }> }>;
  facialTransformationMatrixes?: Array<{ data: number[] }>;
}

// ── ProctorEngine ─────────────────────────────────────────────────────────────
export class ProctorEngine {
  private policy: ProctorPolicyConfig;
  private onEvent: (e: ProctorEvent) => void;
  private onMetrics: (m: ProctorMetrics) => void;

  private calibration: CalibrationModel | null = null;
  private candidateSpeaking = false;

  // Auto-baseline (first 240 frames when no calibration)
  private autoBaselineFrames = 0;
  private autoBaselineIrisX = new RollingBuf(240);
  private autoBaselineIrisY = new RollingBuf(240);
  private autoBaselineYaw = new RollingBuf(240);
  private autoBaselinePitch = new RollingBuf(240);
  private autoBaselineLocked = false;
  private autoBaselineX = 0;
  private autoBaselineY = 0;

  // Rolling buffers
  private rawIrisX = new RollingBuf(90);
  private rawIrisY = new RollingBuf(90);
  private yawBuf = new RollingBuf(90);
  private screenGazeMag = new RollingBuf(30);
  private eyeContactBuf = new RollingBuf(90);
  private gazeXHistory = new RollingBuf(30);
  private readingScoreBuf = new RollingBuf(60);
  private blinkTimestamps: number[] = [];

  // Dwell flags (base durations; multiplied by policy.dwellMultiplier)
  private dwellGaze: DwellFlag;
  private dwellReading: DwellFlag;
  private dwellSecondPerson: DwellFlag;
  private dwellCameraAbsent: DwellFlag;
  private dwellSynthetic: DwellFlag;

  // Phone detection (notePhoneDetection)
  private phoneHits = 0;
  private lastPhoneHitMs = 0;
  private lastPhoneFireMs = 0;

  // Blink state
  private lastBlinkClosed = false;

  constructor(opts: {
    policy: ProctorPolicyConfig;
    onEvent: (e: ProctorEvent) => void;
    onMetrics: (m: ProctorMetrics) => void;
  }) {
    this.policy = opts.policy;
    this.onEvent = opts.onEvent;
    this.onMetrics = opts.onMetrics;
    const m = this.policy.dwellMultiplier;
    this.dwellGaze         = new DwellFlag(2500 * m, 4000);
    this.dwellReading      = new DwellFlag(3000 * m, 8000);
    this.dwellSecondPerson = new DwellFlag(1500 * m, 8000);
    this.dwellCameraAbsent = new DwellFlag(4000 * m, 8000);
    this.dwellSynthetic    = new DwellFlag(5000 * m, 15000);
  }

  setCalibration(model: CalibrationModel | null): void {
    this.calibration = model;
  }

  setCandidateSpeaking(speaking: boolean): void {
    this.candidateSpeaking = speaking;
  }

  notePhoneDetection(score: number, nowMs: number): void {
    if (score < 0.55) return;
    if (nowMs - this.lastPhoneHitMs < 3000) {
      this.phoneHits++;
    } else {
      this.phoneHits = 1;
    }
    this.lastPhoneHitMs = nowMs;
    if (this.phoneHits >= 2 && nowMs - this.lastPhoneFireMs > 15000) {
      this.lastPhoneFireMs = nowMs;
      this.phoneHits = 0;
      this.onEvent({ eventType: 'PHONE_DETECTED', confidence: score, tsMs: nowMs });
    }
  }

  noteExternalViolation(type: 'TAB_SWITCHED' | 'FOCUS_LOST', nowMs: number): void {
    this.onEvent({ eventType: type, confidence: 1.0, tsMs: nowMs });
  }

  processFrame(result: FaceLandmarkerResult, nowMs: number): void {
    const numFaces = result.faceLandmarks.length;

    // ── Camera absent ──────────────────────────────────────────────
    if (this.dwellCameraAbsent.update(numFaces === 0, nowMs)) {
      this.onEvent({ eventType: 'CAMERA_ABSENT', confidence: 1.0, tsMs: nowMs });
    }

    // ── Second person ──────────────────────────────────────────────
    if (this.dwellSecondPerson.update(numFaces >= 2, nowMs)) {
      this.onEvent({ eventType: 'SECOND_PERSON', confidence: 1.0, tsMs: nowMs });
    }

    if (numFaces === 0) {
      this.onMetrics(this._buildMetrics(0, 'CENTER', { yaw: 0, pitch: 0, roll: 0 },
        0, 0, 0, false, false, false, false, 0));
      return;
    }

    const lm = result.faceLandmarks[0];
    if (!lm || lm.length < 478) return;

    // ── Head pose ──────────────────────────────────────────────────
    const matData = result.facialTransformationMatrixes?.[0]?.data;
    const headPose = matData && matData.length >= 16
      ? headPoseFromMatrix(matData)
      : { yaw: 0, pitch: 0, roll: 0 };

    // ── Tracking confidence ────────────────────────────────────────
    const lmYs = lm.map(p => p.y);
    const faceHeight = Math.max(...lmYs) - Math.min(...lmYs);
    let trackingConf = 1.0;
    if (faceHeight < 0.12) trackingConf *= 0.5;

    // ── Iris gaze ──────────────────────────────────────────────────
    const leftIrisC  = centroid(LEFT_IRIS.map(i => lm[i]));
    const rightIrisC = centroid(RIGHT_IRIS.map(i => lm[i]));
    const leftOff  = irisOffset(leftIrisC,  lm[LEFT_EYE_CORNERS[0]],  lm[LEFT_EYE_CORNERS[1]]);
    const rightOff = irisOffset(rightIrisC, lm[RIGHT_EYE_CORNERS[0]], lm[RIGHT_EYE_CORNERS[1]]);
    const rawIrisXv = (leftOff.x + rightOff.x) / 2;
    const rawIrisYv = (leftOff.y + rightOff.y) / 2;

    // Per-eye disagreement check
    const eyeDisagree = Math.abs(leftOff.x - rightOff.x);
    if (eyeDisagree > 0.15) trackingConf *= 0.6;

    this.rawIrisX.push(rawIrisXv);
    this.rawIrisY.push(rawIrisYv);
    this.yawBuf.push(headPose.yaw);

    // ── Auto-baseline collection ───────────────────────────────────
    if (!this.calibration && !this.autoBaselineLocked) {
      this.autoBaselineIrisX.push(rawIrisXv);
      this.autoBaselineIrisY.push(rawIrisYv);
      this.autoBaselineYaw.push(headPose.yaw);
      this.autoBaselinePitch.push(headPose.pitch);
      this.autoBaselineFrames++;
      if (this.autoBaselineFrames >= 240) {
        this.autoBaselineX = this.autoBaselineIrisX.avg();
        this.autoBaselineY = this.autoBaselineIrisY.avg();
        this.autoBaselineLocked = true;
      }
    }

    // ── Screen gaze computation ────────────────────────────────────
    let screenX: number, screenY: number;
    if (this.calibration && this.calibration.quality !== 'auto') {
      const { ax, ay } = this.calibration;
      screenX = ax[0] + ax[1]*rawIrisXv + ax[2]*headPose.yaw;
      screenY = ay[0] + ay[1]*rawIrisYv + ay[2]*headPose.pitch;
    } else {
      // Auto-baseline: compensate yaw, subtract baseline
      const compX = rawIrisXv - 0.006 * headPose.yaw;
      const compY = rawIrisYv - 0.004 * headPose.pitch;
      screenX = compX - this.autoBaselineX;
      screenY = compY - this.autoBaselineY;
    }

    const gazeMag = Math.sqrt(screenX**2 + screenY**2);
    this.screenGazeMag.push(gazeMag);
    const stableGazeMag = this.screenGazeMag.avg();

    const threshold = this.calibration && this.calibration.quality !== 'auto' ? 1.15 : 0.28;
    const onScreen = Math.abs(screenX) <= threshold && Math.abs(screenY) <= threshold;
    const eyeContactThreshold = this.calibration && this.calibration.quality !== 'auto' ? 0.55 : 0.15;
    const isContact = stableGazeMag < eyeContactThreshold;

    this.eyeContactBuf.push(isContact ? 100 : 0);
    const eyeContactPercent = Math.round(this.eyeContactBuf.avg());

    // ── Gaze direction label ───────────────────────────────────────
    let gazeDirection: ProctorMetrics['gazeDirection'] = 'CENTER';
    if (Math.abs(screenX) > Math.abs(screenY)) {
      gazeDirection = screenX > 0.12 ? 'RIGHT' : screenX < -0.12 ? 'LEFT' : 'CENTER';
    } else {
      gazeDirection = screenY > 0.12 ? 'DOWN' : screenY < -0.12 ? 'UP' : 'CENTER';
    }

    // ── Blink (blendshapes) ────────────────────────────────────────
    const shapes = result.faceBlendshapes?.[0]?.categories ?? [];
    const blinkL = shapes.find(c => c.categoryName === 'eyeBlinkLeft')?.score ?? 0;
    const blinkR = shapes.find(c => c.categoryName === 'eyeBlinkRight')?.score ?? 0;
    const isClosed = (blinkL + blinkR) / 2 > 0.5;
    if (isClosed && !this.lastBlinkClosed) {
      this.blinkTimestamps.push(nowMs);
    }
    this.lastBlinkClosed = isClosed;
    // Keep only last 60s
    this.blinkTimestamps = this.blinkTimestamps.filter(t => nowMs - t < 60000);
    const blinkRatePerMin = this.blinkTimestamps.length;

    // ── Reading detection ──────────────────────────────────────────
    this.gazeXHistory.push(screenX);
    let dirChanges = 0;
    const gxv = this.gazeXHistory.values;
    for (let i = 2; i < gxv.length; i++) {
      if ((gxv[i] - gxv[i-1]) * (gxv[i-1] - gxv[i-2]) < -0.001) dirChanges++;
    }
    const readingLikely = dirChanges > 6 && eyeContactPercent < 50;
    this.readingScoreBuf.push(readingLikely ? 100 : 0);
    const isReading = this.readingScoreBuf.avg() > 60;

    // ── Synthetic eye contact detection ───────────────────────────
    let syntheticScore = 0;
    if (this.rawIrisX.length >= 90) {
      const fdv = this.rawIrisX.firstDiffVariance();
      const cue1 = fdv < 5e-7 ? 1 : 0;
      const cue2 = (this.yawBuf.std() > 1.5 && this.rawIrisX.std() < 0.004) ? 1 : 0;
      const masd = this.rawIrisX.masd();
      const std  = this.rawIrisX.std();
      const cue4 = masd / (std + 1e-6) < 0.08 ? 1 : 0;
      // Cue3: blink decorrelation — simplified: if blinking but iris variance near zero
      const cue3 = (blinkRatePerMin > 5 && std < 0.003) ? 1 : 0;
      syntheticScore = (cue1 + cue2 + cue3 + cue4) / 4;
    }
    const isSyntheticEyeContact = syntheticScore >= 0.5 && isContact;

    // ── Nervousness / confidence / focus ──────────────────────────
    const blinkNervous = blinkRatePerMin > 25 ? 40 : blinkRatePerMin > 20 ? 20 : 0;
    const gazeNervous  = gazeDirection !== 'CENTER' ? 30 : 0;
    const readNervous  = isReading ? 30 : 0;
    const nervousnessScore = Math.min(100, blinkNervous + gazeNervous + readNervous);
    const confScore = eyeContactPercent - nervousnessScore * 0.5;
    let confidenceLevel: ProctorMetrics['confidenceLevel'] = 'Medium';
    if (confScore > 70) confidenceLevel = 'Very High';
    else if (confScore > 55) confidenceLevel = 'High';
    else if (confScore > 35) confidenceLevel = 'Medium';
    else if (confScore > 15) confidenceLevel = 'Low';
    else confidenceLevel = 'Very Low';
    let focusLevel: ProctorMetrics['focusLevel'] = 'High';
    if (eyeContactPercent < 40 || isReading) focusLevel = 'Low';
    else if (eyeContactPercent < 65) focusLevel = 'Medium';

    // ── Emit violation events (gated by trackingConf) ─────────────
    const gated = trackingConf >= 0.5;

    // GAZE_OFF_SCREEN — grace: brief upward glance while speaking
    const gazeOffActive = gated && !onScreen && !(gazeDirection === 'UP' && this.candidateSpeaking);
    if (this.dwellGaze.update(gazeOffActive, nowMs)) {
      this.onEvent({ eventType: 'GAZE_OFF_SCREEN', confidence: trackingConf, tsMs: nowMs,
                     meta: { direction: gazeDirection } });
    }

    // READING_DETECTED
    if (this.dwellReading.update(gated && isReading, nowMs)) {
      this.onEvent({ eventType: 'READING_DETECTED', confidence: 0.8, tsMs: nowMs });
    }

    // SYNTHETIC_EYE_CONTACT
    if (this.dwellSynthetic.update(gated && isSyntheticEyeContact, nowMs)) {
      this.onEvent({ eventType: 'SYNTHETIC_EYE_CONTACT', confidence: syntheticScore, tsMs: nowMs });
    }

    this.onMetrics(this._buildMetrics(
      eyeContactPercent, gazeDirection, headPose, blinkRatePerMin,
      nervousnessScore, trackingConf, isReading, isSyntheticEyeContact,
      true, !!(this.calibration && this.calibration.quality !== 'auto'),
      numFaces, confidenceLevel, focusLevel,
    ));
  }

  private _buildMetrics(
    eyeContactPercent: number,
    gazeDirection: ProctorMetrics['gazeDirection'],
    headPose: { yaw: number; pitch: number; roll: number },
    blinkRatePerMin: number,
    nervousnessScore: number,
    trackingConfidence: number,
    isReading: boolean,
    isSyntheticEyeContact: boolean,
    facePresent: boolean,
    calibrated: boolean,
    facesDetected: number,
    confidenceLevel: ProctorMetrics['confidenceLevel'] = 'Medium',
    focusLevel: ProctorMetrics['focusLevel'] = 'High',
  ): ProctorMetrics {
    return {
      eyeContactPercent,
      gazeDirection,
      headPose,
      blinkRatePerMin: Math.min(50, blinkRatePerMin),
      facesDetected,
      nervousnessScore,
      confidenceLevel,
      focusLevel,
      isReading,
      facePresent,
      isSyntheticEyeContact,
      calibrated,
      trackingConfidence,
    };
  }
}
