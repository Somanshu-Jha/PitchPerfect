/**
 * CandidateObserver.tsx — thin React shell over ProctorEngine.
 * Loads MediaPipe FaceLandmarker, feeds frames to ProctorEngine,
 * renders the metrics panel, and forwards ProctorEvents upstream.
 *
 * Backward-compat: exports CandidateMetrics alias + legacy onSignal prop
 * so InterviewSimulation.tsx keeps compiling without changes.
 */
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ProctorEngine } from './ProctorEngine';
import { PhoneDetector } from './PhoneDetector';
import type {
  ProctorMetrics,
  ProctorEvent,
  ProctorPolicyConfig,
  CalibrationModel,
} from './ProctorTypes';
import type { CandidateSignal } from './HRBehaviorEngine';

// ── Public re-exports ─────────────────────────────────────────────────────────
export type CandidateMetrics = ProctorMetrics;
export type { ProctorMetrics, ProctorEvent, ProctorPolicyConfig, CalibrationModel };

// ── Props ─────────────────────────────────────────────────────────────────────
interface CandidateObserverProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  isActive: boolean;
  policy?: ProctorPolicyConfig;
  calibration?: CalibrationModel | null;
  candidateSpeaking?: boolean;
  onProctorEvent?: (e: ProctorEvent) => void;
  onMetricsUpdate?: (m: ProctorMetrics) => void;
  // Legacy prop for InterviewSimulation.tsx compatibility
  onSignal?: (signal: CandidateSignal) => void;
}

// ── MediaPipe loader (shared; also exported for CalibrationOverlay) ───────────
export async function loadMediaPipeFaceLandmarker(): Promise<any | null> {
  try {
    const { FaceLandmarker, FilesetResolver } = await import(
      /* @vite-ignore */
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    ) as any;
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    return await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: 'VIDEO',
      numFaces: 2,
    });
  } catch (e) {
    console.warn('[CandidateObserver] MediaPipe failed to load:', e);
    return null;
  }
}

// ── Default metrics ───────────────────────────────────────────────────────────
const DEFAULT_METRICS: ProctorMetrics = {
  eyeContactPercent: 75,
  gazeDirection: 'CENTER',
  headPose: { yaw: 0, pitch: 0, roll: 0 },
  blinkRatePerMin: 15,
  facesDetected: 1,
  nervousnessScore: 20,
  confidenceLevel: 'Medium',
  focusLevel: 'High',
  isReading: false,
  facePresent: true,
  isSyntheticEyeContact: false,
  calibrated: false,
  trackingConfidence: 1.0,
};

// ── Legacy signal rate-limiter ────────────────────────────────────────────────
const LEGACY_COOLDOWN_MS = 4000;

// ── Component ─────────────────────────────────────────────────────────────────
export const CandidateObserver: React.FC<CandidateObserverProps> = ({
  videoRef,
  isActive,
  policy = { dwellMultiplier: 1.0 },
  calibration = null,
  candidateSpeaking = false,
  onProctorEvent,
  onMetricsUpdate,
  onSignal,
}) => {
  const faceLandmarkerRef = useRef<any>(null);
  const engineRef = useRef<ProctorEngine | null>(null);
  const phoneDetectorRef = useRef<PhoneDetector | null>(null);
  const rafRef = useRef<number>(0);
  const lastVideoTimeRef = useRef(-1);
  const [mpReady, setMpReady] = useState(false);
  const [metrics, setMetrics] = useState<ProctorMetrics>(DEFAULT_METRICS);
  const legacyLastSignal = useRef<Record<string, number>>({});

  // ── Build/rebuild engine when policy changes ──────────────────────────────
  useEffect(() => {
    const engine = new ProctorEngine({
      policy,
      onEvent: (e) => {
        onProctorEvent?.(e);
        // Legacy signal bridge for InterviewSimulation.tsx
        if (onSignal) {
          const now = Date.now();
          const last = legacyLastSignal.current[e.eventType] ?? 0;
          if (now - last > LEGACY_COOLDOWN_MS) {
            legacyLastSignal.current[e.eventType] = now;
            if (e.eventType === 'READING_DETECTED') onSignal('READING_DETECTED');
            if (e.eventType === 'CAMERA_ABSENT') onSignal('CAMERA_COVERED_OR_ABSENT');
            if (e.eventType === 'SYNTHETIC_EYE_CONTACT') onSignal('SYNTHETIC_EYE_CONTACT_DETECTED');
            if (e.eventType === 'GAZE_OFF_SCREEN') onSignal('EYE_CONTACT_LOST');
          }
        }
      },
      onMetrics: (m) => {
        setMetrics(m);
        onMetricsUpdate?.(m);
      },
    });
    engine.setCalibration(calibration ?? null);
    engineRef.current = engine;
  }, [policy, onProctorEvent, onMetricsUpdate, onSignal]);

  // ── Sync calibration changes ──────────────────────────────────────────────
  useEffect(() => {
    engineRef.current?.setCalibration(calibration ?? null);
  }, [calibration]);

  // ── Sync speaking state ───────────────────────────────────────────────────
  useEffect(() => {
    engineRef.current?.setCandidateSpeaking(candidateSpeaking);
  }, [candidateSpeaking]);

  // ── Load MediaPipe ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isActive) return;
    let cancelled = false;
    loadMediaPipeFaceLandmarker().then(fl => {
      if (cancelled) return;
      faceLandmarkerRef.current = fl;
      setMpReady(!!fl);
    });
    return () => { cancelled = true; };
  }, [isActive]);

  // ── Start/stop PhoneDetector ──────────────────────────────────────────────
  useEffect(() => {
    if (!isActive || !videoRef.current) return;
    const pd = new PhoneDetector((score, nowMs) => {
      engineRef.current?.notePhoneDetection(score, nowMs);
    });
    phoneDetectorRef.current = pd;
    pd.start(videoRef.current);
    return () => { pd.stop(); phoneDetectorRef.current = null; };
  }, [isActive, videoRef]);

  // ── Tab/focus external violations ────────────────────────────────────────
  useEffect(() => {
    if (!isActive) return;
    const onVisibility = () => {
      if (document.hidden) engineRef.current?.noteExternalViolation('TAB_SWITCHED', Date.now());
    };
    const onBlur = () => engineRef.current?.noteExternalViolation('FOCUS_LOST', Date.now());
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('blur', onBlur);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('blur', onBlur);
    };
  }, [isActive]);

  // ── Detection loop ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isActive || !mpReady || !faceLandmarkerRef.current) return;
    let running = true;

    const detectLoop = () => {
      if (!running) return;
      const video = videoRef.current;
      if (video && video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        try {
          const result = faceLandmarkerRef.current.detectForVideo(video, performance.now());
          engineRef.current?.processFrame(result, Date.now());
        } catch { /* ignore single-frame errors */ }
      }
      rafRef.current = requestAnimationFrame(detectLoop);
    };

    rafRef.current = requestAnimationFrame(detectLoop);
    return () => { running = false; cancelAnimationFrame(rafRef.current); };
  }, [isActive, mpReady, videoRef]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-3">
      <MetricRow
        label="Eye Contact"
        value={`${metrics.eyeContactPercent}%`}
        fill={metrics.eyeContactPercent / 100}
        color={metrics.eyeContactPercent > 65 ? '#22c55e' : metrics.eyeContactPercent > 40 ? '#f59e0b' : '#ef4444'}
      />

      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/40 font-medium">Confidence</span>
        <span className={`text-[11px] font-black uppercase tracking-wider ${
          metrics.confidenceLevel === 'Very High' || metrics.confidenceLevel === 'High'
            ? 'text-emerald-400' : metrics.confidenceLevel === 'Medium'
            ? 'text-amber-400' : 'text-red-400'
        }`}>{metrics.confidenceLevel}</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/40 font-medium">Focus Level</span>
        <span className={`text-[11px] font-black uppercase tracking-wider ${
          metrics.focusLevel === 'High' ? 'text-emerald-400'
            : metrics.focusLevel === 'Medium' ? 'text-amber-400' : 'text-red-400'
        }`}>{metrics.focusLevel}</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/40 font-medium">Reading Check</span>
        <span className={`text-[11px] font-black uppercase tracking-wider ${
          metrics.isReading ? 'text-red-400 animate-pulse' : 'text-emerald-400'
        }`}>{metrics.isReading ? '⚠ Detected' : '✓ None'}</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/40 font-medium">Gaze</span>
        <GazeIndicator direction={metrics.gazeDirection} />
      </div>

      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/40 font-medium">Faces</span>
        <span className={`text-[11px] font-black uppercase tracking-wider ${
          metrics.facesDetected >= 2 ? 'text-red-400 animate-pulse' : 'text-emerald-400'
        }`}>{metrics.facesDetected >= 2 ? `⚠ ${metrics.facesDetected} detected` : '✓ 1'}</span>
      </div>

      <MetricRow
        label="Composure"
        value={`${Math.round(100 - metrics.nervousnessScore)}%`}
        fill={(100 - metrics.nervousnessScore) / 100}
        color={(100 - metrics.nervousnessScore) > 60 ? '#22c55e' : '#f59e0b'}
      />

      <MetricRow
        label="Tracking"
        value={`${Math.round(metrics.trackingConfidence * 100)}%`}
        fill={metrics.trackingConfidence}
        color={metrics.trackingConfidence >= 0.5 ? '#22c55e' : '#f59e0b'}
      />

      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/40 font-medium">Calibrated</span>
        <span className={`text-[11px] font-black uppercase tracking-wider ${
          metrics.calibrated ? 'text-emerald-400' : 'text-white/40'
        }`}>{metrics.calibrated ? '✓ Yes' : 'Adaptive'}</span>
      </div>

      {!mpReady && isActive && (
        <p className="text-[9px] text-white/20 text-center mt-1">Loading face analysis...</p>
      )}
    </div>
  );
};

// ── Sub-components ────────────────────────────────────────────────────────────
function MetricRow({ label, value, fill, color }: {
  label: string; value: string; fill: number; color: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/40 font-medium">{label}</span>
        <span className="text-[11px] font-black text-white/80">{value}</span>
      </div>
      <div className="h-1 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.round(fill * 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

function GazeIndicator({ direction }: { direction: ProctorMetrics['gazeDirection'] }) {
  const arrows: Record<string, string> = {
    CENTER: '●', LEFT: '←', RIGHT: '→', UP: '↑', DOWN: '↓',
  };
  const colors: Record<string, string> = {
    CENTER: 'text-emerald-400', LEFT: 'text-amber-400', RIGHT: 'text-amber-400',
    UP: 'text-amber-400', DOWN: 'text-amber-400',
  };
  return (
    <span className={`text-[13px] font-black ${colors[direction]} transition-colors duration-300`}>
      {arrows[direction]}
    </span>
  );
}

export default CandidateObserver;
