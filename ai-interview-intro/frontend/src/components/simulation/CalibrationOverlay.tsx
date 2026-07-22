/**
 * CalibrationOverlay.tsx — 5-point gaze calibration before the interview.
 * Shows 5 targets sequentially; fits a CalibrationModel from samples.
 * Falls back to auto-baseline on noisy data or skip.
 */
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { fitCalibration, headPoseFromMatrix } from './ProctorEngine';
import type { CalibrationModel } from './ProctorTypes';

interface CalibrationOverlayProps {
  faceLandmarker: any;
  videoRef: React.RefObject<HTMLVideoElement>;
  onComplete: (model: CalibrationModel | null) => void;
  onSkip: () => void;
}

// 5 targets: [normalizedX(-1..1), normalizedY(-1..1), label]
const TARGETS: Array<[number, number, string]> = [
  [0,    0,    'Center'],
  [-0.8, -0.8, 'Top-left'],
  [0.8,  -0.8, 'Top-right'],
  [0.8,   0.8, 'Bottom-right'],
  [-0.8,  0.8, 'Bottom-left'],
];

const DWELL_MS = 1600;
const SKIP_TRAVEL_MS = 300;

const LEFT_IRIS  = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const LEFT_EYE_CORNERS:  [number, number] = [33, 133];
const RIGHT_EYE_CORNERS: [number, number] = [362, 263];

function irisOffsetX(lm: Array<{x:number;y:number;z:number}>, iris: number[], inner: number, outer: number): number {
  const cx = (lm[inner].x + lm[outer].x) / 2;
  const w  = Math.abs(lm[outer].x - lm[inner].x);
  if (w < 1e-3) return 0;
  const ic = iris.reduce((s, i) => s + lm[i].x, 0) / iris.length;
  return (ic - cx) / w;
}
function irisOffsetY(lm: Array<{x:number;y:number;z:number}>, iris: number[], inner: number, outer: number): number {
  const cy = (lm[inner].y + lm[outer].y) / 2;
  const w  = Math.abs(lm[outer].x - lm[inner].x);
  if (w < 1e-3) return 0;
  const ic = iris.reduce((s, i) => s + lm[i].y, 0) / iris.length;
  return (ic - cy) / w;
}

export const CalibrationOverlay: React.FC<CalibrationOverlayProps> = ({
  faceLandmarker, videoRef, onComplete, onSkip,
}) => {
  const [targetIdx, setTargetIdx] = useState(0);
  const [progress, setProgress] = useState(0); // 0-1 for current target
  const [statusMsg, setStatusMsg] = useState('');
  const rafRef = useRef<number>(0);
  const samplesRef = useRef<Array<{
    irisX: number; irisY: number; yawDeg: number; pitchDeg: number;
    targetX: number; targetY: number;
  }>>([]);
  const targetStartRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const currentTargetIdxRef = useRef(0);
  const doneRef = useRef(false);

  const finish = useCallback((allSamples: typeof samplesRef.current) => {
    if (doneRef.current) return;
    doneRef.current = true;
    cancelAnimationFrame(rafRef.current);
    if (allSamples.length < 10) {
      onComplete(null);
      return;
    }
    const model = fitCalibration(allSamples);
    if (model.quality === 'noisy') {
      setStatusMsg('Lighting looks tricky — using adaptive mode instead');
      setTimeout(() => onComplete(null), 1500);
    } else {
      onComplete(model);
    }
  }, [onComplete]);

  useEffect(() => {
    if (!faceLandmarker) { onSkip(); return; }
    let running = true;

    const loop = () => {
      if (!running) return;
      const video = videoRef.current;
      if (!video || video.readyState < 2 || video.currentTime === lastVideoTimeRef.current) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      lastVideoTimeRef.current = video.currentTime;

      let result: any;
      try { result = faceLandmarker.detectForVideo(video, performance.now()); }
      catch { rafRef.current = requestAnimationFrame(loop); return; }

      const lm = result?.faceLandmarks?.[0];
      if (!lm || lm.length < 478) { rafRef.current = requestAnimationFrame(loop); return; }

      const matData = result?.facialTransformationMatrixes?.[0]?.data;
      const pose = matData && matData.length >= 16
        ? headPoseFromMatrix(matData)
        : { yaw: 0, pitch: 0, roll: 0 };

      const irisX = (irisOffsetX(lm, LEFT_IRIS,  LEFT_EYE_CORNERS[0],  LEFT_EYE_CORNERS[1]) +
                     irisOffsetX(lm, RIGHT_IRIS, RIGHT_EYE_CORNERS[0], RIGHT_EYE_CORNERS[1])) / 2;
      const irisY = (irisOffsetY(lm, LEFT_IRIS,  LEFT_EYE_CORNERS[0],  LEFT_EYE_CORNERS[1]) +
                     irisOffsetY(lm, RIGHT_IRIS, RIGHT_EYE_CORNERS[0], RIGHT_EYE_CORNERS[1])) / 2;

      const now = Date.now();
      const idx = currentTargetIdxRef.current;
      const [tx, ty] = TARGETS[idx];

      if (targetStartRef.current === null) targetStartRef.current = now;
      const elapsed = now - targetStartRef.current;

      if (elapsed > SKIP_TRAVEL_MS) {
        samplesRef.current.push({
          irisX, irisY, yawDeg: pose.yaw, pitchDeg: pose.pitch,
          targetX: tx, targetY: ty,
        });
        setProgress(Math.min(1, (elapsed - SKIP_TRAVEL_MS) / (DWELL_MS - SKIP_TRAVEL_MS)));
      }

      if (elapsed >= DWELL_MS) {
        const nextIdx = idx + 1;
        if (nextIdx >= TARGETS.length) {
          finish(samplesRef.current);
          return;
        }
        currentTargetIdxRef.current = nextIdx;
        setTargetIdx(nextIdx);
        setProgress(0);
        targetStartRef.current = null;
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => { running = false; cancelAnimationFrame(rafRef.current); };
  }, [faceLandmarker, videoRef, finish, onSkip]);

  const [tx, ty] = TARGETS[targetIdx];
  // Convert normalized [-1,1] to CSS percent [8%, 92%]
  const left = `${((tx + 1) / 2) * 84 + 8}%`;
  const top  = `${((ty + 1) / 2) * 84 + 8}%`;

  return (
    <div className="fixed inset-0 z-50 bg-black/95 flex flex-col items-center justify-center">
      {statusMsg ? (
        <p className="text-amber-400 text-sm font-semibold tracking-widest uppercase">{statusMsg}</p>
      ) : (
        <>
          <p className="text-white/60 text-xs uppercase tracking-widest mb-2">
            Gaze Calibration — {targetIdx + 1} / {TARGETS.length}
          </p>
          <p className="text-white/40 text-xs mb-8">Look at each dot until it fills</p>

          {/* Target dot */}
          <div className="absolute" style={{ left, top, transform: 'translate(-50%, -50%)' }}>
            <span className="relative flex h-6 w-6">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-6 w-6 bg-indigo-500" />
            </span>
            {/* Progress ring */}
            <svg className="absolute inset-0 -m-2" width="40" height="40" viewBox="0 0 40 40">
              <circle cx="20" cy="20" r="18" fill="none" stroke="rgba(99,102,241,0.3)" strokeWidth="2" />
              <circle
                cx="20" cy="20" r="18" fill="none" stroke="#818cf8" strokeWidth="2"
                strokeDasharray={`${2 * Math.PI * 18}`}
                strokeDashoffset={`${2 * Math.PI * 18 * (1 - progress)}`}
                strokeLinecap="round"
                style={{ transform: 'rotate(-90deg)', transformOrigin: '20px 20px', transition: 'stroke-dashoffset 0.05s linear' }}
              />
            </svg>
          </div>

          <button
            onClick={onSkip}
            className="absolute bottom-8 text-white/30 hover:text-white/60 text-xs uppercase tracking-widest transition-colors"
          >
            Skip calibration
          </button>
        </>
      )}
    </div>
  );
};

export default CalibrationOverlay;
