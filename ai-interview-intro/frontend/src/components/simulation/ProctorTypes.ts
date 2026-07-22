/**
 * ProctorTypes.ts — shared types for the ProctorEngine pipeline.
 */

export type ProctorViolationType =
  | 'GAZE_OFF_SCREEN'
  | 'READING_DETECTED'
  | 'FOCUS_LOST'
  | 'SECOND_PERSON'
  | 'CAMERA_ABSENT'
  | 'PHONE_DETECTED'
  | 'SYNTHETIC_EYE_CONTACT'
  | 'TAB_SWITCHED';

export interface ProctorEvent {
  eventType: ProctorViolationType;
  confidence: number;      // 0-1
  tsMs: number;            // Date.now()
  meta?: Record<string, unknown>;
}

export interface ProctorMetrics {
  eyeContactPercent: number;
  gazeDirection: 'CENTER' | 'LEFT' | 'RIGHT' | 'UP' | 'DOWN';
  headPose: { yaw: number; pitch: number; roll: number };
  blinkRatePerMin: number;
  facesDetected: number;
  nervousnessScore: number;
  confidenceLevel: 'Very Low' | 'Low' | 'Medium' | 'High' | 'Very High';
  focusLevel: 'Low' | 'Medium' | 'High';
  isReading: boolean;
  facePresent: boolean;
  isSyntheticEyeContact: boolean;
  calibrated: boolean;
  trackingConfidence: number;
}

export interface CalibrationModel {
  ax: [number, number, number];
  ay: [number, number, number];
  eyeOpenBaseline: number;
  quality: 'good' | 'noisy' | 'auto';
}

export interface ProctorPolicyConfig {
  dwellMultiplier: number;
}
