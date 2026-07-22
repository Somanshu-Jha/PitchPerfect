/**
 * HRBehaviorEngine.ts — Markov-Chain Micro-Behavior Engine
 * ----------------------------------------------------------
 * Drives the HR avatar's non-repetitive, probabilistic idle behaviors.
 * Ensures the interviewer NEVER feels like an animation loop.
 * Every session produces different, human-like behavior sequences.
 */

import type { AvatarPhysicsEngine } from './PhysicsEngine';

// ── Behavior States ─────────────────────────────────────────────────────────

export type BehaviorState =
  | 'ATTENTIVE'     // Looking at candidate, engaged
  | 'THINKING'      // Gaze up-left, processing
  | 'NOTING'        // Gaze down, writing notes
  | 'SKEPTICAL'     // Direct stare, evaluating
  | 'IMPRESSED'     // Brief forward lean + nod
  | 'WAITING'       // Neutral, listening patiently
  | 'CROSS_CHECK'   // Slightly narrowed gaze, verifying
  | 'APPROVING';    // Warm, slight nod + forward lean

// ── Behavior Emission ───────────────────────────────────────────────────────

export interface BehaviorEvent {
  state: BehaviorState;
  pose: string;
  gaze: string;
  emotion: string;
  microExpression: string;
  durationMs: number;
}

// ── Transition Table (Markov Chain) ─────────────────────────────────────────

interface Transition {
  to: BehaviorState;
  weight: number;
  minDurationMs: number;
  maxDurationMs: number;
}

const TRANSITIONS: Record<BehaviorState, Transition[]> = {
  ATTENTIVE: [
    { to: 'ATTENTIVE',  weight: 0.40, minDurationMs: 3000, maxDurationMs: 8000 },
    { to: 'THINKING',   weight: 0.20, minDurationMs: 2000, maxDurationMs: 5000 },
    { to: 'NOTING',     weight: 0.15, minDurationMs: 2500, maxDurationMs: 6000 },
    { to: 'WAITING',    weight: 0.15, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'SKEPTICAL',  weight: 0.05, minDurationMs: 1500, maxDurationMs: 3000 },
    { to: 'APPROVING',  weight: 0.05, minDurationMs: 1000, maxDurationMs: 2000 },
  ],
  THINKING: [
    { to: 'ATTENTIVE',  weight: 0.55, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'SKEPTICAL',  weight: 0.20, minDurationMs: 1500, maxDurationMs: 3000 },
    { to: 'CROSS_CHECK',weight: 0.15, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'NOTING',     weight: 0.10, minDurationMs: 2000, maxDurationMs: 5000 },
  ],
  NOTING: [
    { to: 'ATTENTIVE',  weight: 0.60, minDurationMs: 3000, maxDurationMs: 7000 },
    { to: 'THINKING',   weight: 0.25, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'WAITING',    weight: 0.15, minDurationMs: 2000, maxDurationMs: 4000 },
  ],
  SKEPTICAL: [
    { to: 'ATTENTIVE',  weight: 0.40, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'CROSS_CHECK',weight: 0.35, minDurationMs: 2000, maxDurationMs: 5000 },
    { to: 'THINKING',   weight: 0.25, minDurationMs: 1500, maxDurationMs: 3000 },
  ],
  IMPRESSED: [
    { to: 'ATTENTIVE',  weight: 0.50, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'APPROVING',  weight: 0.30, minDurationMs: 1500, maxDurationMs: 3000 },
    { to: 'THINKING',   weight: 0.20, minDurationMs: 2000, maxDurationMs: 4000 },
  ],
  WAITING: [
    { to: 'ATTENTIVE',  weight: 0.55, minDurationMs: 2000, maxDurationMs: 6000 },
    { to: 'THINKING',   weight: 0.25, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'NOTING',     weight: 0.20, minDurationMs: 2000, maxDurationMs: 5000 },
  ],
  CROSS_CHECK: [
    { to: 'SKEPTICAL',  weight: 0.40, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'ATTENTIVE',  weight: 0.35, minDurationMs: 2000, maxDurationMs: 5000 },
    { to: 'THINKING',   weight: 0.25, minDurationMs: 1500, maxDurationMs: 3000 },
  ],
  APPROVING: [
    { to: 'ATTENTIVE',  weight: 0.60, minDurationMs: 2000, maxDurationMs: 4000 },
    { to: 'IMPRESSED',  weight: 0.25, minDurationMs: 1500, maxDurationMs: 3000 },
    { to: 'THINKING',   weight: 0.15, minDurationMs: 2000, maxDurationMs: 4000 },
  ],
};

// ── State → Avatar Mapping ───────────────────────────────────────────────────

const STATE_CONFIG: Record<BehaviorState, {
  pose: string;
  gaze: string;
  emotion: string;
  microExpr: string;
  gazeJitterMultiplier: number; // How much gaze drifts in this state
}> = {
  ATTENTIVE:   { pose: 'PROFESSIONAL_DEFAULT', gaze: 'DIRECT',   emotion: 'NEUTRAL',    microExpr: 'NONE',          gazeJitterMultiplier: 0.3 },
  THINKING:    { pose: 'THINKER_POSE',         gaze: 'THINKING', emotion: 'THOUGHTFUL', microExpr: 'HEAD_TILT',     gazeJitterMultiplier: 0.8 },
  NOTING:      { pose: 'NOTE_TAKING',          gaze: 'NOTES',    emotion: 'NEUTRAL',    microExpr: 'NONE',          gazeJitterMultiplier: 0.2 },
  SKEPTICAL:   { pose: 'LEAN_BACK',            gaze: 'DIRECT',   emotion: 'SKEPTICAL',  microExpr: 'EYEBROW_RAISE', gazeJitterMultiplier: 0.1 },
  IMPRESSED:   { pose: 'LEAN_FORWARD',         gaze: 'DIRECT',   emotion: 'IMPRESSED',  microExpr: 'SUBTLE_SMILE',  gazeJitterMultiplier: 0.2 },
  WAITING:     { pose: 'OPEN_PALMS',           gaze: 'DIRECT',   emotion: 'NEUTRAL',    microExpr: 'NONE',          gazeJitterMultiplier: 0.5 },
  CROSS_CHECK: { pose: 'STEEPLED_FINGERS',     gaze: 'DIRECT',   emotion: 'CURIOUS',    microExpr: 'EYEBROW_RAISE', gazeJitterMultiplier: 0.1 },
  APPROVING:   { pose: 'PROFESSIONAL_NOD',     gaze: 'DIRECT',   emotion: 'IMPRESSED',  microExpr: 'SUBTLE_SMILE',  gazeJitterMultiplier: 0.3 },
};

// ── Weighted Random Sampler ───────────────────────────────────────────────────

function weightedRandom<T extends { weight: number }>(items: T[]): T {
  const totalWeight = items.reduce((sum, item) => sum + item.weight, 0);
  let rand = Math.random() * totalWeight;
  for (const item of items) {
    rand -= item.weight;
    if (rand <= 0) return item;
  }
  return items[items.length - 1];
}

// ── Candidate Signal → Forced Transition ─────────────────────────────────────

export type CandidateSignal =
  | 'STRONG_ANSWER'
  | 'WEAK_ANSWER'
  | 'VAGUE_ANSWER'
  | 'EYE_CONTACT_LOST'
  | 'READING_DETECTED'
  | 'NERVOUS_DETECTED'
  | 'EXCELLENT_ANSWER'
  | 'LONG_PAUSE'
  | 'SYNTHETIC_EYE_CONTACT_DETECTED'
  | 'CAMERA_COVERED_OR_ABSENT';

const SIGNAL_OVERRIDES: Record<CandidateSignal, BehaviorState> = {
  STRONG_ANSWER:      'APPROVING',
  WEAK_ANSWER:        'THINKING',
  VAGUE_ANSWER:       'SKEPTICAL',
  EYE_CONTACT_LOST:   'CROSS_CHECK',
  READING_DETECTED:   'SKEPTICAL',
  NERVOUS_DETECTED:   'ATTENTIVE',
  EXCELLENT_ANSWER:   'IMPRESSED',
  LONG_PAUSE:         'WAITING',
  SYNTHETIC_EYE_CONTACT_DETECTED: 'SKEPTICAL',
  CAMERA_COVERED_OR_ABSENT: 'CROSS_CHECK',
};

// ── Main Behavior Engine ─────────────────────────────────────────────────────

export class HRBehaviorEngine {
  private currentState: BehaviorState = 'ATTENTIVE';
  private stateTimer = 0;
  private stateDuration = 5000;
  private listeners: Array<(event: BehaviorEvent) => void> = [];
  private physicsEngine?: AvatarPhysicsEngine;
  private lastNodTime = 0;

  constructor(physicsEngine?: AvatarPhysicsEngine) {
    this.physicsEngine = physicsEngine;
    this.scheduleNextTransition();
  }

  private scheduleNextTransition() {
    const transitions = TRANSITIONS[this.currentState];
    const next = weightedRandom(transitions);
    this.stateDuration = next.minDurationMs +
      Math.random() * (next.maxDurationMs - next.minDurationMs);
    this.stateTimer = 0;
  }

  private emitEvent() {
    const cfg = STATE_CONFIG[this.currentState];
    const event: BehaviorEvent = {
      state: this.currentState,
      pose: cfg.pose,
      gaze: cfg.gaze,
      emotion: cfg.emotion,
      microExpression: cfg.microExpr,
      durationMs: this.stateDuration,
    };

    // Drive physics engine directly
    if (this.physicsEngine) {
      this.physicsEngine.setPose(cfg.pose);
      this.physicsEngine.setGaze(cfg.gaze);

      // Trigger nod on APPROVING / PROFESSIONAL_NOD
      if (cfg.pose === 'PROFESSIONAL_NOD' && Date.now() - this.lastNodTime > 2000) {
        this.physicsEngine.triggerNod();
        this.lastNodTime = Date.now();
      }
    }

    this.listeners.forEach(fn => fn(event));
  }

  update(dtMs: number) {
    this.stateTimer += dtMs;
    if (this.stateTimer >= this.stateDuration) {
      // Transition to next state
      const transitions = TRANSITIONS[this.currentState];
      const next = weightedRandom(transitions);
      this.currentState = next.to;
      this.stateDuration = next.minDurationMs +
        Math.random() * (next.maxDurationMs - next.minDurationMs);
      this.stateTimer = 0;
      this.emitEvent();
    }
  }

  // Force a behavior change based on candidate signal
  reactToSignal(signal: CandidateSignal) {
    const newState = SIGNAL_OVERRIDES[signal];
    if (newState && newState !== this.currentState) {
      this.currentState = newState;
      this.stateTimer = 0;
      const cfg = STATE_CONFIG[newState];
      this.stateDuration = 2000 + Math.random() * 3000;
      this.emitEvent();

      // Trigger physics reaction immediately
      if (this.physicsEngine) {
        this.physicsEngine.setPose(cfg.pose);
        this.physicsEngine.setGaze(cfg.gaze);
        if (signal === 'EXCELLENT_ANSWER' || signal === 'STRONG_ANSWER') {
          this.physicsEngine.triggerNod();
        }
      }
    }
  }

  // React to external avatar state (from dialogue manager)
  reactToAvatarState(emotion: string, pose: string, gaze: string) {
    if (this.physicsEngine) {
      this.physicsEngine.setPose(pose);
      this.physicsEngine.setGaze(gaze);
    }
  }

  get state(): BehaviorState { return this.currentState; }
  get config() { return STATE_CONFIG[this.currentState]; }

  onBehaviorChange(fn: (event: BehaviorEvent) => void) {
    this.listeners.push(fn);
    return () => { this.listeners = this.listeners.filter(l => l !== fn); };
  }

  setPhysicsEngine(engine: AvatarPhysicsEngine) {
    this.physicsEngine = engine;
  }
}
