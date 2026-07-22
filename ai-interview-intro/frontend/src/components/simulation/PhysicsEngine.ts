/**
 * PhysicsEngine.ts — Spring-Mass Physics + Smooth Noise for Avatar Realism
 * -------------------------------------------------------------------------
 * Drives ALL avatar motion with physically believable inertia, damping,
 * and organic micro-variation. No CSS animations — pure physics math.
 */

// ── Spring Physics (Critically Damped Spring) ──────────────────────────────

export class SpringPhysics {
  private pos: number;
  private vel: number;
  private readonly stiffness: number;
  private readonly damping: number;

  constructor(initialPos = 0, stiffness = 180, damping = 22) {
    this.pos = initialPos;
    this.vel = 0;
    this.stiffness = stiffness;
    this.damping = damping;
  }

  update(target: number, dt = 0.016): number {
    const force = this.stiffness * (target - this.pos) - this.damping * this.vel;
    this.vel += force * dt;
    this.pos += this.vel * dt;
    return this.pos;
  }

  get position(): number { return this.pos; }
  get velocity(): number { return this.vel; }
  set position(v: number) { this.pos = v; }
  reset(pos = 0) { this.pos = pos; this.vel = 0; }
}

// ── Smooth Noise (Cosine-Interpolated, Non-Repeating for Human Timescales) ─

export class SmoothNoise {
  private readonly values: Float32Array;
  private readonly size: number;

  constructor(size = 512, seed = Math.random()) {
    this.size = size;
    this.values = new Float32Array(size);
    // Seeded LCG for reproducible-ish noise
    let s = (seed * 0x7fffffff) | 0;
    for (let i = 0; i < size; i++) {
      s = (s * 1664525 + 1013904223) & 0x7fffffff;
      this.values[i] = (s / 0x7fffffff) * 2 - 1;
    }
  }

  sample(t: number): number {
    const scaled = ((t % 1) + 1) % 1 * this.size;
    const i0 = Math.floor(scaled) % this.size;
    const i1 = (i0 + 1) % this.size;
    const f = scaled - Math.floor(scaled);
    // Smoothstep (ease-in-out)
    const smooth = f * f * (3 - 2 * f);
    return this.values[i0] * (1 - smooth) + this.values[i1] * smooth;
  }

  // Multi-octave noise for more organic feel
  fractal(t: number, octaves = 3): number {
    let result = 0;
    let amplitude = 1;
    let frequency = 1;
    let totalAmp = 0;
    for (let i = 0; i < octaves; i++) {
      result += this.sample(t * frequency) * amplitude;
      totalAmp += amplitude;
      amplitude *= 0.5;
      frequency *= 2.1;
    }
    return result / totalAmp;
  }
}

// ── Blink Controller ────────────────────────────────────────────────────────

export class BlinkController {
  private blinkProgress = 0;
  private isBlinking = false;
  private blinkPhase: 'closing' | 'holding' | 'opening' | 'idle' = 'idle';
  private phaseTimer = 0;
  private lastBlinkTime = 0;
  private nextBlinkDelay: number;
  private pendingDoubleBlink = false;

  private readonly CLOSE_DUR = 0.075; // seconds
  private readonly HOLD_DUR  = 0.035;
  private readonly OPEN_DUR  = 0.110;

  constructor() {
    this.nextBlinkDelay = 2 + Math.random() * 3;
  }

  update(dt: number, time: number): number {
    // Schedule auto-blink
    if (!this.isBlinking && time - this.lastBlinkTime > this.nextBlinkDelay) {
      this.triggerBlink(time);
    }

    if (!this.isBlinking) return 0;

    this.phaseTimer += dt;

    switch (this.blinkPhase) {
      case 'closing': {
        this.blinkProgress = Math.min(1, this.phaseTimer / this.CLOSE_DUR);
        if (this.phaseTimer >= this.CLOSE_DUR) {
          this.blinkPhase = 'holding';
          this.phaseTimer = 0;
        }
        break;
      }
      case 'holding': {
        this.blinkProgress = 1;
        if (this.phaseTimer >= this.HOLD_DUR) {
          this.blinkPhase = 'opening';
          this.phaseTimer = 0;
        }
        break;
      }
      case 'opening': {
        this.blinkProgress = 1 - Math.min(1, this.phaseTimer / this.OPEN_DUR);
        if (this.phaseTimer >= this.OPEN_DUR) {
          this.blinkProgress = 0;
          this.isBlinking = false;
          this.blinkPhase = 'idle';
          // Trigger double-blink?
          if (this.pendingDoubleBlink) {
            this.pendingDoubleBlink = false;
            setTimeout(() => this.triggerBlink(Date.now() / 1000), 250);
          }
        }
        break;
      }
    }
    return this.blinkProgress;
  }

  triggerBlink(time: number) {
    this.isBlinking = true;
    this.blinkPhase = 'closing';
    this.phaseTimer = 0;
    this.blinkProgress = 0;
    this.lastBlinkTime = time;
    this.nextBlinkDelay = 2.5 + Math.random() * 4.5;
    // 15% double-blink
    if (Math.random() < 0.15 && !this.pendingDoubleBlink) {
      this.pendingDoubleBlink = true;
    }
  }
}

// ── Master Avatar Physics State ─────────────────────────────────────────────

export interface AvatarPhysicsState {
  headRotX: number;       // Nodding (degrees)
  headRotY: number;       // Side tilt (degrees)
  headRotZ: number;       // Forward/back lean (degrees)
  shoulderY: number;      // Breathing offset (px)
  gazeX: number;          // Eye X offset (-1 to 1)
  gazeY: number;          // Eye Y offset (-1 to 1)
  blinkProgress: number;  // 0 = eyes open, 1 = fully closed
  mouthOpenness: number;  // 0 = closed, 1 = fully open
  breathScale: number;    // Subtle chest scale (0.998 to 1.002)
}

export type PoseTarget = {
  headRotX: number;
  headRotY: number;
  headRotZ: number;
  gazeX: number;
  gazeY: number;
};

export const POSE_TARGETS: Record<string, PoseTarget> = {
  PROFESSIONAL_DEFAULT: { headRotX: 0,    headRotY: 0,   headRotZ: 0,    gazeX: 0,   gazeY: 0 },
  LEAN_FORWARD:         { headRotX: -2,   headRotY: 0,   headRotZ: 0.5,  gazeX: 0,   gazeY: 0.1 },
  LEAN_BACK:            { headRotX: 2,    headRotY: 0,   headRotZ: -0.5, gazeX: 0,   gazeY: -0.1 },
  THINKER_POSE:         { headRotX: -1,   headRotY: 1.5, headRotZ: 0,    gazeX: 0.3, gazeY: -0.3 },
  PROFESSIONAL_NOD:     { headRotX: -3,   headRotY: 0,   headRotZ: 0,    gazeX: 0,   gazeY: 0 },
  STEEPLED_FINGERS:     { headRotX: 1,    headRotY: 0,   headRotZ: 0,    gazeX: 0,   gazeY: 0 },
  OPEN_PALMS:           { headRotX: -1,   headRotY: 0,   headRotZ: 0,    gazeX: 0,   gazeY: 0 },
  NOTE_TAKING:          { headRotX: 3,    headRotY: -1,  headRotZ: 0,    gazeX: -0.4, gazeY: 0.5 },
};

export const GAZE_TARGETS: Record<string, { x: number; y: number }> = {
  DIRECT:   { x: 0,     y: 0 },
  THINKING: { x: 0.4,   y: -0.5 },
  NOTES:    { x: -0.5,  y: 0.7 },
  DRIFT:    { x: 0.2,   y: 0.1 },
};

// ── Main Physics Engine ─────────────────────────────────────────────────────

export class AvatarPhysicsEngine {
  // Head DOF springs
  private headX = new SpringPhysics(0, 100, 16);
  private headY = new SpringPhysics(0, 80,  14);
  private headZ = new SpringPhysics(0, 60,  12);

  // Gaze springs (fast, snappy like eyes)
  private gazeXSpring = new SpringPhysics(0, 380, 28);
  private gazeYSpring = new SpringPhysics(0, 380, 28);

  // Shoulder (slow, heavy)
  private shoulderSpring = new SpringPhysics(0, 25, 7);

  // Mouth
  private mouthSpring = new SpringPhysics(0, 220, 24);

  // Noise generators (different seeds for independent channels)
  private noiseHeadX  = new SmoothNoise(512, 0.13);
  private noiseHeadY  = new SmoothNoise(512, 0.47);
  private noiseGazeX  = new SmoothNoise(512, 0.71);
  private noiseGazeY  = new SmoothNoise(512, 0.89);

  // Blink
  private blinker = new BlinkController();

  // Breathing
  private breathPhase = 0;
  private breathRate = 0.267; // ~16 breaths/min

  // Time
  private time = 0;

  // Targets
  private targetHeadX = 0;
  private targetHeadY = 0;
  private targetHeadZ = 0;
  private targetGazeX = 0;
  private targetGazeY = 0;
  private targetMouth = 0;

  update(dt: number): AvatarPhysicsState {
    this.time += dt;
    this.breathPhase += dt * this.breathRate;

    // Micro-tremor (imperceptible organic motion — very subtle)
    const microHeadX = this.noiseHeadX.fractal(this.time * 0.04) * 0.6;
    const microHeadY = this.noiseHeadY.fractal(this.time * 0.03) * 0.4;
    const microGazeX = this.noiseGazeX.fractal(this.time * 0.12) * 0.12;
    const microGazeY = this.noiseGazeY.fractal(this.time * 0.10) * 0.08;

    // Breathing (sinusoidal, modulates shoulder + subtle head)
    const breathCycle = Math.sin(this.breathPhase * Math.PI * 2);
    const breathY = breathCycle * 0.18;
    const breathHeadX = Math.sin(this.breathPhase * Math.PI * 2 + 0.3) * 0.15;

    // Update springs
    const headRotX  = this.headX.update(this.targetHeadX + microHeadX + breathHeadX, dt);
    const headRotY  = this.headY.update(this.targetHeadY + microHeadY, dt);
    const headRotZ  = this.headZ.update(this.targetHeadZ, dt);
    const shoulderY = this.shoulderSpring.update(breathY, dt);
    const gazeX     = this.gazeXSpring.update(this.targetGazeX + microGazeX, dt);
    const gazeY     = this.gazeYSpring.update(this.targetGazeY + microGazeY, dt);
    const mouthOpenness = this.mouthSpring.update(this.targetMouth, dt);

    // Blink
    const blinkProgress = this.blinker.update(dt, this.time);

    // Breath scale (very subtle, ~0.1% chest expansion)
    const breathScale = 1 + breathCycle * 0.001;

    return {
      headRotX, headRotY, headRotZ,
      shoulderY,
      gazeX, gazeY,
      blinkProgress,
      mouthOpenness,
      breathScale,
    };
  }

  setPose(poseName: string) {
    const target = POSE_TARGETS[poseName] || POSE_TARGETS.PROFESSIONAL_DEFAULT;
    this.targetHeadX = target.headRotX;
    this.targetHeadY = target.headRotY;
    this.targetHeadZ = target.headRotZ;
    this.targetGazeX = target.gazeX;
    this.targetGazeY = target.gazeY;
  }

  setGaze(gazeName: string) {
    const g = GAZE_TARGETS[gazeName] || GAZE_TARGETS.DIRECT;
    this.targetGazeX = g.x;
    this.targetGazeY = g.y;
  }

  setGazeDirect(x: number, y: number) {
    this.targetGazeX = x;
    this.targetGazeY = y;
  }

  setMouthTarget(openness: number) {
    this.targetMouth = Math.max(0, Math.min(1, openness));
  }

  triggerBlink() {
    this.blinker.triggerBlink(this.time);
  }

  triggerNod() {
    // Quick nod: push head down then let spring return
    const prevTarget = this.targetHeadX;
    this.targetHeadX = prevTarget - 4;
    setTimeout(() => { this.targetHeadX = prevTarget + 1; }, 200);
    setTimeout(() => { this.targetHeadX = prevTarget; }, 450);
  }
}
