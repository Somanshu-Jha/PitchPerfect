/**
 * PhoneDetector.ts — throttled MediaPipe ObjectDetector loop (~1 fps).
 * EfficientDet-Lite0 (COCO 80 classes) — only "cell phone" matters.
 * Loads from CDN like CandidateObserver's FaceLandmarker; degrades silently.
 */
export class PhoneDetector {
  private detector: any = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastVideoTime = -1;

  constructor(private onDetection: (score: number, nowMs: number) => void) {}

  async start(video: HTMLVideoElement): Promise<boolean> {
    try {
      const { ObjectDetector, FilesetResolver } = await import(
        /* @vite-ignore */
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
      ) as any;
      const fileset = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      this.detector = await ObjectDetector.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
          delegate: 'GPU',
        },
        scoreThreshold: 0.4,
        runningMode: 'VIDEO',
        maxResults: 5,
      });
    } catch (e) {
      console.warn('[PhoneDetector] ObjectDetector failed to load:', e);
      return false;
    }
    this.timer = setInterval(() => {
      if (!this.detector || video.readyState < 2) return;
      if (video.currentTime === this.lastVideoTime) return;
      this.lastVideoTime = video.currentTime;
      try {
        const res = this.detector.detectForVideo(video, performance.now());
        for (const det of res?.detections ?? []) {
          const cat = det.categories?.[0];
          if (cat?.categoryName === 'cell phone' && cat.score >= 0.4) {
            this.onDetection(cat.score, Date.now());
          }
        }
      } catch { /* single-frame failures are fine at 1 fps */ }
    }, 1000);
    return true;
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    try { this.detector?.close?.(); } catch { /* noop */ }
    this.detector = null;
  }
}
