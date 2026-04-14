# Workflow

This document describes the end-to-end execution flow of PitchPerfect from input to feedback delivery.

## End-to-End Flow

1. Audio input (upload or live stream)
2. Audio preprocessing
3. Speech recognition (ASR)
4. Transcript validation
5. Semantic extraction
6. Feature extraction
7. Scoring
8. Feedback generation
9. Response delivery + storage

---

## Step 1: Audio Input
- User records or uploads audio
- Optional resume file is attached
- Payload is sent to `POST /student/evaluate`

## Step 2: Audio Preprocessing
- Normalization
- Silence trimming
- Noise handling

## Step 3: Speech Recognition
- Faster-Whisper (GPU)
- Beam search decoding
- VAD filtering

## Step 4: Transcript Validation
- Compare transcript length vs. audio duration
- Reprocess if incomplete
- Merge segments using timestamps

## Step 5: Semantic Extraction
- Intent and structured profile extraction
- Resume alignment checks

## Step 6: Feature Extraction

**Audio**
- Speech rate
- Pause ratio
- Pitch variance
- Energy stability

**Text**
- Embeddings
- Structure quality
- Vocabulary richness

## Step 7: Scoring
- Hybrid scoring model
- Combines rubric-based reasoning with DL metrics

## Step 8: Feedback Generation
- Positives, improvements, and suggestions
- Coaching summary aligned with user level
- Guarded against contradictions

## Step 9: Response Delivery
- JSON response returned to the client
- History stored for progress tracking

---

## Live Streaming Path

For real-time transcription, the frontend uses `WS /student/stream`. This path focuses on live ASR feedback and does not execute the full scoring pipeline.
