# Architecture

## High-Level Architecture

```
Frontend (React + Vite)
        ↓
API Gateway (FastAPI)
        ↓
Speech Pipeline Orchestrator
        ↓
------------------------------------------
| ASR | Semantic NLP | Scoring | Feedback |
------------------------------------------
        ↓
Persistence + Response Delivery
```

## Core Components

### 1. API Layer
- FastAPI routes for evaluation, history, streaming, and authentication
- CORS configuration for frontend access
- JWT-based security and request validation

### 2. Speech Pipeline
- Orchestrates the end-to-end evaluation flow
- Handles preprocessing, inference, and response formatting

### 3. ASR Engine
- Faster-Whisper GPU inference
- Multi-pass decoding with VAD filtering

### 4. Semantic & NLP Layer
- Structured extraction (intent, profile signals)
- Resume alignment and evidence mapping

### 5. Scoring Engine
- Hybrid scoring: LLM rubric + deep learning signals
- Generates per-dimension metrics and confidence labels

### 6. Feedback Engine
- Generates positives, improvements, suggestions, and coaching summary
- Guards against contradictions and repetition

### 7. Data & History
- SQLite-backed persistence for user attempts
- History and progress endpoints for analytics

### 8. Model Manager
- Handles model loading/unloading and GPU safety
- Preloads critical models at startup

## Resource Usage

**GPU**
- ASR inference
- LLM-based reasoning

**CPU**
- Embeddings
- Feature extraction
- Persistence and API orchestration

## Design Principles

- Modular services with clear separation of concerns
- Low-latency streaming support
- Scalable model orchestration and fault tolerance
- Secure authentication and sanitized inputs
