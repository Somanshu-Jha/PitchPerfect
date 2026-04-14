# PitchPerfect — AI Interview Evaluation Platform

PitchPerfect is an AI-driven interview evaluation platform that analyzes spoken responses, extracts structured intent, scores communication quality, and generates actionable coaching feedback. The system combines speech processing, NLP, and deep learning with a React-based dashboard for candidates and reviewers.

## Key Capabilities

- Audio upload and live transcription (WebSocket streaming)
- Semantic extraction and structured candidate profile
- Hybrid scoring (LLM rubric + deep learning signals)
- Audio intelligence (fluency, pitch variance, filler detection)
- Personalized feedback with coaching summaries
- Resume alignment insights and history tracking
- JWT-based authentication and admin controls

## System Overview

- **Architecture:** See [architecture.md](architecture.md)
- **Workflow:** See [workflow.md](workflow.md)
- **API Schema:** See [parameter_schema.md](parameter_schema.md)

## Tech Stack

**Backend**
- FastAPI
- PyTorch
- Faster-Whisper (CTranslate2)
- SentenceTransformers
- SQLite (local persistence)

**Frontend**
- React + Vite
- TypeScript
- Tailwind CSS

**AI/ML**
- Transformer-based embeddings
- FFNN-based scoring model
- LLM-driven rubric feedback

## Project Structure

```
backend/   FastAPI services, models, and pipelines
frontend/  React UI and dashboards
```

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+

### Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend currently targets `http://localhost:8000` for API calls.

## Core API Endpoints

- `POST /student/evaluate` — Evaluate audio (optional resume)
- `GET /student/progress/{user_id}` — Progress summary
- `GET /student/history/{user_id}?days=30` — Historical attempts
- `WS /student/stream` — Live ASR transcription
- `GET /student/health` — Service health check
- `POST /auth/signup` / `POST /auth/login` — Auth flows

## Roadmap & Status

- Roadmap: [developement_roadmap.md](developement_roadmap.md)
- Current status: [CURRENT_PROGRESS.md](CURRENT_PROGRESS.md)
- Implementation guidance: [implementation_plan.md](implementation_plan.md)

## Training Summary

- Model training details: [backend/data/training_summary_report.md](backend/data/training_summary_report.md)
