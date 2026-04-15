# Introlytics – AI Interview Evaluation Engine

Introlytics is a production-grade AI system designed to evaluate interview responses using speech processing, deep learning, and generative AI. The platform analyzes spoken answers, extracts semantic meaning, evaluates communication quality, and generates personalized feedback.

## 🚀 Features

* 🎤 Speech-to-text using optimized Whisper (GPU)
* 🧠 Semantic understanding (LLM-based)
* 📊 Hybrid scoring (LLM + Deep Learning)
* 🗣️ Audio intelligence (confidence, fluency, pitch)
* 🎯 Adaptive feedback (based on user level)
* 🔁 Non-repetitive GenAI responses
* 🛡️ Feedback contradiction guard
* 🔐 JWT Authentication (Login/Signup)
* ⚡ GPU optimized pipeline

---

## 🧠 How It Works

1. User records or uploads audio
2. Audio is preprocessed and transcribed
3. Semantic meaning is extracted
4. Audio + text features are analyzed
5. Scoring is computed (Hybrid AI)
6. Feedback is generated dynamically
7. Results displayed in dashboard

---

## 🏗️ Tech Stack

**Backend**

* FastAPI
* PyTorch
* Faster-Whisper (CTranslate2)
* SentenceTransformers
* SQLite

**Frontend**

* React (Vite)
* Tailwind CSS

**AI/ML**

* Transformer models
* BiLSTM + Dense layers
* Custom scoring networks

---

## ⚙️ Setup

```bash
# Backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
npm install
npm run dev
```

---

## 🔐 Authentication

* JWT-based authentication
* Secure password hashing (bcrypt)
* Protected API routes

---

## 📊 System Goals

* High transcription accuracy (~97%)
* Context-aware evaluation
* Personalized feedback
* Stable real-time performance

---

## 📌 Status

System is production-ready with ongoing improvements in:

* real-time streaming
* advanced scoring models
* UX enhancements

---

## 👨‍💻 Author

Built as a production-grade AI system focusing on real-world interview evaluation.
