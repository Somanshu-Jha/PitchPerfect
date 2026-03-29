# Architecture

## 🧠 High-Level Architecture

```
Frontend (React)
        ↓
API Layer (FastAPI)
        ↓
Speech Pipeline Controller
        ↓
-----------------------------------
| ASR | NLP | DL Models | Feedback |
-----------------------------------
        ↓
Database + Response
```

---

## 🔧 Core Components

### 1. Speech Pipeline

Controls the full execution flow.

### 2. ASR Engine

* Faster-Whisper (GPU)
* Multi-pass decoding

### 3. NLP Layer

* Semantic extraction
* Name correction

### 4. DL Models

* Scoring model
* English level classifier

### 5. Feedback Engine

* LLM-based generation
* Guard system

### 6. Model Manager

* Loads/unloads models dynamically
* Prevents GPU overload

---

## ⚙️ Hardware Usage

**GPU**

* ASR
* LLM inference

**CPU**

* embeddings
* feature extraction
* database

---

## 🔐 Security

* JWT authentication
* Protected routes
* Hashed passwords

---

## 📊 Design Principles

* Modular architecture
* Fault tolerance
* Low latency
* Scalable design
