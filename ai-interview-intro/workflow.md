# Workflow

This document explains the complete execution flow of Introlytics from input to output.

## 🔁 End-to-End Flow

1. Audio Input
2. Audio Preprocessing
3. Speech Recognition
4. Transcript Validation
5. Semantic Extraction
6. Feature Extraction
7. Scoring
8. Feedback Generation
9. Response Delivery

---

## 🎤 Step 1: Audio Input

* User records or uploads audio
* File is sent to backend API

---

## 🧹 Step 2: Audio Preprocessing

* Normalization
* Silence trimming
* Noise handling

---

## 🧠 Step 3: Speech Recognition

* Faster-Whisper (GPU)
* Beam search decoding
* VAD filtering

---

## 🔄 Step 4: Transcript Validation

* Compare transcript length vs audio duration
* Reprocess if incomplete
* Merge segments using timestamps

---

## 🔍 Step 5: Semantic Extraction

* LLM extracts:

  * name
  * education
  * skills
  * experience
  * goals

---

## 📊 Step 6: Feature Extraction

Audio:

* speech rate
* pause ratio
* pitch variance
* energy stability

Text:

* embeddings
* structure
* vocabulary richness

---

## ⚖️ Step 7: Scoring

Hybrid scoring:

* 60% LLM (context understanding)
* 40% DL model (objective signals)

---

## 💬 Step 8: Feedback Generation

* Generated via LLM
* Adapted based on user level
* Guarded against contradictions
* Non-repetitive

---

## 📤 Step 9: Response

* JSON response returned
* Displayed in frontend dashboard
