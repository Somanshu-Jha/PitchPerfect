# Parameter Schema

## 📥 Input

```json
{
  "audio_file": "binary"
}
```

---

## 📤 Output

```json
{
  "transcript": "string",
  "confidence": 78.5,
  "semantic": {
    "name": "string",
    "education": "string",
    "skills": ["string"],
    "experience": {
      "role": "string"
    },
    "goals": "string"
  },
  "scores": {
    "clarity": 7.5,
    "completeness": 6.8,
    "structure": 7.2,
    "confidence": 6.5,
    "technical_depth": 7.0,
    "overall_score": 7.1
  },
  "feedback": {
    "positives": ["string"],
    "improvements": ["string"]
  }
}
```

---

## 📊 Internal Features

* speech_rate
* pause_ratio
* pitch_variance
* energy_consistency
* embedding_vector

---

## 🧠 Derived Parameters

* dynamic_confidence
* english_level
* semantic_flags
