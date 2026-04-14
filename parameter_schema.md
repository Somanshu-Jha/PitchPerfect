# Parameter Schema

## Input (Form Data)

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `file` | audio file | ✅ | Primary interview audio |
| `resume` | file | Optional | Resume for alignment insights |
| `user_id` | string | Optional | Defaults to `local_demo` |
| `strictness` | string | Optional | `beginner` / `intermediate` / `advanced` |

Example (multipart):
```http
POST /student/evaluate
```

## Output (JSON)

```json
{
  "user_id": "string",
  "raw_transcript": "string",
  "refined_transcript": "string",
  "semantic": {
    "structured": {},
    "intent": {},
    "confidence_map": {},
    "evidence_map": {}
  },
  "audio_features": {},
  "audio_flags": {},
  "fillers": [],
  "filler_stats": {},
  "scores": {
    "overall_score": 0,
    "details": {}
  },
  "feedback": {
    "positives": ["string"],
    "improvements": ["string"],
    "suggestions": ["string"],
    "coaching_summary": "string"
  },
  "resume_alignment": {
    "matched": [],
    "missed": []
  },
  "processing_time": 0,
  "timings": {},
  "english_level": "string",
  "confidence": {
    "transcript_confidence": 0,
    "dynamic_confidence": 0,
    "confidence_label": "string"
  }
}
```

## Notes

- The API returns additional diagnostic fields (e.g., timings and rubric breakdowns) to aid observability.
- Fields may expand as new scoring dimensions and feedback modules are added.
