#!/usr/bin/env python3
# =====================================================================
# HR TEACHER LABELER — DeepSeek-R1:14B via Ollama
# =====================================================================
# Ye file "Teacher" hai. Har interview transcript ko DeepSeek se score
# aur feedback dilwati hai. Custom model inhi labels se seekhega.
# Checkpoint har 50 samples pe save hota hai (crash-proof).
# =====================================================================

import os, sys, json, time, re, logging, argparse, requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
INPUT_FILE = os.path.join(BASE_DIR, "hr_training_dataset.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "hr_teacher_labels.jsonl")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b")

# ── Teacher ka prompt — LEAN format (DeepSeek-R1 ka <think> block bahut tokens khata hai,
# isliye prompt chhota rakhna zaroori hai warna output empty aayega) ──
SYSTEM_PROMPT = """You are a Senior HR Manager. Score this interview pitch 0-10 on these dimensions and give feedback.
Return JSON: {"skills":0,"education":0,"projects":0,"confidence":0,"fluency":0,"structure":0,"overall":0,"pos":"positive","imp":"improvement","coach":"coaching advice"}"""


def build_user_prompt(record: dict) -> str:
    """
    Purpose: Dataset record ko prompt mein convert karna.
    Role: DeepSeek ko transcript + audio info dikhana taaki wo sahi score de sake.
    """
    parts = []
    if record.get("resume"):
        parts.append(f"[RESUME]: {record['resume'][:500]}")
    af = record.get("audio_features", {})
    parts.append(
        f"[AUDIO]: WPM={af.get('wpm_estimate',140)}, "
        f"Fluency={af.get('fluency_score',0.5):.2f}, "
        f"Fillers={af.get('filler_count',0)}, "
        f"Words={af.get('word_count',0)}"
    )
    parts.append(f"[TRANSCRIPT]: {record['transcript'][:500]}")
    return "\n".join(parts)


def call_ollama(prompt: str, timeout: int = 180) -> str:
    """
    Purpose: Ollama API se DeepSeek ko call karna.
    DL Role: Teacher model se "Gold Standard" labels lena (Knowledge Distillation).
    Numerical Impact:
      - num_predict=2048: DeepSeek-R1 pehle <think> block mein sochta hai (~500-1500 tokens),
        phir actual JSON deta hai. 500 tokens kam pad rahe the isliye empty response aa raha tha.
        2048 se <think> + JSON dono fit honge.
      - timeout=180: 14B model ko complex prompts pe 30-60s lag sakte hain.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}\n\nJSON:",
        "stream": False,
        "options": {
            "temperature": 0.0,      # 0 = deterministic, hamesha same answer (best for labeling)
            "num_predict": 2048,      # CRITICAL: Pehle 500 tha jo kam pad raha tha (<think> sab kha leta tha)
            "num_ctx": 4096,          # Input context window
            "num_gpu": 99,            # Puri GPU use karo for speed
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", "")
        logger.warning(f"Ollama HTTP {r.status_code}")
        return ""
    except requests.exceptions.Timeout:
        logger.warning("Ollama timeout!")
        return ""
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return ""


def parse_teacher_output(raw: str) -> dict:
    """
    Purpose: DeepSeek ke raw text se JSON nikalna (<think> blocks hata ke).
    DL Role: Clean data banana training ke liye.
    """
    # DeepSeek-R1 <think> blocks hatao
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    # Markdown code blocks hatao
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()
    # JSON dhundho
    brace = raw.find("{")
    if brace < 0:
        return None
    raw = raw[brace:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Matching brace dhundho
        depth = 0
        for i in range(len(raw)):
            if raw[i] == '{': depth += 1
            elif raw[i] == '}': depth -= 1
            if depth == 0:
                try: return json.loads(raw[:i+1])
                except: break
        # Last resort: missing braces add karo
        try:
            fixed = raw + '}' * (raw.count('{') - raw.count('}'))
            return json.loads(fixed)
        except:
            return None


def expand_labels(parsed: dict) -> dict:
    """
    Purpose: DeepSeek ke output ko standard training format mein badalna.
    Role: Chahe DeepSeek kisi bhi format mein de (flat ya nested), ye standardize karega.
    """
    # Handle new lean format (flat keys: skills, education, projects...)
    if "skills" in parsed and "rubric_scores" not in parsed:
        rubric = {}
        for key in ["skills", "education", "projects", "confidence", "fluency", "structure"]:
            val = parsed.get(key, 5.0)
            rubric[key] = {"score": float(val), "reasoning": ""}
        feedback = {
            "positives": [parsed.get("pos", "")],
            "improvements": [parsed.get("imp", "")],
            "coaching_summary": parsed.get("coach", "")
        }
        return {
            "rubric_scores": rubric,
            "overall_score": float(parsed.get("overall", 5.0)),
            "feedback": feedback,
        }
    
    # Handle old complex format (rubric_scores with s/r)
    rubric = parsed.get("rubric_scores", {})
    expanded = {}
    for dim, data in rubric.items():
        if isinstance(data, dict):
            expanded[dim] = {
                "score": float(data.get("s", data.get("score", 5.0))),
                "reasoning": str(data.get("r", data.get("reasoning", "")))
            }
        elif isinstance(data, (int, float)):
            expanded[dim] = {"score": float(data), "reasoning": ""}
    # Ensure feedback is a dictionary
    feedback = parsed.get("feedback", {})
    if not isinstance(feedback, dict):
        feedback = {"positives": [], "improvements": [], "coaching_summary": str(feedback)}

    return {
        "rubric_scores": expanded,
        "overall_score": float(parsed.get("overall", 5.0)),
        "feedback": {
            "positives": feedback.get("pos", feedback.get("positives", [])),
            "improvements": feedback.get("imp", feedback.get("improvements", [])),
            "coaching_summary": feedback.get("coach", feedback.get("coaching_summary", ""))
        },
    }


def run_labeling(limit: int = None):
    """
    Purpose: Poora batch labeling — har interview ko Teacher se score karwana.
    In Project: Dataset ko "Gold Standard" labels dena.
    Checkpoint: Har 50 samples pe save. Crash ho jaye to wahi se resume karega.
    """
    if not os.path.exists(INPUT_FILE):
        logger.error(f"❌ Dataset not found: {INPUT_FILE}")
        logger.error("   Run: python -m backend.ml_models.hr_dataset_generator first")
        sys.exit(1)

    # Ollama check
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            logger.error(f"❌ Model {OLLAMA_MODEL} not found"); sys.exit(1)
        logger.info(f"✅ Ollama online, model: {OLLAMA_MODEL}")
    except Exception as e:
        logger.error(f"❌ Cannot reach Ollama: {e}"); sys.exit(1)

    # Existing labels load karo (resume ke liye)
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try: existing_ids.add(json.loads(line.strip()).get("id", -1))
                except: pass
        logger.info(f"📊 Found {len(existing_ids)} existing labels (skip karenge)")

    # Input dataset load karo
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try: records.append(json.loads(line.strip()))
            except: pass
    if limit: records = records[:limit]

    to_label = [r for r in records if r.get("id", -1) not in existing_ids]
    if not to_label:
        logger.info("✅ Sab samples already labeled!"); return

    logger.info(f"🚀 Labeling {len(to_label):,} samples with {OLLAMA_MODEL}")
    logger.info(f"   Estimated time: {len(to_label) * 10 / 3600:.1f} hours (at ~10s/sample)")

    labeled = 0; failed = 0; t_total = time.perf_counter()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for idx, record in enumerate(to_label):
            t0 = time.perf_counter()
            user_msg = build_user_prompt(record)
            raw = call_ollama(user_msg)

            if not raw:
                failed += 1
                logger.warning(f"   [{idx+1}] FAILED (no response)")
                continue

            parsed = parse_teacher_output(raw)
            if not parsed:
                failed += 1
                logger.warning(f"   [{idx+1}] FAILED (parse error)")
                continue

            try:
                labels = expand_labels(parsed)
                output_record = {
                    "id": record["id"], "archetype": record.get("archetype",""),
                    "transcript": record["transcript"],
                    "audio_features": record.get("audio_features", {}),
                    "resume": record.get("resume"), "labels": labels,
                    "target_llm_text": raw  # Store RAW for high-quality reasoning training
                }
                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                labeled += 1
                elapsed = time.perf_counter() - t0
            except Exception as e:
                failed += 1
                logger.warning(f"   [{idx+1}] SKIPPED (Schema error: {e})")
                continue

            if labeled % 10 == 0:  # Save every 10 instead of 50 for more reliability
                out_f.flush()
                total_elapsed = time.perf_counter() - t_total
                rate = labeled / total_elapsed
                remaining = (len(to_label) - idx - 1) / max(rate, 0.01)
                score = labels.get("overall_score", 0)
                logger.info(
                    f"   [{labeled:,}/{len(to_label):,}] "
                    f"arch={record.get('archetype','?'):15s} | score={score:.1f} | "
                    f"{elapsed:.1f}s | rate={rate:.1f}/s | ETA={remaining/3600:.1f}h | fail={failed}"
                )

    total_time = time.perf_counter() - t_total
    logger.info(f"✅ DONE! Labeled: {labeled:,} | Failed: {failed} | Time: {total_time/3600:.1f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_labeling(limit=args.limit)
