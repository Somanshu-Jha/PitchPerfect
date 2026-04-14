#!/usr/bin/env python3
# =====================================================================
# TRAIN HR MODEL — QLoRA Fine-Tuning of DeepSeek-R1-Distill-Qwen-1.5B
# =====================================================================
# Distills DeepSeek-R1:14b's HR reasoning into a compact 1.5B model
# that runs locally on your GPU at inference speed.
#
# Training approach: QLoRA (4-bit NF4 quantization + LoRA adapters)
#   - Base model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   - LoRA rank: 64, alpha: 128, dropout: 0.05
#   - Target modules: all attention + MLP layers
#   - VRAM: ~7-8 GB (fits RTX 5070 Ti 12.8GB)
#
# =====================================================================
# SIKHO (Educational Notes For You):
# Q: Adapter kya hota hai?
# A: Samajh lo ek badha AI model (1.5 Billion parameters) ek bohot padha-likha insaan hai jisey sab aata hai. 
# Adapter ek chota sa "Notes ka Parcha" hai jo hum uske dimaag me chipka dete hain taaki wo ek NAYA kaam 
# (jaise HR interview lena) seekh jaaye bina pichla bhule.
# 
# Q: LoRA aur QLoRA ka kya use hota hai?
# A: LoRA (Low-Rank Adaptation) ka matlab hai ki pure 1.5 Billion parameters ko train karne ki jagah, 
# hum sirf 2-3 Million naye parameters train karte hain (wo notes ka parcha). Isse training fast hoti hai 
# aur GPU memory bachti hai.
# QLoRA (Quantized LoRA) ka matlab hai base model ko "Compress" kardena (4-bit mein) taaki wo saste gaming 
# laptop k GPU (tumhare RTX 5070 Ti) me fit aa jaye warna cloud rent leni padegi.
# =====================================================================

import os
import sys
import json
import logging
import argparse
import time

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
LABELS_FILE = os.path.join(BASE_DIR, "hr_dataset_50L_labeled.jsonl") # ⬅️ Pointing to the massive 50 Lakhs (8GB) dataset
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "hr_lora_adapter")

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Ye humara raw model hai. Isme hum upar se "HR Adapter" chipkaenge.

# ═══════════════════════════════════════════════════════════════════════
# TRAINING DATA FORMATTER
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_MSG = """You are a Principal Recruiter at a FAANG (Google/Amazon/Meta) tech company. 
Evaluate this candidate's interview pitch with extreme precision. 
Analyze if their Audio Pitch matches their Resume claims. 
BE STRICT: If there is a mismatch (e.g., claiming a skill but sounding confused), penalize heavily.

Return ONLY a single valid JSON object with this exact structure:
{
  "rubric_scores": {
    "skills": {"score": 0.0, "reasoning": "Deduction reason if < 10"},
    "education": {"score": 0.0, "reasoning": "Deduction reason if < 10"},
    "projects": {"score": 0.0, "reasoning": "Deduction reason if < 10"},
    "confidence": {"score": 0.0, "reasoning": "Deduction reason if < 10"},
    "fluency": {"score": 0.0, "reasoning": "Deduction reason if < 10"},
    "structure": {"score": 0.0, "reasoning": "Deduction reason if < 10"}
  },
  "overall_score": 0.0,
  "score_deduction_reason": "Summary of WHY the candidate didn't get 10/10",
  "feedback": {
    "positives": ["point1", "point2"],
    "improvements": ["point1", "point2"],
    "coaching_summary": "Professional FAANG-style summary"
  }
}"""

def format_training_sample(record: dict) -> str:
    """Convert a labeled record into a training conversation string."""
    
    # Build user message (input) - Match exactly what the labeler used
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
    user_msg = "\n".join(parts)
    
    # Assistant message (target output) - We use the pure LLM output that the Teacher generated
    assistant_msg = record.get("target_llm_text", "")
    if not assistant_msg:
        # We skip samples without reasoning to maintain 'High Intelligence'
        return None
        
    # Format as conversation (DeepSeek chat template)
    conversation = (
        f"<|begin▁of▁sentence|>"
        f"<|System|>{SYSTEM_MSG}<|End|>\n"
        f"<|User|>{user_msg}<|End|>\n"
        f"<|Assistant|>{assistant_msg}<|end▁of▁sentence|>"
    )
    
    return conversation


def load_training_data(max_samples: int = None) -> Dataset:
    """Load labeled data into a HuggingFace Dataset."""
    
    if not os.path.exists(LABELS_FILE):
        logger.error(f"❌ Labels not found: {LABELS_FILE}")
        logger.error("   Run: python -m backend.ml_models.hr_teacher_labeler first")
        sys.exit(1)
    
    records = []
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and len(records) >= max_samples:
                break
            try:
                rec = json.loads(line.strip())
                # Only use records with labels AND reasoning blocks
                if rec.get("labels") and rec.get("target_llm_text"):
                    records.append(rec)
            except:
                pass
    
    logger.info(f"📊 Loaded {len(records):,} labeled training samples from 50L dataset")

    
    # Format for training
    texts = []
    for rec in records:
        try:
            formatted = format_training_sample(rec)
            if formatted:
                # Skip overly long samples
                if len(formatted) < 4000: # Increased limit for deep reasoning
                    texts.append(formatted)
        except Exception as e:
            continue
    
    logger.info(f"✅ Formatted {len(texts):,} training samples")
    
    return Dataset.from_dict({"text": texts})


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train(epochs: int = 3, batch_size: int = 2, grad_accum: int = 16, max_samples: int = None):
    """Run QLoRA fine-tuning."""
    
    logger.info("=" * 60)
    logger.info("🚀 HR MODEL TRAINING — QLoRA Fine-Tuning")
    logger.info(f"   Base model: {BASE_MODEL}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Effective batch: {batch_size * grad_accum}")
    logger.info("=" * 60)
    
    # ── 1. Load Dataset ──────────────────────────────────────────────
    dataset = load_training_data(max_samples)
    
    if len(dataset) < 100:
        logger.error(f"❌ Need at least 100 labeled samples, got {len(dataset)}")
        logger.error("   Run the teacher labeler first to generate labels.")
        sys.exit(1)
    
    # Train/eval split (95/5)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    logger.info(f"   Train: {len(train_data):,} | Eval: {len(eval_data):,}")
    
    # ── 2. Quantization Config (4-bit NF4) ───────────────────────────
    # Quantization ka matlab hai "Compression". DeepSeek model float16 (16-bit decimal numbers) use karta hai.
    # Hum usko 4-bit me compress kar rahe hain (nf4 format).
    # IMPACT: 1.5B parameters ka model 3GB ki jagah sirf 1GB VRAM lega. 
    # Aur 'bnb_4bit_compute_dtype=torch.bfloat16' ka matlab hai calculations wapas 16-bit me honge taaki speed aur accuracy bani rahe.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # ── 3. Load Base Model ───────────────────────────────────────────
    logger.info(f"📦 Loading {BASE_MODEL} (4-bit quantized)...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Base model params: {total_params:,}")
    
    # ── 4. LoRA Config ───────────────────────────────────────────────
    lora_config = LoraConfig(
        r=128,                    # Ultra Capacity for complex patterns
        lora_alpha=256,           # 2x rank
        lora_dropout=0.1,         # Slightly higher for regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[          # Target ALL transformer layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   LoRA trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # ── 5. Training Arguments ────────────────────────────────────────
    training_args = SFTConfig(
        dataset_text_field="text",
        max_length=1024,
        packing=False,
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,   # Windows compatibility
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_8bit",         # Memory-efficient optimizer
        max_grad_norm=0.3,
        seed=42,
    )
    
    # ── 6. Trainer ───────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args,
    )
    
    # ── 7. Train! ────────────────────────────────────────────────────
    logger.info("🏋️ Starting training...")
    t0 = time.perf_counter()
    
    train_result = trainer.train()
    
    elapsed = time.perf_counter() - t0
    logger.info(f"✅ Training complete in {elapsed/3600:.1f} hours")
    logger.info(f"   Final loss: {train_result.training_loss:.4f}")
    
    # ── 8. Save LoRA Adapter ─────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info(f"💾 LoRA adapter saved to: {OUTPUT_DIR}")
    
    # Save training metadata
    meta = {
        "base_model": BASE_MODEL,
        "lora_rank": 64,
        "lora_alpha": 128,
        "epochs": epochs,
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "final_loss": float(train_result.training_loss),
        "training_time_hours": round(elapsed / 3600, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("🎉 HR MODEL TRAINING COMPLETE")
    logger.info(f"   Adapter: {OUTPUT_DIR}")
    logger.info(f"   To use: Load base model + apply LoRA adapter")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HR reasoning model via QLoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.grad_accum, args.max_samples)
