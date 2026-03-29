# backend/ml_models/train_dl_scoring.py

import pandas as pd
import torch
import os
import multiprocessing

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# ---------------- GLOBAL CONFIG ----------------
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="backend/data/hf_cache"
)


# ---------------- DATASET CLASS ----------------
class InterviewDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["score"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=128,
            # ❌ NO padding here
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ---------------- MAIN FUNCTION ----------------
def main():

    # ---------------- CPU OPTIMIZATION ----------------
    num_cores = multiprocessing.cpu_count()

    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)

    torch.set_num_threads(num_cores)

    print(f"🚀 Using {num_cores} CPU cores")

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv("backend/data/semantic_dataset_multi.csv")

    # ---------------- GENERATE SCORE ----------------
    def generate_score(row):
        return sum([
            row["introduction"],
            row["education"],
            row["skills"],
            row["experience"],
            row["career_goals"]
        ]) * 2

    df["score"] = df.apply(generate_score, axis=1)
    df["score"] = df["score"].clip(0, 10)

    # ---------------- DATASET ----------------
    dataset = InterviewDataset(df)

    # ---------------- MODEL ----------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
        cache_dir="backend/data/hf_cache"
    )

    # ---------------- DATA COLLATOR (🔥 CRITICAL FIX) ----------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---------------- TRAINING CONFIG ----------------
    training_args = TrainingArguments(
        output_dir="backend/data/models/dl_scoring",

        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,

        num_train_epochs=3,
        logging_steps=50,
        save_steps=200,

        dataloader_num_workers=2,   # 🔥 FIXED (was 16)
        dataloader_pin_memory=False,

        max_grad_norm=1.0,
        learning_rate=2e-5,

        use_cpu=True,
        report_to="none"
    )

    # ---------------- TRAINER ----------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator   # 🔥 MUST HAVE
    )

    # ---------------- TRAIN ----------------
    trainer.train()

    # ---------------- SAVE ----------------
    model.save_pretrained("backend/data/models/dl_scoring")
    tokenizer.save_pretrained("backend/data/models/dl_scoring")

    print("✅ Training complete")


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()