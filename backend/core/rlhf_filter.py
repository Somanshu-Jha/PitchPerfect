import logging
import os
import csv
import subprocess

logger = logging.getLogger(__name__)

class RLHFFilter:
    """
    Data Quality Filter (Phase 7).
    Validates user outputs preventing RLHF poisoning. Appends safe numerical
    vectors to the scoring_dataset.csv and triggers autonomous background retraining
    if batch capacity (+50) is reached.
    """

    def __init__(self):
        self.dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "scoring_dataset.csv")
        self.batch_trigger = 50

    def validate_and_ingest(self, transcript: str, score: float, DL_features: list):
        """
        Guards the ML loop against broken audio, silent entries, or ASR artifacts.
        Only valid contextual interviews are permitted to teach the PyTorch model.
        """
        # ---------------- 1. DATA QUALITY RULES ----------------
        if not transcript or len(transcript.split()) < 5:
            logger.warning("🚫 [RLHF Filter] Rejected: Transcript too short or empty.")
            return False

        if score < 1.0 or score > 10.0:
            logger.warning("🚫 [RLHF Filter] Rejected: Hallucinated mathematical bounds.")
            return False

        if len(DL_features) != 7:
            logger.warning("🚫 [RLHF Filter] Rejected: Corrupt numerical vector.")
            return False

        # ---------------- 2. SECURE CSV APPEND ----------------
        logger.info("✅ [RLHF Filter] Audio validated. Appending vector logic to RLHF dataset.")
        try:
            with open(self.dataset_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # The PyTorch Vector structure natively ordered as 7 floats exactly.
                # Features: Completeness, Confidence, Length, Diversity, Fillers, RAG Flag, Coherence
                append_row = DL_features + [score]
                writer.writerow(append_row)

            # ---------------- 3. AUTO-RETRAIN TRIGGER ----------------
            self._check_batch_and_retrain()

            return True

        except Exception as e:
            logger.error(f"❌ [RLHF Append] Failed to write into dataset: {e}")
            return False

    def _check_batch_and_retrain(self):
        """Reads CSV row counts. If the delta between the last trained iteration is high enough, spawn background retrain."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                row_count = sum(1 for line in f)

            # We assume initial dataset shipped with ~25 entries.
            # Realistically, you store the 'last_trained_size' in a text file to check deltas.
            last_trained_file = os.path.join(os.path.dirname(__file__), "..", "data", "last_retrain_count.txt")
            
            last_count = 0
            if os.path.exists(last_trained_file):
                with open(last_trained_file, "r") as cf:
                    last_count = int(cf.read().strip())

            if row_count - last_count >= self.batch_trigger:
                logger.warning(f"🔄 [RLHF Target Reached] Triggering Auto-Retrain! (+{row_count - last_count} samples)")
                
                # Update tracker immediately to prevent runaway spawns
                with open(last_trained_file, "w") as cf:
                    cf.write(str(row_count))
                    
                # Spaun subprocess for incremental train (Non-blocking)
                script_path = os.path.join(os.path.dirname(__file__), "..", "ml_models", "train_scoring_model.py")
                executor = os.path.join(os.path.dirname(__file__), "..", "..", "venv", "Scripts", "python.exe")
                
                subprocess.Popen([executor, script_path], shell=True)

        except Exception as e:
            logger.error(f"❌ [RLHF Trigger] Failed to spawn training daemon: {e}")

rlhf_filter = RLHFFilter()
