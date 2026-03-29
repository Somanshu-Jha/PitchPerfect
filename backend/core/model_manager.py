import os
import gc
import torch
import logging

# --- CTranslate2 Workaround ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import pipeline, AutoProcessor, AutoModelForCTC

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Sequential Model Manager enforcing strict VRAM budget on the RTX 5070 Ti (12 GB, sm_120).

    Device Allocation:
      GPU (cuda:0):    Whisper, LLM (4-bit BnB), DL Scoring FFNN
      CPU (System RAM): wav2vec2, sentence-transformer embedder, RAG + DB (no model needed)

    Threading:
      torch.set_num_threads(8)          — CPU-bound models (wav2vec2, embedder)
      OMP_NUM_THREADS / MKL_NUM_THREADS — set here as safety anchor (also set in speech_pipeline)

    CUDA Safety:
      Performs sm_120 capability check at singleton init.
      Warns clearly if PyTorch nightly cu128 is not installed.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.active_models = {}
            cls._instance._init_hardware()
        return cls._instance

    def _init_hardware(self):
        """Called once at singleton creation. Checks GPU capability and enforces thread limits."""
        # ── CPU Threading (safety anchor) ─────────────────────────────────────
        # NOTE: torch thread counts can only be set once before parallel work starts.
        # speech_pipeline.py sets them at module level first; these calls are a no-op
        # if already configured, wrapped defensively to avoid RuntimeError on reload.
        os.environ.setdefault("OMP_NUM_THREADS", "8")
        os.environ.setdefault("MKL_NUM_THREADS", "8")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
        try:
            torch.set_num_threads(8)
        except RuntimeError:
            pass  # Already set by speech_pipeline module-level call
        try:
            torch.set_num_interop_threads(4)
        except RuntimeError:
            pass  # Already set — can only be called once before parallel work starts
        logger.info(
            f"🧵 [ModelManager] Thread config: torch={torch.get_num_threads()}, "
            f"interop={torch.get_num_interop_threads()}, OMP=8, MKL=8"
        )

        # ── GPU Capability Check ───────────────────────────────────────────────
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            sm = f"sm_{major}{minor}"
            logger.info(f"🖥️  [ModelManager] GPU detected: {device_name} | Capability: {sm}")

            if major >= 12:
                # sm_120 = RTX 5000 series (Blackwell). Requires PyTorch nightly cu128.
                try:
                    # Probe: if a tiny tensor op runs on CUDA without error, nightly is working.
                    _probe = torch.zeros(1, device="cuda:0")
                    del _probe
                    logger.info(f"✅ [ModelManager] CUDA kernel probe passed for {sm}. Nightly build confirmed.")
                except Exception as e:
                    logger.error(
                        f"❌ [ModelManager] CUDA probe FAILED for {sm}: {e}\n"
                        "⚠️  Your current PyTorch does NOT support sm_120 (RTX 5070 Ti / Blackwell).\n"
                        "    Fix: pip install --pre torch torchvision torchaudio "
                        "--index-url https://download.pytorch.org/whl/nightly/cu128"
                    )
        else:
            logger.warning(
                "⚠️  [ModelManager] No CUDA GPU detected. All GPU models will fall back to CPU. "
                "Performance will be severely degraded."
            )

    # ── Model Loaders ──────────────────────────────────────────────────────────

    def load_faster_whisper(self, model_id: str = "large-v3"):
        """Loads Faster-Whisper-Large-v3 onto GPU in float16 for maximum speed + precision."""
        if "faster_whisper" not in self.active_models:
            logger.info(f"💾 [ModelManager] Loading Faster-Whisper ({model_id}) → GPU (float16)...")
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            try:
                self.active_models["faster_whisper"] = WhisperModel(model_id, device=device, compute_type=compute_type)
                logger.info("✅ [ModelManager] Faster-Whisper loaded into VRAM.")
            except Exception as e:
                logger.error(f"❌ [ModelManager] Faster-Whisper load failed: {e}")
                raise
        return self.active_models["faster_whisper"]

    def load_wav2vec(self, model_id: str = "facebook/wav2vec2-base-960h"):
        """Loads Wav2Vec2 strictly onto CPU RAM to avoid VRAM contention with Whisper."""
        if "wav2vec" not in self.active_models:
            logger.info(f"💾 [ModelManager] Loading {model_id} → CPU RAM...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForCTC.from_pretrained(model_id)
            self.active_models["wav2vec"] = {
                "processor": processor,
                "model": model.to("cpu")   # Explicitly pinned to CPU
            }
            logger.info("✅ [ModelManager] Wav2Vec2 loaded into System RAM.")
        return self.active_models["wav2vec"]

    def load_llm(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        """Loads LLM via 4-bit BitsAndBytes quantization (≈2.5 GB VRAM)."""
        if "llm" not in self.active_models:
            logger.info(f"💾 [ModelManager] Loading {model_id} → GPU (4-bit BnB)...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                self.active_models["llm"] = {"tokenizer": tokenizer, "model": model}
                logger.info("✅ [ModelManager] LLM loaded in 4-bit on GPU.")
            except ImportError:
                logger.error("❌ [ModelManager] bitsandbytes not installed! Cannot load LLM in 4-bit.")
                raise

        return self.active_models["llm"]

    def load_embedder(self, model_id: str = "all-MiniLM-L6-v2"):
        """Loads SentenceTransformer strictly on CPU to preserve zero VRAM footprint."""
        if "embedder" not in self.active_models:
            logger.info(f"💾 [ModelManager] Loading Embedder {model_id} → CPU RAM...")
            from sentence_transformers import SentenceTransformer
            self.active_models["embedder"] = SentenceTransformer(model_id, device="cpu")
            logger.info("✅ [ModelManager] Embedder locked into System RAM.")
        return self.active_models["embedder"]

    # ── Memory Management ──────────────────────────────────────────────────────

    def unload(self, identifier: str):
        """Explicitly deletes a model's memory references."""
        if identifier in self.active_models:
            logger.info(f"🧹 [ModelManager] Unloading '{identifier}'...")
            del self.active_models[identifier]

    def clear(self):
        """CRITICAL: Force GC + purge PyTorch CUDA allocators after each request."""
        logger.info("🔥 [ModelManager] Purging VRAM and invoking Garbage Collection...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        allocated_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        logger.info(f"📊 [GPU STATUS] Current allocated: {allocated_gb:.2f} GB")


# Singleton instance
model_manager = ModelManager()
