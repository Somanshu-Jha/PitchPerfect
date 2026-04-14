import os
import gc
import torch
import logging

# --- CTranslate2 Workaround ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)


class ModelManager:
    """
    High-Performance Model Manager with PERSISTENT model caching.
    
    CRITICAL CHANGE: Models are loaded ONCE and kept in memory forever.
    Previous behavior: unload after each request → 30-60s reload penalty.
    New behavior: load at startup → 0s reload on subsequent requests.
    
    Device Allocation:
      GPU (cuda:0):   Whisper (float16), LLM (4-bit BnB), DL Scoring FFNN
      CPU (System RAM): sentence-transformer embedder
    
    Memory Budget (RTX 5070 Ti, 12GB VRAM):
      Whisper large-v3 float16: ~3.0 GB
      LLM 4-bit: ~2.5 GB  
      DL FFNN: ~0.001 GB
      Total: ~5.5 GB → well within 12GB budget
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.active_models = {}
            cls._instance._init_hardware()
        return cls._instance

    def _init_hardware(self):
        """Called once at singleton creation. Uses ALL CPU cores dynamically."""
        _n = os.cpu_count() or 8
        os.environ["OMP_NUM_THREADS"] = str(_n)
        os.environ["MKL_NUM_THREADS"] = str(_n)
        os.environ["OPENBLAS_NUM_THREADS"] = str(_n)
        os.environ["NUMEXPR_NUM_THREADS"] = str(_n)
        try:
            torch.set_num_threads(_n)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(max(1, _n // 4))
        except RuntimeError:
            pass
        logger.info(
            f"🧵 [ModelManager] Using {_n} CPU cores: torch={torch.get_num_threads()}, "
            f"interop={torch.get_num_interop_threads()}, OMP={_n}, MKL={_n}"
        )

        # GPU capability check
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            sm = f"sm_{major}{minor}"
            logger.info(f"🖥️  [ModelManager] GPU: {device_name} | {sm}")

            if major >= 12:
                try:
                    _probe = torch.zeros(1, device="cuda:0")
                    del _probe
                    logger.info(f"✅ [ModelManager] CUDA probe passed for {sm}")
                except Exception as e:
                    logger.error(f"❌ [ModelManager] CUDA probe FAILED: {e}")
        else:
            logger.warning("⚠️ [ModelManager] No CUDA GPU — CPU fallback mode")

    # ── Model Loaders (PERSISTENT — never unloaded) ──────────────────────────

    def load_faster_whisper(self, model_id: str = "large-v3"):
        """Loads Faster-Whisper onto GPU. Cached permanently after first load."""
        if "faster_whisper" not in self.active_models:
            import time
            t = time.perf_counter()
            logger.info(f"💾 [ModelManager] Loading Faster-Whisper ({model_id}) → GPU...")
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            try:
                self.active_models["faster_whisper"] = WhisperModel(
                    model_id, device=device, compute_type=compute_type
                )
                logger.info(f"✅ [ModelManager] Whisper loaded in {time.perf_counter()-t:.1f}s")
            except Exception as e:
                logger.error(f"❌ [ModelManager] Whisper load failed: {e}")
                raise
        return self.active_models["faster_whisper"]

    def load_embedder(self, model_id: str = "all-MiniLM-L6-v2"):
        """Loads SentenceTransformer on CPU. Cached permanently."""
        if "embedder" not in self.active_models:
            import time
            t = time.perf_counter()
            logger.info(f"💾 [ModelManager] Loading Embedder {model_id} → CPU...")
            from sentence_transformers import SentenceTransformer
            self.active_models["embedder"] = SentenceTransformer(model_id, device="cpu")
            logger.info(f"✅ [ModelManager] Embedder loaded in {time.perf_counter()-t:.1f}s")
        return self.active_models["embedder"]

    def load_llm(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
        """Loads LLM via 4-bit BitsAndBytes quantization. Cached permanently."""
        if "llm" not in self.active_models:
            import time
            t = time.perf_counter()
            logger.info(f"💾 [ModelManager] Loading {model_id} → GPU/CPU (4-bit BnB)...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                # SIKHO: 'Quantization' (Model ko Pichkana).
                # 14 Billion ka model RAM me 28GB space khayega (float16).
                # `load_in_4bit`: Isko 28GB se 7GB mein pichka do. 
                # `bnb_4bit_compute_dtype`: Par math operations 16-bit me karo taki dimag weak na ho.
                # `llm_int8_enable_fp32_cpu_offload`: Agar thoda sa data bach jaye, usko CPU RAM (LaptopRAM) mein shift kar do GPU se.
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Cap GPU based on actual free VRAM (leave 1.5GiB buffer) and CPU RAM at 85%
                # SIKHO: Memory Management. CPU vs GPU Allocation.
                # `safe_vram_gb`: Agar GPU ke paas 12 GB RAM(RTX 5070 Ti) hai, to 1.5GB OS/Display kelye chorh do. Phir baqi sara(e.g., 10GB) Model ko de do!
                import psutil
                if torch.cuda.is_available():
                    free_vram, _ = torch.cuda.mem_get_info(0)
                    safe_vram_gb = max(2, int((free_vram / (1024**3)) - 1.5))
                    gpu_cap = f"{safe_vram_gb}GiB"
                else:
                    gpu_cap = "0GiB"
                
                memory_cap = {
                    0: gpu_cap,
                    "cpu": f"{int((psutil.virtual_memory().total / (1024**3)) * 0.85)}GiB"
                }
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=memory_cap
                )
                self.active_models["llm"] = {"tokenizer": tokenizer, "model": model}
                logger.info(f"✅ [ModelManager] LLM loaded in {time.perf_counter()-t:.1f}s with Memory Caps: {memory_cap}")
            except ImportError:
                logger.error("❌ [ModelManager] bitsandbytes not installed!")
                raise

        return self.active_models["llm"]

    def check_ollama(self, model_id: str = "deepseek-r1:14b"):
        """Checks if Ollama is running and has the required model."""
        import requests
        import time
        t = time.perf_counter()
        url = "http://127.0.0.1:11434/api/tags"
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                found = any(model_id in m for m in models)
                status = "✅ detected" if found else "⚠️ running (model missing)"
                logger.info(f"🧠 [ModelManager] Ollama {status} in {time.perf_counter()-t:.3f}s")
                return {"status": "online", "model_found": found, "models": models}
            return {"status": "error", "model_found": False}
        except Exception:
            logger.warning(f"❌ [ModelManager] Ollama OFFLINE (Could not connect to {url})")
            return {"status": "offline", "model_found": False}

    # ── Preloading (called at startup) ──────────────────────────────────────

    def preload_critical_models(self):
        """
        Preloads Whisper + SBERT at server startup so the first request
        doesn't pay a 30-60 second loading tax.
        """
        import time
        t_start = time.perf_counter()
        logger.info("🚀 [ModelManager] PRELOADING critical models at startup...")

        # 1. Whisper (GPU) — heaviest model, ~30s to load
        try:
            self.load_faster_whisper()
        except Exception as e:
            logger.error(f"❌ [Preload] Whisper failed: {e}")

        # 2. SBERT embedder (CPU) — used by scoring + feedback
        try:
            self.load_embedder()
        except Exception as e:
            logger.error(f"❌ [Preload] Embedder failed: {e}")

        # 3. Ollama Health Check (Right-Brain)
        self.check_ollama()

        total = time.perf_counter() - t_start
        logger.info(f"✅ [ModelManager] Preloading complete in {total:.1f}s")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"📊 [GPU] VRAM in use after preload: {vram_gb:.2f} GB")

    # ── Memory Management ──────────────────────────────────────────────────

    def unload(self, identifier: str):
        """Explicitly deletes a model. USE SPARINGLY — models should stay loaded."""
        if identifier in self.active_models:
            logger.info(f"🧹 [ModelManager] Unloading '{identifier}'...")
            del self.active_models[identifier]

    def clear(self):
        """
        Light VRAM cleanup — clears fragmented memory WITHOUT unloading models.
        Safe to call between requests.
        SIKHO: Memory Garbage Collection. 
        Jaise hum laptop restart kark RAM saaf krte hyn, vaise hi `torch.cuda.empty_cache()`
        bekar purani tensors ko flush kardeta hai bina models delete kiye! Model intact rehtay hn!
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        allocated_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        logger.info(f"📊 [GPU] Post-cleanup VRAM: {allocated_gb:.2f} GB")


# Singleton instance
model_manager = ModelManager()
