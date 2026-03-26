from pathlib import Path
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(BASE_DIR, "data", "models")

SEMANTIC_MODEL_PATH = os.path.join(MODEL_DIR, "semantic_model.pkl")

DEFAULT_PARAMETER_FILE = os.path.join(BASE_DIR, "data", "default_parameters.json")

AUDIO_SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 60

WHISPER_MODEL_SIZE = "large-v3"