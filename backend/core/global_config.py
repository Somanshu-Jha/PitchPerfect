import os
import json
import logging

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "global_config.json"))

def load_global_strictness() -> str:
    """Reads the universal application strictness score."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("universal_strictness", "intermediate")
        except Exception as e:
            logger.error(f"❌ [GlobalConfig] Error reading config: {e}")
            return "intermediate"
    return "intermediate"

def save_global_strictness(strictness: str) -> bool:
    """Saves the universal application strictness from the Admin Panel."""
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        config["universal_strictness"] = strictness
        
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        logger.info(f"⚙️ [GlobalConfig] Universal Strictness updated to: {strictness.upper()}")
        return True
    except Exception as e:
        logger.error(f"❌ [GlobalConfig] Error saving config: {e}")
        return False
