import hashlib
import logging

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Minimal in-memory cache for audio evaluation results.
    Key: SHA256(audio_bytes + user_id)
    Value: full pipeline result dict

    Prevents recomputation for repeated identical audio uploads within the same server session.
    Max 50 entries — LRU-style eviction by insertion order (oldest deleted when limit hit).
    """

    MAX_ENTRIES = 50

    def __init__(self):
        self._store: dict = {}
        logger.info("🔒 [ResultCache] In-memory cache initialized (max 50 entries).")

    def _make_key(self, audio_bytes: bytes, user_id: str) -> str:
        combined = audio_bytes + user_id.encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def get(self, audio_bytes: bytes, user_id: str):
        key = self._make_key(audio_bytes, user_id)
        if key in self._store:
            logger.info(f"⚡ [ResultCache] Cache HIT for user={user_id}. Skipping pipeline.")
            return self._store[key]
        return None

    def set(self, audio_bytes: bytes, user_id: str, result: dict):
        key = self._make_key(audio_bytes, user_id)
        if len(self._store) >= self.MAX_ENTRIES:
            # Evict oldest
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
            logger.info("🗑️ [ResultCache] Evicted oldest cache entry (limit reached).")
        self._store[key] = result
        logger.info(f"💾 [ResultCache] Cached result for user={user_id}.")


# Singleton
result_cache = ResultCache()


