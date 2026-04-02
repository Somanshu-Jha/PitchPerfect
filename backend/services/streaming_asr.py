import logging
import asyncio
import numpy as np
import time
from backend.core.model_manager import model_manager
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class StreamingASRService:
    """
    Handles live streaming ASR using Faster-Whisper.
    Uses a sliding window / accumulator approach because Faster-Whisper
    doesn't natively support pure token-by-token streaming.
    """
    def __init__(self):
        self._gpu_pool = ThreadPoolExecutor(max_workers=1)
        # 16kHz audio required for Whisper
        self.sample_rate = 16000

    async def process_stream(self, websocket):
        """
        Receives raw WebM / audio chunks from websocket.
        Extracts the last 8 seconds of audio (Sliding Window), diffs it against
        the global transcript sequence to prevent broken words, and returns full context.
        """
        import pydub
        import io

        audio_buffer = bytearray()
        last_transcribe_time = time.time()
        
        # Load model once
        model = model_manager.load_faster_whisper()
        
        # Track full interview context
        global_transcript = ""
        indian_prompt = "The speaker has an Indian accent. Speech may include pauses, fillers, and informal phrasing."

        def merge_transcripts(old_text, new_text):
            if not old_text: return new_text
            if not new_text: return old_text
            
            old_clean = old_text.strip()
            new_clean = new_text.strip()
            
            if new_clean.startswith(old_clean):
                return new_clean
                
            old_words = old_clean.split()
            new_words = new_clean.split()
            
            max_overlap = min(len(old_words), len(new_words), 20)
            
            import re
            def normalize(w): return re.sub(r'[^a-zA-Z0-9]', '', w.lower())
            
            old_norm = [normalize(w) for w in old_words]
            new_norm = [normalize(w) for w in new_words]
            
            for i in range(max_overlap, 0, -1):
                if old_norm[-i:] == new_norm[:i]:
                    return " ".join(old_words[:-i] + new_words)
                    
            return old_clean + " " + new_clean

        try:
            while True:
                chunk = await websocket.receive_bytes()
                audio_buffer.extend(chunk)

                current_time = time.time()
                # Transcribe every 1.5 seconds to send live updates
                if current_time - last_transcribe_time >= 1.5 and len(audio_buffer) > 0:
                    last_transcribe_time = current_time

                    try:
                        # 1. Parse full webm into PCM, very fast natively
                        audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_buffer))
                        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0

                        # 2. SLIDING WINDOW: only feed max 8 seconds to GPU
                        # 16000 samples/sec * 8 sec = 128000 max samples
                        max_samples = 128000
                        if len(samples) > max_samples:
                            window_samples = samples[-max_samples:]
                        else:
                            window_samples = samples

                        # 3. Transcribe in separate thread to not block WS
                        loop = asyncio.get_running_loop()
                        
                        prompt_context = f"{indian_prompt} {global_transcript[-100:]}"
                        
                        segments, info = await loop.run_in_executor(
                            self._gpu_pool,
                            lambda: model.transcribe(
                                window_samples,
                                beam_size=5,
                                best_of=5,
                                temperature=0.0,
                                vad_filter=True,
                                condition_on_previous_text=True,
                                initial_prompt=prompt_context
                            )
                        )

                        live_text = " ".join(seg.text for seg in segments).strip()

                        if live_text:
                            # 4. Seamlessly merge sliding window output into global text
                            global_transcript = merge_transcripts(global_transcript, live_text)
                            await websocket.send_json({"status": "live", "text": global_transcript})

                    except Exception as e:
                        logger.warning(f"Live transcribe error: {e}")
                        pass # Ignore chunk parse errors, wait for next buffer

        except Exception as e:
            logger.info("WebSocket disconnected or finished.")
