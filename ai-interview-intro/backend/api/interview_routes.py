# =====================================================================
# INTERVIEW & SHADOW MODE API ROUTES
# =====================================================================
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

from fastapi import APIRouter, UploadFile, File, Form, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from backend.interview.question_bank import get_modes, get_questions, get_random_question
from backend.agents.dialogue_manager import dialogue_manager
from backend.services.tts_service import tts_service
from fastapi import Request

router = APIRouter(prefix="/interview", tags=["interview"])


@router.get("/modes")
async def list_modes():
    """Get all available interview modes."""
    return JSONResponse(content={"modes": get_modes()})


@router.get("/questions/{mode}")
async def list_questions(mode: str, difficulty: str = None):
    """Get questions for a specific mode."""
    questions = get_questions(mode, difficulty)
    return JSONResponse(content={"mode": mode, "questions": questions})


@router.get("/random/{mode}")
async def random_question(mode: str = "self_intro"):
    """Get a random question from a mode."""
    question = get_random_question(mode)
    return JSONResponse(content={"question": question})


# ── TTS Routes ──────────────────────────────────────────────────────────────

@router.post("/tts")
async def synthesize_speech(
    text: str = Form(...),
    voice_profile: str = Form("friendly_hr"),
    speed: float = Form(1.0),
    pitch: float = Form(1.0),
    pause_before_ms: int = Form(0),
):
    """
    Convert text to human-quality speech using Edge TTS (Zero Cost).
    Returns MP3 audio bytes directly.
    """
    # Map voice profile to Edge TTS natural male voices
    voice_profile_lower = voice_profile.lower()
    if "strict" in voice_profile_lower:
        voice = "en-US-ChristopherNeural"
    elif "faang" in voice_profile_lower or "tech" in voice_profile_lower:
        voice = "en-US-ChristopherNeural"
    elif "behavioral" in voice_profile_lower:
        voice = "en-US-AndrewNeural"
    elif "startup" in voice_profile_lower:
        voice = "en-US-GuyNeural"
    else: # friendly_hr or default
        voice = "en-US-GuyNeural"
        
    # Convert speed float to Edge TTS rate string (e.g., "+0%", "+10%", "-5%")
    rate_val = int((speed - 1.0) * 100)
    rate_str = f"{rate_val:+d}%"

    # Convert pitch float to Edge TTS pitch string
    pitch_val = int((pitch - 1.0) * 100)
    pitch_str = f"{pitch_val:+d}Hz"
    
    audio_bytes = await tts_service.generate_audio(
        text, voice=voice, rate=rate_str, pitch=pitch_str
    )
    
    if not audio_bytes:
        return JSONResponse(
            content={"error": "TTS unavailable."},
            status_code=503,
        )
        
    # Kokoro returns WAV ("RIFF..."), Edge TTS returns MP3 — label correctly
    media_type = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/tts/setup")
async def setup_tts():
    """Initialize the cloud-native Edge TTS engine."""
    result = tts_service.setup()
    return JSONResponse(content=result)


@router.get("/tts/status")
async def tts_status():
    """Check TTS availability and available voice profiles."""
    return JSONResponse(content={
        "available": tts_service.is_available,
        "voices": tts_service.get_voice_profiles(),
    })


# ── Shadow Mode Routes ──────────────────────────────────────────────────

shadow_router = APIRouter(prefix="/shadow", tags=["shadow"])


@shadow_router.post("/evaluate")
async def evaluate_shadow(
    file: UploadFile = File(...),
    target_sentence: str = Form(...),
    strictness: str = Form("intermediate"),
):
    """
    Compare a shadow mode recording against a target sentence.
    User reads improved version → records → we compare accuracy + delivery.
    """
    # Save uploaded audio
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "shadow.webm")
    content = await file.read()
    with open(audio_path, "wb") as f:
        f.write(content)

    try:
        # Quick transcription
        from backend.services.transcription_service import TranscriptionService
        from backend.services.filler_detection_service import FillerDetectionService
        
        transcriber = TranscriptionService()
        segments = list(transcriber.transcribe_stream(audio_path))
        spoken_text = " ".join(seg[0] for seg in segments).strip()
        
        filler_svc = FillerDetectionService()
        filler_stats = filler_svc.detect_with_stats(spoken_text)
        
        # Compare with target
        from backend.core.model_manager import model_manager
        import numpy as np
        
        embedder = model_manager.load_embedder()
        target_vec = embedder.encode([target_sentence])[0]
        spoken_vec = embedder.encode([spoken_text])[0]
        
        denominator = np.linalg.norm(target_vec) * np.linalg.norm(spoken_vec)
        similarity = float(np.dot(target_vec, spoken_vec) / denominator) if denominator else 0.0
        
        # Scoring
        accuracy_score = round(similarity * 10, 1)
        filler_penalty = min(2.0, filler_stats.get("count", 0) * 0.3)
        final_score = max(1.0, min(10.0, accuracy_score - filler_penalty))
        
        feedback = []
        if similarity >= 0.9:
            feedback.append("Excellent match! Your delivery closely mirrors the improved version.")
        elif similarity >= 0.75:
            feedback.append("Good attempt! Most key phrases were captured correctly.")
        elif similarity >= 0.5:
            feedback.append("Partial match. Practice reading the sentence more carefully before recording.")
        else:
            feedback.append("Low match. Try reading the target sentence aloud several times before recording.")
        
        if filler_stats.get("count", 0) > 0:
            feedback.append(f"Detected {filler_stats['count']} filler(s). Try to deliver without hesitation.")
        
        return JSONResponse(content={
            "target_sentence": target_sentence,
            "spoken_text": spoken_text,
            "similarity": round(similarity, 3),
            "accuracy_score": accuracy_score,
            "final_score": final_score,
            "filler_stats": filler_stats,
            "feedback": feedback,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(audio_path)
            os.rmdir(temp_dir)
        except:
            pass


@router.post("/simulate/respond")
async def simulate_audio_response(
    file: UploadFile = File(...),
    history: str = Form("[]"),
    user_response_hint: str = Form(""),
    difficulty: str = Form("intermediate"),
    mode: str = Form("hr"),
    resume_text: str = Form(""),
    pitch_text: str = Form(""),
    covered_topics: str = Form("[]"),
    use_premium_model: str = Form("false"),
    candidate_metrics: str = Form("{}"),
):
    """
    Transcribe a candidate's simulation answer, then generate the next turn.
    """
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, file.filename or "simulation_response.webm")

    # Determine premium mode per-request (no global state mutation)
    use_premium = use_premium_model.lower() == "true"

    try:
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        fallback_hint = (user_response_hint or "").strip()
        content_type = (file.content_type or "").lower()
        filename = (file.filename or "").lower()
        can_attempt_asr = content_type.startswith("audio/") or filename.endswith((".webm", ".wav", ".mp3", ".ogg", ".m4a"))

        try:
            parsed_history = json.loads(history)
            if not isinstance(parsed_history, list):
                parsed_history = []
        except json.JSONDecodeError:
            parsed_history = []

        try:
            parsed_covered = json.loads(covered_topics)
            if not isinstance(parsed_covered, list):
                parsed_covered = []
        except json.JSONDecodeError:
            parsed_covered = []

        try:
            parsed_metrics = json.loads(candidate_metrics)
            if not isinstance(parsed_metrics, dict):
                parsed_metrics = {}
        except json.JSONDecodeError:
            parsed_metrics = {}

        import asyncio

        async def get_transcript():
            if not can_attempt_asr:
                return ""
            try:
                from backend.services.transcription_service import TranscriptionService
                res_dict = await asyncio.to_thread(TranscriptionService().transcribe, audio_path)
                return res_dict.get("transcription", "") if isinstance(res_dict, dict) else ""
            except Exception as e:
                logger.error(f"Transcription Error: {e}")
                return ""

        async def get_next_turn_with_flags(text_to_use, is_silence=False):
            return await asyncio.to_thread(
                dialogue_manager.generate_next_turn,
                parsed_history,
                text_to_use,
                difficulty=difficulty,
                mode=mode,
                resume_text=resume_text,
                pitch_text=pitch_text,
                covered_topics=parsed_covered,
                use_premium=use_premium,
                candidate_metrics=parsed_metrics,
                is_vad_pause=is_silence
            )

        if fallback_hint:
            # Parallel execution: LLM uses fast browser hint, ASR processes in background for accurate transcript
            is_silence = not fallback_hint.strip()
            transcript, result = await asyncio.gather(get_transcript(), get_next_turn_with_flags(fallback_hint, is_silence))
            user_response = transcript.strip() or fallback_hint
        else:
            # Sequential execution: No hint, must wait for ASR first
            transcript = await get_transcript()
            user_response = transcript.strip()
            
            is_silence = not user_response
            if is_silence:
                user_response = "I need a moment to answer that clearly."
                
            result = await get_next_turn_with_flags(user_response, is_silence)

        result["transcript"] = user_response
        return JSONResponse(content=result)
    finally:
        try:
            os.remove(audio_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

@router.post("/simulate/next")
async def simulate_next_turn(request: Request):
    """
    Handles turn-taking for the roleplay simulation.
    Takes history and latest user response.
    """
    data = await request.json()
    history = data.get("history", [])
    user_response = data.get("user_response", "")
    difficulty = data.get("difficulty", "intermediate")
    mode = data.get("mode", "HR")
    
    result = dialogue_manager.generate_next_turn(history, user_response, difficulty, mode)
    return result

# ── Zero-Latency Streaming WebSocket ──────────────────────────────────────

@router.websocket("/ws/stream")
async def websocket_simulation_endpoint(websocket: WebSocket):
    """
    Zero-latency WebSocket endpoint with proctoring, per-answer scoring and
    turn sequencing. One ProctorSession per connection.
    """
    await websocket.accept()
    import asyncio
    import random
    from backend.interview.proctor_session import ProctorSession
    from backend.services.answer_scorer import answer_scorer
    from backend.services.report_generator import report_generator

    session = None
    session_difficulty = "intermediate"
    session_mode = "hr"
    session_job_role = "Software Engineer"
    last_history = []

    async def speak(text: str, difficulty: str):
        """Stream one TTS utterance (used by interjects and termination)."""
        voice = ("en-US-ChristopherNeural"
                 if difficulty.lower() in ["advanced", "extreme", "faang"]
                 else "en-US-GuyNeural")
        await websocket.send_json({"type": "status", "message": "speaking"})
        audio_buffer = bytearray()
        async for chunk in tts_service.stream_audio(text, voice=voice, rate="+0%", pitch="+0Hz"):
            if chunk:
                audio_buffer.extend(chunk)
        if audio_buffer:
            await websocket.send_bytes(bytes(audio_buffer))
        await websocket.send_json({"type": "audio_end"})

    async def send_final_report():
        await websocket.send_json({"type": "status", "message": "generating_report"})
        snapshot = session.snapshot() if session else {
            "integrity_score": 100, "violations": [], "answers": [],
            "terminated": False, "termination_reason": None}
        report = await asyncio.to_thread(
            report_generator.generate, last_history, snapshot,
            session_job_role, session_difficulty, session_mode)
        await websocket.send_json({"type": "final_report", "data": report,
                                   "integrity": report.get("integrity", {})})

    while True:
        try:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # ── keep-alive pings (Cloudflare tunnel timeout prevention) ──
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # ── proctor events: absorbed by ledger, NEVER a full LLM turn ──
            if msg_type == "proctor_event":
                if session is None:
                    continue
                result = session.record_event(
                    event_type=data.get("event_type", ""),
                    confidence=float(data.get("confidence", 0.0)),
                    ts_ms=int(data.get("ts_ms", 0)),
                    candidate_speaking=bool(data.get("candidate_speaking", False)),
                    meta=data.get("meta") or {},
                )
                if result is None:
                    continue
                await websocket.send_json({"type": "proctor_action",
                                           "action": result["action"],
                                           "event_type": result["event_type"]})
                if result["action"] == "TERMINATE":
                    closing = ("I'm going to stop the interview here. "
                               f"{result['warning_text']} "
                               "Integrity matters more than any single answer — "
                               "your report will explain what happened. Goodbye.")
                    await speak(closing, session_difficulty)
                    await websocket.send_json({"type": "session_terminated",
                                               "reason": result["event_type"],
                                               "closing_text": closing})
                    await send_final_report()
                elif result["deliver_now"]:
                    await speak(result["warning_text"], session_difficulty)
                continue

            # ── end call → final report ──────────────────────────────
            if msg_type == "end_call":
                await send_final_report()
                continue

            # ── normal turn (vad_pause / kickoff) ────────────────────
            history = data.get("history", [])
            user_response = data.get("user_response", "")
            difficulty = data.get("difficulty", "intermediate")
            mode = data.get("mode", "HR")
            resume_text = data.get("resume_text", "")
            job_role = data.get("job_role", "Software Engineer")
            covered_topics = data.get("covered_topics", [])
            use_premium_mode = data.get("use_fast_mode", False)
            selected_llm = data.get("selected_llm", None)
            turn_id = int(data.get("turn_id", 0))

            session_difficulty, session_mode, session_job_role = difficulty, mode, job_role
            if session is None:
                session = ProctorSession(difficulty)
                await websocket.send_json({
                    "type": "session_config",
                    "policy": {"key": session.policy["key"],
                               "dwell_multiplier": session.policy["dwell_multiplier"]},
                    "difficulty": difficulty})
            if session.terminated:
                continue

            # Send immediate acknowledgment to frontend to trigger "Thinking" state
            await websocket.send_json({"type": "status", "message": "thinking"})

            is_kickoff = data.get("is_kickoff", False)
            if is_kickoff:
                # ZERO LATENCY GREETING BYPASS
                greetings = [
                    "Hello, welcome to PitchPerfect AI! I'll be your HR recruiter today. Let's get started — could you please introduce yourself and tell me a bit about your background?",
                    "Good morning! I'm very excited to speak with you today. To kick things off, why don't you walk me through your experience?",
                    "Hi there, it's great to meet you. I'll be conducting your interview today. Could you start by giving me a brief overview of your professional journey?",
                    "Welcome! Let's dive right in. I'd love to hear more about you — could you introduce yourself and explain why you're interested in this role?",
                    "Hi, thanks for making the time today. Before we get into specifics, tell me a little about yourself and what you've been working on lately.",
                    "Welcome aboard — I've been looking forward to this conversation. Why don't you start by telling me about yourself, in your own words?",
                    "Good to see you. I like to keep these conversations fairly relaxed — so to begin, give me the short version of your story so far.",
                    "Hello! Let's make this feel like a conversation, not an interrogation. First things first — who are you, and what drives you professionally?",
                    "Thanks for joining me today. I've got your application in front of me, but I'd much rather hear it from you — walk me through your background.",
                    "Hi, welcome. Let's start simple: introduce yourself, and tell me about one piece of work you're genuinely proud of.",
                    "Great to meet you. Before I ask anything specific, set the stage for me — where are you in your career right now, and how did you get here?",
                    "Welcome — settle in. To open things up, tell me about yourself and what kind of role you're hoping this turns into."
                ]
                result = {
                    "feedback": "",
                    "next_question": random.choice(greetings),
                    "should_end": False,
                    "topic": "self_intro",
                    "reasoning": "Initial greeting",
                    "avatar_state": {
                        "emotion": "FRIENDLY",
                        "pose": "OPEN_PALMS",
                        "gaze": "DIRECT",
                        "micro_expression": "HEAD_TILT"
                    },
                    "vocal_params": {
                        "pitch_multiplier": 1.05,
                        "speed_multiplier": 1.0,
                        "pause_before_ms": 0,
                        "filler_prefix": ""
                    }
                }
                score_result = None
                last_history = []
            else:
                is_vad_pause = msg_type == "vad_pause"
                proctor_context = session.consume_pending_prompt_notes()
                last_question = ""
                for t in reversed(history):
                    if t.get("role") == "assistant":
                        last_question = t.get("content", "")
                        break

                turn_coro = asyncio.to_thread(
                    dialogue_manager.generate_next_turn,
                    history, user_response, difficulty, mode,
                    resume_text=resume_text, pitch_text=job_role,
                    covered_topics=covered_topics,
                    is_vad_pause=is_vad_pause,
                    use_premium=use_premium_mode,
                    selected_llm=selected_llm,
                    proctor_context=proctor_context,
                )
                score_coro = asyncio.to_thread(
                    answer_scorer.score, last_question, user_response,
                    job_role, difficulty)
                result, score_result = await asyncio.gather(turn_coro, score_coro)
                session.record_answer_score(turn_id, last_question, user_response, score_result)
                last_history = [*history, {"role": "user", "content": user_response}]

            result["integrity_score"] = session.integrity_score()
            await websocket.send_json({"type": "turn_result", "turn_id": turn_id, "data": result})
            if score_result is not None:
                await websocket.send_json({"type": "answer_score", "turn_id": turn_id,
                                           "data": score_result})

            feedback = result.get("feedback", "")
            next_q = result.get("next_question", "")
            combined_text = f"{feedback} {next_q}".strip()

            vocal_params = result.get("vocal_params", {})
            speed_mult = vocal_params.get("speed_multiplier", 1.0)
            pitch_mult = vocal_params.get("pitch_multiplier", 1.0)
            rate_pct = int((speed_mult - 1.0) * 100)
            rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
            pitch_hz = int((pitch_mult - 1.0) * 50)
            pitch_str = f"+{pitch_hz}Hz" if pitch_hz >= 0 else f"{pitch_hz}Hz"
            voice = ("en-US-ChristopherNeural"
                     if difficulty.lower() in ["advanced", "extreme", "faang"]
                     else "en-US-GuyNeural")

            if combined_text:
                await websocket.send_json({"type": "status", "message": "speaking"})
                audio_buffer = bytearray()
                async for chunk in tts_service.stream_audio(combined_text, voice=voice,
                                                            rate=rate_str, pitch=pitch_str):
                    if chunk:
                        audio_buffer.extend(chunk)
                if audio_buffer:
                    await websocket.send_bytes(bytes(audio_buffer))
                await websocket.send_json({"type": "audio_end"})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected gracefully.")
            break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

