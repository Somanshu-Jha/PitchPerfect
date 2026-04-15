from fastapi import APIRouter, WebSocket
from backend.services.correction_service import StreamingEvaluationService

router = APIRouter(prefix="/student", tags=["Streaming"])

stream_service = StreamingEvaluationService()


@router.websocket("/live-interview")
async def live_interview(ws: WebSocket):

    await ws.accept()

    await stream_service.start_session()

    while True:

        audio_chunk = await ws.receive_bytes()

        result = stream_service.process_audio_chunk(audio_chunk)

        if result:
            await ws.send_json(result)