from fastapi import FastAPI, UploadFile, File, HTTPException
from tools import create_rag_tool, update_retriever
from chatbot import app as app_graph
from langchain_core.messages import HumanMessage
import os
from fastapi.responses import StreamingResponse, FileResponse
from langchain_core.messages import AIMessage
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from utils import TTS, STT


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str


UPLOAD_DIR = "uploads"

@app.get("/")
def health():
    return {'Status' : 'The api is live and running'}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    update_retriever(file_path)

    return {
        "status": "success",
        "filename": file.filename
    }


@app.post("/chat")
async def chat(message: str, session_id: str = "default"):

    async def event_generator():
        async for chunk in app_graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages"
        ):
            if len(chunk) >= 1:
                message_chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                if hasattr(message_chunk, 'content') and message_chunk.content:
                    data = str(message_chunk.content).replace("\n", "\\n")
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.01)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
# ---------------- STT ---------------- #

@app.post("/stt")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        return await STT(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is empty")

        audio_path = await TTS(text=request.text)

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Audio file not created")

        return FileResponse(
            path=audio_path,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))