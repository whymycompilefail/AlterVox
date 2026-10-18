"""AlterVox V1 服务：本地后端 + Web UI。

链路（SPEC.md 5.0 节）：
    浏览器麦克风 →(WS, 16k PCM)→ ASR(SenseVoice) → 文字 → TTS(Qwen3-TTS) →(WS, wav)→ 页面内播放

防漏音设计：麦克风只进管线；直播软件抓浏览器源仅取网页播放的合成音频。
"""

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import Qwen3TTS, SenseVoiceASR

STATIC_DIR = Path(__file__).parent / "static"

asr: SenseVoiceASR | None = None
tts: Qwen3TTS | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, tts
    print("[startup] loading ASR (SenseVoice int8)...")
    asr = SenseVoiceASR()
    print("[startup] loading TTS (Qwen3-TTS resident engine, warmup in background)...")
    tts = Qwen3TTS()
    print(f"[startup] ready: http://{config.HOST}:{config.PORT}")
    yield
    # atexit 已负责杀 llama-tts 常驻进程


app = FastAPI(lifespan=lifespan)


def pcm16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def ws_pipeline(ws: WebSocket):
    """一次连接处理多句话。客户端消息：

    {"type": "utterance", "pcm16_b64": ..., "sample_rate": 16000}   一段完整录音
    {"type": "text", "text": ...}                                    直接文本合成（校对/测试用）
    """
    await ws.accept()
    print("[ws] client connected")
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            mtype = msg.get("type")

            if mtype == "utterance":
                pcm = base64.b64decode(msg["pcm16_b64"])
                audio = pcm16_to_float32(pcm)
                if audio.size < int(0.2 * config.ASR_SAMPLE_RATE):
                    await ws.send_json({"type": "error", "message": "录音太短"})
                    continue

                t0 = time.perf_counter()
                text = await asyncio.to_thread(asr.transcribe, audio)
                t_asr = time.perf_counter() - t0
                await ws.send_json({"type": "asr", "text": text, "asr_s": round(t_asr, 2)})
                print(f"[asr {t_asr:.2f}s] {text}")

                if not text.strip():
                    continue
                try:
                    wav_path, t_tts, wav_bytes = await asyncio.to_thread(_synth, text)
                    await ws.send_json({
                        "type": "audio",
                        "text": text,
                        "tts_s": round(t_tts, 2),
                        "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    })
                    print(f"[tts {t_tts:.2f}s] -> {len(wav_bytes)} bytes")
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"TTS 失败: {e}"})

            elif mtype == "text":
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                try:
                    wav_path, t_tts, wav_bytes = await asyncio.to_thread(_synth, text)
                    await ws.send_json({
                        "type": "audio",
                        "text": text,
                        "tts_s": round(t_tts, 2),
                        "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    })
                    print(f"[tts {t_tts:.2f}s] (direct text) -> {len(wav_bytes)} bytes")
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"TTS 失败: {e}"})

    except WebSocketDisconnect:
        print("[ws] client disconnected")


def _synth(text: str) -> tuple[str, float, bytes]:
    """合成并读出 wav 字节（阻塞，跑在线程池里）。"""
    t0 = time.perf_counter()
    path = tts.synth_to_file(text)
    t_tts = time.perf_counter() - t0
    data = Path(path).read_bytes()
    tts.synth_cleanup(path)
    return path, t_tts, data


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")
