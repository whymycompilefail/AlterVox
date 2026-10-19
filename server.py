"""AlterVox V2 服务：本地后端 + Web UI。

链路（SPEC.md 5.0+ 定稿）：
    浏览器麦克风 →(WS, 16k PCM)→ ASR(SenseVoice)
      → 情绪感知(emotion2vec_plus_large) → 纠错润色(Omni-3B 文本通道)
      → 规则词表兜底 → TTS(Qwen3-TTS) →(WS, wav)→ 页面内播放

情绪有两层接口：润色层按情绪分布改台词（语气词/标点）；TTS 层按 emo_top
通过 @spk= 切换到该情绪参考音频（voices/emotion_bank/manifest.json），
参考音频的逐帧声学 token 直接参与条件生成，情绪落在音色和韵律上。
合规轨（默认）用 VoiceDesign 批量产出的原创声线；SFT 定制轨不挂参考
（身份先验会压过参考韵律），EMOTION_REFS 为空。
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
from pipeline.emotion import EmotionRecognizer
from pipeline.rewriter import OmniRewriter
from pipeline.corrector import correct
from pipeline.vad import VoiceFilter

STATIC_DIR = Path(__file__).parent / "static"

asr: SenseVoiceASR | None = None
tts: Qwen3TTS | None = None
emotion: EmotionRecognizer | None = None
rewriter: OmniRewriter | None = None
vad: VoiceFilter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, tts, emotion, rewriter, vad
    print("[startup] loading ASR (SenseVoice int8)...")
    asr = SenseVoiceASR()
    print("[startup] loading TTS (Qwen3-TTS resident engine, warmup in background)...")
    tts = Qwen3TTS()
    print("[startup] loading Silero VAD (voice filter)...")
    try:
        vad = VoiceFilter()
    except Exception as e:
        print(f"[startup] VAD 不可用，不过滤直接进 ASR: {e}")
    print("[startup] loading emotion (emotion2vec_plus_large)...")
    try:
        emotion = EmotionRecognizer()
    except Exception as e:
        print(f"[startup] emotion 不可用，降级为 V1: {e}")
    print("[startup] spawning Omni-3B text channel (llama-server, ready in bg)...")
    try:
        rewriter = OmniRewriter()
    except Exception as e:
        print(f"[startup] rewriter 不可用，降级为 V1: {e}")
    print(f"[startup] ready: http://{config.HOST}:{config.PORT}")
    yield
    # atexit 已负责杀 llama-tts / llama-server 常驻进程


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
                # 服务端上限：客户端 25s 断句之上再兜一道，防异常超长录音拖垮管线
                if audio.size > int(30 * config.ASR_SAMPLE_RATE):
                    audio = audio[:int(30 * config.ASR_SAMPLE_RATE)]

                # Silero VAD 人声过滤：键盘声/音乐/环境噪声在这里被整体剔除
                if vad is not None:
                    speech = await asyncio.to_thread(vad.speech_only, audio)
                    speech_s = len(speech) / config.ASR_SAMPLE_RATE
                    if speech_s < 0.3:
                        await ws.send_json({
                            "type": "error",
                            "message": "未检测到人声，已忽略（噪声过滤）",
                        })
                        print(f"[vad] 人声 {speech_s:.2f}s < 0.3s，丢弃")
                        continue
                    print(f"[vad] 人声 {speech_s:.2f}s / 原始 {len(audio)/config.ASR_SAMPLE_RATE:.2f}s")
                    audio = speech

                t0 = time.perf_counter()
                text = await asyncio.to_thread(asr.transcribe, audio)
                t_asr = time.perf_counter() - t0
                print(f"[asr {t_asr:.2f}s] {text}")

                if not text.strip():
                    await ws.send_json({"type": "asr", "text": text, "asr_s": round(t_asr, 2)})
                    continue

                final_text, v2 = await asyncio.to_thread(_v2_polish, audio, text)
                await ws.send_json({"type": "asr", "text": text, "asr_s": round(t_asr, 2),
                                    "polished": final_text, **v2})
                print(f"[emo] {v2.get('emotion', '-')}  ->  [polish] {final_text}")

                try:
                    wav_path, t_tts, wav_bytes = await asyncio.to_thread(
                        _synth, final_text, v2.get("emo_top"))
                    await ws.send_json({
                        "type": "audio",
                        "text": final_text,
                        "emo_ref": v2.get("emo_top", "默认"),
                        "tts_s": round(t_tts, 2),
                        "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    })
                    print(f"[tts {t_tts:.2f}s] ref={v2.get('emo_top', '默认')} "
                          f"-> {len(wav_bytes)} bytes")
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


def _v2_polish(audio, text):
    """V2 三段：情绪感知 → Omni 纠错润色 → 规则兜底。返回 (最终台词, 元信息)。

    每段独立降级：情绪失败→按中立喂给润色层；润色失败→用原始转写；
    规则兜底永远跑（对润色输出与原始转写都做确定性审计）。
    """
    info = {}

    # 1) 情绪感知（emotion2vec 分布，标签取中文半截）
    evidence = ""
    if emotion is not None:
        try:
            top, emo_s = emotion.recognize(audio, config.ASR_SAMPLE_RATE)
            evidence = " ".join(f"{p['label']}({p['score']})" for p in top)
            info["emotion"] = evidence
            info["emo_s"] = round(emo_s, 2)
            # 只在有参考音频的标签里取最高分：top-1 常是「恐惧/吃惊」这类
            # 没有参考的邻居类，会把本该走「难过」的样本拦掉。
            candidates = [(p["score"], p["label"]) for p in top
                          if p["label"] in config.EMOTION_REFS]
            if candidates:
                score, label = max(candidates)
                if score >= config.EMOTION_REF_MIN_SCORE:
                    info["emo_top"] = label
                    info["emo_top_score"] = round(score, 3)
        except Exception as e:
            print(f"[emo] 失败（按中立处理）: {e}")

    # 2) Omni-3B 文本通道纠错润色
    polished = text
    if rewriter is not None:
        try:
            polished, llm_s = rewriter.rewrite(text, evidence)
            info["rewrite_s"] = round(llm_s, 2)
        except Exception as e:
            print(f"[omni] 润色失败（用原始转写）: {e}")
            polished = text

    # 3) 规则词表兜底（确定性替换，edits 可审计）
    fixed, edits = correct(polished)
    if edits:
        info["edits"] = [list(e) for e in edits]
    return fixed.strip() or polished.strip(), info


def _synth(text: str, emo: str | None = None) -> tuple[str, float, bytes]:
    """合成并读出 wav 字节（阻塞，跑在线程池里）。emo 决定用哪条情绪参考。"""
    t0 = time.perf_counter()
    ref = config.EMOTION_REFS.get(emo) if emo else None
    path = tts.synth_to_file(text, speaker_file=ref)
    t_tts = time.perf_counter() - t0
    data = Path(path).read_bytes()
    tts.synth_cleanup(path)
    return path, t_tts, data


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")
