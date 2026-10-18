"""V1 管线冒烟测试：读一个 wav 过 ASR，再让 TTS 读识别结果。"""
import sys
import time
import wave

import numpy as np

import config
from pipeline import SenseVoiceASR, Qwen3TTS

SRC = sys.argv[1] if len(sys.argv) > 1 else None
if not SRC:
    sys.exit("用法: python test_pipeline.py <测试.wav>")

with wave.open(SRC, "rb") as w:
    sr = w.getframerate()
    n = w.getnframes()
    ch = w.getnchannels()
    sw = w.getsampwidth()
    raw = w.readframes(n)
print(f"src: {sr}Hz {ch}ch {sw*8}bit {n/sr:.1f}s")

assert sw == 2, "expect 16-bit"
data = np.frombuffer(raw, dtype=np.int16)
if ch > 1:
    data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
audio = data.astype(np.float32) / 32768.0
if sr != config.ASR_SAMPLE_RATE:
    ratio = sr / config.ASR_SAMPLE_RATE
    idx = (np.arange(int(len(audio) / ratio)) * ratio).astype(int)
    audio = audio[idx]
    sr = config.ASR_SAMPLE_RATE

t0 = time.perf_counter()
asr = SenseVoiceASR()
print(f"asr load: {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
text = asr.transcribe(audio)
t_asr = time.perf_counter() - t0
print(f"asr {t_asr:.2f}s: {text!r}")

t0 = time.perf_counter()
tts = Qwen3TTS()
print(f"tts init: {time.perf_counter()-t0:.2f}s (resident spawning, warmup in bg)")

t0 = time.perf_counter()
path = tts.synth_to_file(text or "你好，这里是腹语测试。")
t_tts = time.perf_counter() - t0
print(f"tts {t_tts:.2f}s -> {path}")

with wave.open(path, "rb") as w:
    print(f"out: {w.getframerate()}Hz {w.getnchannels()}ch {(w.getnframes()/w.getframerate()):.1f}s")
tts.synth_cleanup(path)
print("SMOKE OK")
