#!/usr/bin/env python
"""微调数据第一步：把整段音频切句对齐，产出 (wav, text) 训练对。

流程：silero VAD v5 切句 → SenseVoice 逐句转写 → 每句存 24kHz wav + metadata.csv
24kHz 是 Qwen3-TTS-12Hz codec 的原生采样率，训练数据统一存这个规格。

用法：<venv>/Scripts/python.exe cut_align.py <音频文件>
输出：voices/sft_dataset/clips/*.wav + metadata.csv
"""

import csv
import os
import sys
import time

import librosa
import numpy as np
import sherpa_onnx
import soundfile as sf

import config

VAD_MODEL = config.ASR_VAD_MODEL

SRC = sys.argv[1] if len(sys.argv) > 1 else None
if not SRC:
    sys.exit(__doc__)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "voices", "sft_dataset")
CLIP_DIR = os.path.join(OUT_DIR, "clips")
os.makedirs(CLIP_DIR, exist_ok=True)

TARGET_SR = 24000   # Qwen3-TTS codec 原生
PROC_SR = 16000     # VAD / ASR 用

MIN_DUR = 0.8       # 短于此丢弃（语气词孤立片段，训练价值低）
MAX_DUR = 14.0      # 长于此让 VAD 内部强制断开
MIN_CHARS = 2       # 转写太短的句子丢弃（训练文本对要成句）


def resample(y, orig, target):
    try:
        return librosa.resample(y, orig_sr=orig, target_sr=target,
                                res_type="soxr_hq")
    except Exception:
        return librosa.resample(y, orig_sr=orig, target_sr=target)


def vad_segments(y16):
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = VAD_MODEL
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.min_silence_duration = 0.4
    config.silero_vad.max_speech_duration = MAX_DUR
    config.sample_rate = PROC_SR
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)

    segs = []
    window = 512  # silero 窗口，必须按窗口喂，大块喂会误判
    for i in range(0, len(y16), window):
        vad.accept_waveform(y16[i:i + window])
        while not vad.empty():
            s = vad.front
            segs.append((s.start, s.samples))
            vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front
        segs.append((s.start, s.samples))
        vad.pop()
    return segs


def main():
    from pipeline.asr import SenseVoiceASR
    asr = SenseVoiceASR()

    y, sr = sf.read(SRC, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    print(f"源: {SRC}  {len(y)/sr:.1f}s @ {sr}Hz", flush=True)

    y16 = resample(y, sr, PROC_SR).astype(np.float32)
    segs = vad_segments(y16)
    print(f"VAD 切出 {len(segs)} 段", flush=True)

    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    kept, dropped = 0, 0
    t0 = time.time()
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["file", "text"])
        for start, samples in segs:
            dur = len(samples) / PROC_SR
            if not (MIN_DUR <= dur <= MAX_DUR):
                dropped += 1
                continue
            text = asr.transcribe(samples).strip()
            if len(text) < MIN_CHARS:
                dropped += 1
                continue
            # 原始采样率上截句（精度比在 16k 上截好），再统一到 24k
            a = int(start / PROC_SR * sr)
            b = min(int((start + len(samples)) / PROC_SR * sr), len(y))
            clip = resample(y[a:b], sr, TARGET_SR).astype(np.float32)
            name = f"clip_{kept:04d}.wav"
            sf.write(os.path.join(CLIP_DIR, name), clip, TARGET_SR,
                     subtype="PCM_16")
            w.writerow([name, text])
            kept += 1
            print(f"[{kept:3d}] {dur:5.2f}s  {text[:40]}", flush=True)

    total = kept
    print(f"\n完成: 保留 {kept} 句, 丢弃 {dropped} 段, 用时 {time.time()-t0:.0f}s")
    print(f"数据集: {OUT_DIR}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
