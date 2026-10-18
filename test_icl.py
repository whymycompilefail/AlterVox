#!/usr/bin/env python
"""常规路线评测：底模 + 任意参考音频的完整 ICL 声音克隆（HF fp32）。

用法：<venv-qwentts>/Scripts/python.exe test_icl.py <ref1.wav> [ref2.wav ...]
参考文本自动用 SenseVoice 转写；测试文本固定，输出到 voices/icl_<ref名>.wav
"""

import os
import sys

import soundfile as sf
import torch

import config

BASE = config.TTS_BASE_MODEL
TEXT = "世界因你而瑰丽。不论何时何地，我都会回应你的期待。"


def load():
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import (Qwen3TTSConfig, Qwen3TTSForConditionalGeneration,
                                      Qwen3TTSProcessor)
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    model = AutoModel.from_pretrained(BASE, dtype=torch.float32,
                                      attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(BASE, fix_mistral_regex=True)
    return model, processor


def lookup_text(ref):
    """从数据集 metadata.csv 查参考音频的转写；查不到则要求文件名相同主名。"""
    import csv
    name = os.path.basename(ref)
    meta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "voices", "sft_dataset", "metadata.csv")
    with open(meta, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            if row["file"] == name:
                return row["text"]
    raise SystemExit(f"metadata.csv 里没有 {name}，请换数据集内的 clip 或手动补转写")


def main():
    refs = sys.argv[1:]
    if not refs:
        print(__doc__)
        sys.exit(1)

    model, processor = load()
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    wrapper = Qwen3TTSModel(model=model, processor=processor)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
    for ref in refs:
        name = os.path.splitext(os.path.basename(ref))[0]
        ref_text = lookup_text(ref)
        info = sf.info(ref)
        print(f"[{name}] {info.duration:.2f}s  ref_text: {ref_text}", flush=True)
        items = wrapper.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text)
        wavs, fs = wrapper.generate_voice_clone(text=TEXT, language="chinese",
                                                voice_clone_prompt=items)
        out = os.path.join(out_dir, f"icl_{name}.wav")
        sf.write(out, wavs[0], fs)
        print(f"  -> {out}  ({len(wavs[0]) / fs:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
