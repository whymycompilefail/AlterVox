#!/usr/bin/env python
"""AlterVox 选声第一步：用 VoiceDesign 从文字描述生成候选声线。

复用已验证的流程：
VoiceDesign 1.7B 在 CPU 上由「文字描述」直接合成 3-5s 参考音频——
声音是合成的，无真人主体，无授权问题（Apache-2.0）。

用法：
  <venv-qwentts>/Scripts/python.exe voice_design.py
  （需要装了 qwen_tts 包的训练 venv，模型路径见 config.VOICEDESIGN_MODEL）

改 DESIGNS 里的描述 → 跑 → 去 voices/design/ 听 → 挑中意的进第二步 voice_seed_scan.py
"""
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch

import config

torch.set_num_threads(22)

MODEL = config.VOICEDESIGN_MODEL
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "design")
os.makedirs(OUT, exist_ok=True)

# 候选参考用的统一句子：交叉文本比音高没意义，同句对比才直接
TEXT = "诶，你又来啦，我还以为你把我忘了呢。"

# ★★★ 在这里写新声线的人设描述（这是产品决策，值得认真写）★★★
# 每条描述攻击不同的风格子集，跑一轮听一遍就知道模型能听懂什么。
DESIGNS = [
    ("A_calm",   "二十五岁的年轻女性，声音温和自然，像在跟朋友聊天，"
                 "语速平稳但有声调起伏，句尾不刻板。"),
    ("B_bright", "二十岁的年轻女性，声音清亮，说话利落干脆，"
                 "语气明快，偶尔带笑意，不做作。"),
    ("C_soft",   "二十三岁的年轻女性，声音轻柔偏低，说话慢悠悠的，"
                 "有点慵懒，尾音轻，像刚睡醒在聊天。"),
]

from qwen_tts import Qwen3TTSModel  # noqa: E402

print("loading", MODEL, flush=True)
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cpu", dtype=torch.float32)
print(f"loaded in {time.time()-t0:.0f}s\n", flush=True)

for name, instruct in DESIGNS:
    t = time.time()
    try:
        wavs, sr = model.generate_voice_design(
            text=TEXT, language="Chinese", instruct=instruct)
    except Exception as e:
        print(f"{name:10} FAILED: {type(e).__name__}: {e}", flush=True)
        continue
    wav = np.asarray(wavs[0], dtype=np.float32)
    path = os.path.join(OUT, name + ".wav")
    sf.write(path, wav, sr, subtype="PCM_16")
    print(f"{name:10} {len(wav)/sr:5.2f}s  sr={sr}  {time.time()-t:5.1f}s  -> {path}",
          flush=True)

print("\n候选参考在", OUT)
print("听完后把选中的文件名传给 voice_seed_scan.py 做 seed 扫描")
