#!/usr/bin/env python
"""AlterVox 选声第二步：对选中的参考音频做 seed 扫描 + 保留率测量。

复用已验证的结论：
- seed 是音色的主导变量：同一参考换 seed，音高可漂 79Hz——换 seed 等于换人。
  所以新声线必须重新盲听选 seed，选定后钉死进 config.TTS_SEED。
- 克隆只继承约 80% 时长与部分音色，保留率逐对测量（源 vs 克隆同句对比）。

用法：
  <venv>/Scripts/python.exe voice_seed_scan.py <参考音频.wav>
  （用 AlterVox 自己的 venv，librosa 已装）

产出：voices/seedscan/<名字>/seed_<N>.wav（克隆结果）+ src.wav（对照源）
听完后把「参考音频路径 + 选定 seed」填进 config.py，选声完成。
"""
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import librosa
import numpy as np
import soundfile as sf

import config

BIN = config.LLAMA_TTS_BIN
M = config.TTS_BASE_GGUF
MM = os.path.join(os.path.dirname(M) if os.path.isabs(M) else "",
                  "mmproj-Qwen3-TTS-0.6B-q8_0.gguf")
if not os.path.exists(MM):
    import pathlib
    MM = str(pathlib.Path(config.ASSETS) / "mmproj-Qwen3-TTS-0.6B-q8_0.gguf")

TEXT = "诶，你又来啦，我还以为你把我忘了呢。"
SEEDS = [1, 2, 3, 4, 5]  # 盲听候选 seed；要更多就加


def stats(path):
    y, sr = librosa.load(path, sr=None)
    f, v, _ = librosa.pyin(y, fmin=80, fmax=600, sr=sr)
    f = f[v][~np.isnan(f[v])]
    return len(y) / sr, float(np.median(f)) if len(f) else float("nan")


def main(ref_path):
    ref_path = os.path.abspath(ref_path)
    if not os.path.isfile(ref_path):
        sys.exit(f"参考音频不存在: {ref_path}")
    name = os.path.splitext(os.path.basename(ref_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "voices", "seedscan", name)
    os.makedirs(out_dir, exist_ok=True)

    # 源音频放进来，方便同句 A/B
    sf.write(os.path.join(out_dir, "src.wav"),
             *librosa.load(ref_path, sr=None), subtype="PCM_16")
    d0, f0 = stats(ref_path)
    print(f"参考: {name}  {d0:.2f}s  {f0:.0f}Hz\n", flush=True)

    print(f"{'seed':>6} {'时长':>8} {'音高':>8}   保留", flush=True)
    for seed in SEEDS:
        dest = os.path.join(out_dir, f"seed_{seed}.wav")
        subprocess.run(
            [BIN, "-m", M, "-mm", MM, "--tts-lang", "zh", "-t", "12",
             "-n", "96", "-s", str(seed), "--tts-speaker-file", ref_path,
             "-o", dest, "-p", TEXT],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(dest):
            print(f"{seed:>6} FAILED", flush=True)
            continue
        d1, f1 = stats(dest)
        print(f"{seed:>6} {d1:6.2f}s {f1:6.0f}Hz   "
              f"时长 {d1/d0*100:5.1f}%  音高 {f1/f0*100:5.1f}%", flush=True)

    print(f"\n全部在 {out_dir} —— 逐个听，选中后填 config.py：")
    print(f"  TTS_SPEAKER = r\"{ref_path}\"")
    print("  TTS_SEED    = <你选的 seed>")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法: python voice_seed_scan.py <参考音频.wav>")
    main(sys.argv[1])
