"""AlterVox 配置。

路径规则：项目内资产相对仓库根；外部大资产（底模/llama.cpp 构建/SenseVoice 模型）
按"环境变量 > 仓库内路径 > 外部依赖根"的顺序解析。
外部依赖根默认 ./assets（见 ASSETS.md 的下载指引），可用环境变量 ALTERVOX_ASSETS 指向别处。
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("ALTERVOX_ASSETS", ROOT / "assets"))


def _first(*paths):
    """返回第一个真实存在的路径，全不存在则返回最后一个（便于报错定位）。"""
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return str(paths[-1])


# ---------------------------------------------------------------- ASR
SENSEVOICE_DIR = _first(
    os.environ.get("ALTERVOX_SENSEVOICE_DIR"),
    ASSETS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
)
ASR_VAD_MODEL = _first(
    os.environ.get("ALTERVOX_VAD_MODEL"),
    ASSETS / "silero_vad_v5.onnx",
)
ASR_SAMPLE_RATE = 16000
ASR_NUM_THREADS = 2
ASR_USE_ITN = True

# ---------------------------------------------------------------- 底模资产（训练/选声用）
TTS_BASE_MODEL = _first(      # HF 底模（微调脚本用）
    os.environ.get("ALTERVOX_BASE_MODEL"),
    ASSETS / "Qwen3-TTS-12Hz-0.6B-Base",
)
VOICEDESIGN_MODEL = _first(   # VoiceDesign 1.7B（原创声线轨）
    os.environ.get("ALTERVOX_VOICEDESIGN_MODEL"),
    ASSETS / "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)
TTS_BASE_GGUF = _first(       # 底模量化版 + mmproj（选声扫描用）
    os.environ.get("ALTERVOX_BASE_GGUF"),
    ASSETS / "Qwen3-TTS-0.6B-q8_0.gguf",
)

# ---------------------------------------------------------------- TTS
LLAMA_TTS_BIN = _first(
    os.environ.get("ALTERVOX_LLAMA_TTS"),
    ROOT / "llama.cpp" / "build" / "bin" / "llama-tts.exe",
    ASSETS / "llama.cpp" / "build" / "bin" / "llama-tts.exe",
)
TTS_BACKBONE = os.environ.get(
    "ALTERVOX_TTS_BACKBONE", str(ROOT / "voices" / "custom-voice-q8_0.gguf"))
TTS_MMPROJ = os.environ.get(
    "ALTERVOX_TTS_MMPROJ", str(ASSETS / "mmproj-Qwen3-TTS-0.6B-q8_0.gguf"))
TTS_SPEAKER = os.environ.get(
    "ALTERVOX_TTS_SPEAKER", str(ROOT / "voices" / "sft_dataset" / "clips" / "clip_0030.wav"))
TTS_LANG = "zh"
TTS_THREADS = 12
# 钉死的音色组合：seed=2（改 seed = 换人）、pitch=0（+2 起有机器人感）
TTS_SEED = 2
TTS_MAX_NEW_TOKENS = 96  # 8s 音频上限，防止跑飞
TTS_TMP_DIR = str(ROOT / "cache" / "tts")

# ---------------------------------------------------------------- 服务
HOST = "127.0.0.1"
PORT = int(os.environ.get("ALTERVOX_PORT", "8765"))
