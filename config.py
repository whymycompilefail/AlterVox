"""AlterVox 配置。

路径规则：项目内资产相对仓库根；外部大资产（底模/llama.cpp 构建/SenseVoice 模型）
按"环境变量 > 仓库内路径 > 外部依赖根"的顺序解析。
外部依赖根默认 ./assets（见 ASSETS.md 的下载指引），可用环境变量 ALTERVOX_ASSETS 指向别处。
"""

import json
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
TTS_BASE_GGUF = _first(       # 底模量化版（合规轨 backbone / 选声扫描用）
    os.environ.get("ALTERVOX_BASE_GGUF"),
    ASSETS / "models" / "Qwen3-TTS-0.6B-q8_0.gguf",
    ASSETS / "Qwen3-TTS-0.6B-q8_0.gguf",
)

# ---------------------------------------------------------------- 运行轨
# compliant = 合规发布轨：0.6B base + VoiceDesign 原创声线 + 情绪参考库
# self      = 自用轨：角色 LoRA 定制 backbone（不可分发，参考情绪被身份先验压制）
PROFILE = os.environ.get("ALTERVOX_PROFILE", "compliant")

VOICE_BANK_DIR = ROOT / "voices" / "emotion_bank"
VOICE_BANK_MANIFEST = VOICE_BANK_DIR / "manifest.json"


def _load_voice_bank():
    """情绪库 manifest；不存在/损坏时返回空字典（服务仍可按无参考启动）。"""
    try:
        with open(VOICE_BANK_MANIFEST, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


VOICE_BANK = _load_voice_bank()

# ---------------------------------------------------------------- TTS
LLAMA_TTS_BIN = _first(
    os.environ.get("ALTERVOX_LLAMA_TTS"),
    ROOT / "llama.cpp" / "build" / "bin" / "llama-tts.exe",
    ASSETS / "llama.cpp" / "build" / "bin" / "llama-tts.exe",
)
if PROFILE == "self":
    # SFT 定制轨：仅自用。实测身份先验会压过参考韵律，故不挂情绪库
    TTS_BACKBONE = os.environ.get(
        "ALTERVOX_TTS_BACKBONE", str(ROOT / "voices" / "custom-voice-q8_0.gguf"))
    TTS_SPEAKER = os.environ.get(
        "ALTERVOX_TTS_SPEAKER",
        str(ROOT / "voices" / "sft_dataset" / "clips" / "clip_0030.wav"))
else:
    # 合规轨：0.6B base + VoiceDesign 原创声线 + 情绪参考库
    TTS_BACKBONE = os.environ.get("ALTERVOX_TTS_BACKBONE", TTS_BASE_GGUF)
    _bank_default = (VOICE_BANK.get("default") or {}).get("file")
    TTS_SPEAKER = os.environ.get(
        "ALTERVOX_TTS_SPEAKER",
        str(VOICE_BANK_DIR / _bank_default) if _bank_default else "")

TTS_MMPROJ = _first(
    os.environ.get("ALTERVOX_TTS_MMPROJ"),
    ASSETS / "models" / "mmproj-Qwen3-TTS-0.6B-q8_0.gguf",
    ASSETS / "mmproj-Qwen3-TTS-0.6B-q8_0.gguf",
)
TTS_LANG = "zh"
TTS_THREADS = 12
# 钉死的音色组合：seed=4、pitch=0（+2 起有机器人感）
# seed 决定采样轨迹。实测五情绪台词首 take 通过率：seed=2/3/5/7/8/9 只有 2~3/5，
# seed=4 与 seed=6 为 5/5，取 seed=4 作为起始，跑飞时再换 seed 重试。
TTS_SEED = 4
TTS_MAX_NEW_TOKENS = 96  # 8s 音频上限，防止跑飞
TTS_TMP_DIR = str(ROOT / "cache" / "tts")

# ---------------------------------------------------------------- V2 情绪层
EMOTION_MODEL_DIR = _first(    # emotion2vec_plus_large（funasr 格式，~2GB）
    os.environ.get("ALTERVOX_EMOTION_DIR"),
    ROOT / "emotion2vec_plus_large",
)
# 情绪参考音频：emotion2vec 命中已挂载标签时，TTS 切到该情绪的参考（@spk=）。
# 参考由 voice_emotion_bank.py 生成并实链筛选；未达标的情绪不会出现在这里。
if PROFILE == "self":
    EMOTION_REFS = {}
else:
    EMOTION_REFS = {
        label: str(VOICE_BANK_DIR / ref["file"])
        for label, ref in (VOICE_BANK.get("refs") or {}).items()
        if isinstance(ref, dict) and ref.get("file")
    }
EMOTION_REF_MIN_SCORE = 0.4   # 已挂载标签的最高分低于此值视为感知不可靠，用默认音色
LLAMA_SERVER_BIN = _first(     # Omni-3B 常驻服务用（与 llama-tts 同一构建）
    os.environ.get("ALTERVOX_LLAMA_SERVER"),
    ROOT / "llama.cpp" / "build" / "bin" / "llama-server.exe",
    ASSETS / "llama.cpp" / "build" / "bin" / "llama-server.exe",
)
OMNI_DIR = Path(os.environ.get("ALTERVOX_OMNI_DIR", ROOT / "Qwen2.5-Omni-3B-GGUF"))
OMNI_MODEL = os.environ.get(
    "ALTERVOX_OMNI_MODEL", str(OMNI_DIR / "Qwen2.5-Omni-3B-Q4_K_M.gguf"))
OMNI_MMPROJ = os.environ.get(
    "ALTERVOX_OMNI_MMPROJ", str(OMNI_DIR / "mmproj-F16.gguf"))
OMNI_HOST = "127.0.0.1"
OMNI_PORT = int(os.environ.get("ALTERVOX_OMNI_PORT", "8081"))
OMNI_THREADS = 12

# ---------------------------------------------------------------- 服务
HOST = "127.0.0.1"
PORT = int(os.environ.get("ALTERVOX_PORT", "8765"))
