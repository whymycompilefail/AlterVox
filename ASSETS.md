# 第三方资产来源与下载指引（ASSETS.md）

AlterVox 只发布源代码。所有模型权重、量化文件、外部工具按本表自行下载，
默认放到仓库根的 `assets/` 目录（或用环境变量 `ALTERVOX_ASSETS` 指向任意位置）。

| 资产 | 用途 | 来源 | 放置位置 |
| --- | --- | --- | --- |
| Qwen3-TTS-12Hz-0.6B-Base | 微调底模（HF 格式） | HuggingFace / ModelScope 官方仓库 | `assets/Qwen3-TTS-12Hz-0.6B-Base/` |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 原创声线生成（合规轨） | 同上 | `assets/Qwen3-TTS-12Hz-1.7B-VoiceDesign/` |
| Qwen3-TTS-0.6B q8_0 GGUF + mmproj | 底模推理 / 选声扫描 | 官方或自行用本仓 `llama.cpp/conversion` 转换 | `assets/Qwen3-TTS-0.6B-q8_0.gguf`、`assets/mmproj-Qwen3-TTS-0.6B-q8_0.gguf` |
| sherpa-onnx SenseVoice int8 | ASR | sherpa-onnx 官方发布页（zh-en-ja-ko-yue 2024-07-17） | `assets/sherpa-onnx-sense-voice-.../` |
| silero_vad_v5.onnx | 切句 VAD | silero 官方 v5 | `assets/silero_vad_v5.onnx` |
| emotion2vec_plus_large | 情绪感知（V2） | ModelScope `iic/emotion2vec_plus_large` | `assets/emotion2vec_plus_large/` |
| Qwen2.5-Omni-3B GGUF | 文本纠错润色（V2 定选） | 社区 GGUF 仓库（Q4~Q8） | `assets/Qwen2.5-Omni-3B-GGUF/` |
| llama.cpp CPU 特化构建 | TTS 推理引擎 | 本仓已收编 `llama.cpp/` 源码，自行 cmake 构建（见下） | `llama.cpp/build/bin/` |

## llama.cpp 构建

```bash
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j
# 产物：build/bin/llama-tts.exe、build/bin/llama-quantize.exe
```

CPU 调优说明：本仓 `llama.cpp/` 基于 upstream `c649f60`，含 CPU 特化调优
（设计文档见 [cpu-only-ai-tuber](https://github.com/whymycompilefail/cpu-only-ai-tuber) 的
`patches/` 与 `specs/`）。

## 配置覆盖

所有路径都可用环境变量覆盖（见 `config.py`）：`ALTERVOX_ASSETS`、
`ALTERVOX_SENSEVOICE_DIR`、`ALTERVOX_VAD_MODEL`、`ALTERVOX_LLAMA_TTS`、
`ALTERVOX_TTS_BACKBONE`、`ALTERVOX_TTS_MMPROJ`、`ALTERVOX_TTS_SPEAKER`、
`ALTERVOX_BASE_MODEL`、`ALTERVOX_VOICEDESIGN_MODEL`、`ALTERVOX_BASE_GGUF`。
