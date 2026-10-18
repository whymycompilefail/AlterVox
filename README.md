# AlterVox

**你说的话，从另一个人的嘴里说出来——情绪是你的，声音不是。**

全 CPU 本地语音管线：ASR（SenseVoice）→ 情绪感知（emotion2vec）→ 文本纠错润色 → TTS（Qwen3-TTS，SFT 定制声线）。无 GPU、无云端、无遥测。

```
浏览器麦克风 ─(WS 16k PCM)─► ASR ─► 情绪/纠错润色 ─► TTS ─(WS wav)─► 页面内播放
```

防漏音设计：麦克风音频只进本地管线；直播软件抓浏览器源时只能抓到网页播放的合成音频，你的原声不出本地。

## 快速开始

```bash
git clone https://github.com/whymycompilefail/AlterVox.git
cd AlterVox
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
```

1. 按 [ASSETS.md](ASSETS.md) 下载模型与 llama.cpp 构建（默认放 `assets/`）
2. 启动服务：`.venv/Scripts/python server.py`，浏览器开 `http://127.0.0.1:8765`
3. 授权麦克风即自动进入侦听，说话即回复合成语音

路径不合适？全部环境变量可覆盖，见 `config.py`。

## 做一个属于你的声线

两条路线，先读边界（详见 [VOICE_GUIDE.md](VOICE_GUIDE.md)）：

| | 路线 A：原创声线 | 路线 B：角色声线微调 |
| --- | --- | --- |
| 声音来源 | VoiceDesign 模型合成 | 你喜欢的角色/说话人的录音 |
| 权利状态 | 干净（Apache-2.0 合成音源） | 声音属于表演者 |
| 可否分发/商用 | ✅ | ❌ 仅限本人本地自用 |

- **路线 A**（零训练）：`voice_design.py` 文字描述生成候选 → `voice_seed_scan.py` 盲听选 seed → 填进 config
- **路线 B**（CPU 30 分钟级）：`cut_align.py` 切句转写 → 人工校对 → `sft_train.py --mode xvec` 微调（LoRA，loss 14→2.2）→ 合并 → GGUF q8_0 → 接入 config

微调在纯 CPU 上实测：100~200 句语料，8 epochs 约 30 分钟，推理 RTF ≈ 0.8x。

## 架构与设计文档

- [SPEC.md](SPEC.md)：完整设计文档——版本路线图、三层情绪管线、模型 bake-off 数据、未决问题清单
- [VOICE_GUIDE.md](VOICE_GUIDE.md)：声线制作教程（含权利边界）
- [ASSETS.md](ASSETS.md)：模型下载指引与路径配置
- [RELEASE.md](RELEASE.md)：发布与隐私边界说明

## 路线图

- **V1（已完成）**：ASR→TTS 直通 + Web UI + SFT 定制声线
- **V2（架构已定稿）**：情绪层——emotion2vec_plus_large 感知 + 小模型文本润色 + 规则词表纠错
- **V3（远期）**：实时对话档

## 许可

MIT（见 [LICENSE](LICENSE)）。第三方模型权重各自遵循其上游许可；角色声线微调产物的权利边界见 VOICE_GUIDE.md，请务必遵守。
