# AlterVox

**你说的话，从另一个人的嘴里说出来——情绪是你的，声音不是。**

全 CPU 本地管线（V2 已实装）：

```
浏览器麦克风 ─(WS 16k PCM)─► 能量 VAD 断句（自适应噪声底）
  ─► Silero VAD 人声过滤（键盘声/音乐在进 ASR 前整体剔除）
  ─► ASR(SenseVoice) ─► 情绪感知(emotion2vec_plus_large)
  ─► 纠错润色(Qwen2.5-Omni-3B 文本通道，按情绪分布加表现力)
  ─► 规则兜底（英文碎片别名/ITN，确定性可审计；场景词表可会话级注入）
  ─► TTS(Qwen3-TTS + 情绪参考音频) ─(WS wav)─► 页面内播放
```

设计要点（详见 [SPEC.md](SPEC.md)）：情绪有两层接口——感知交给专用小模型；润色层按情绪分布改台词，TTS 层按 `emo_top` 用 `@spk=` 切换到该情绪的参考音频，参考音频的逐帧声学 token 参与条件生成，情绪落在音色与韵律上。默认走合规轨：VoiceDesign 用同一描述批量生成中性/开心/难过/生气/吃惊多条参考（台词贴合各自情绪，否则念不出情绪），经 0.6B base 实链复验（在该情绪自己的场景句上重复 3 次、取 emotion2vec 中位数）+ F0 双筛后入库，不达标的情绪不挂载；SFT 定制轨因身份先验压过参考韵律，不挂情绪参考。润色 prompt 只含通用规则（如"说话人只说中文，英文单词都是识别错误"），不内置场景词表与示例；每一段独立降级（情绪挂→按中立，润色挂→用原文），单点故障不拖垮整条管线。

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

## 噪声环境可调参数

嘈杂环境（室友放歌、机械键盘）实测可调：

| 参数 | 默认 | 位置 | 说明 |
| --- | --- | --- | --- |
| 排队上限 `MAX_PENDING` | **5** | `static/app.js` | 待处理录音条数上限，满则丢弃新录音并提示；噪声连环触发时防止堆几十条排队 |
| 动态阈值倍数 `VAD_NOISE_FLOOR_FACTOR` | 2.5 | `static/app.js` | 前端触发阈值 = 环境噪声底 × 此倍数，越大越不容易被噪声误触发 |
| Silero 阈值 `threshold` | 0.5 | `pipeline/vad.py` | 人声概率门槛，误拒真人声就调低（上游 Open-LLM-VTuber 用 0.4） |
| 最短人声 `speech_s < 0.3` | 0.3s | `server.py` | 过滤后人声短于此值直接丢弃 |

## 做一个属于你的声线

两条路线，先读边界（详见 [VOICE_GUIDE.md](VOICE_GUIDE.md)）：

| | 路线 A：原创声线 | 路线 B：角色声线微调 |
| --- | --- | --- |
| 声音来源 | VoiceDesign 模型合成 | 你喜欢的角色/说话人的录音 |
| 权利状态 | 干净（Apache-2.0 合成音源） | 声音属于表演者 |
| 可否分发/商用 | ✅ | ❌ 仅限本人本地自用 |

- **路线 A**（零训练）：`voice_emotion_bank.py` 一条命令完成——VoiceDesign 用同一张固定声线卡、每个情绪 × 12 seeds 批量生成候选 → 参考层身份预筛（中立 medoid 基准 + 声纹余弦 + F0 音区）→ 0.6B base 实链复验 → 情绪分/韵律判定 → 写入 `voices/emotion_bank/`（`manifest.json` + `selected/*.wav`），config 自动读取。身份优先：换人的候选情绪分再高也挡在库外；不达标的情绪不挂载，不会硬上没效果的参考。
- **路线 B**（CPU 30 分钟级）：`cut_align.py` 切句转写 → 人工校对 → `sft_train.py --mode xvec` 微调（LoRA，loss 14→2.2）→ 合并 → GGUF q8_0 → `ALTERVOX_PROFILE=self` 接入 config

微调在纯 CPU 上实测：100~200 句语料，8 epochs 约 30 分钟，推理 RTF ≈ 0.8x。

## 架构与设计文档

- [SPEC.md](SPEC.md)：完整设计文档——版本路线图、三层情绪管线、模型 bake-off 数据、未决问题清单
- [VOICE_GUIDE.md](VOICE_GUIDE.md)：声线制作教程（含权利边界）
- [ASSETS.md](ASSETS.md)：模型下载指引与路径配置
- [RELEASE.md](RELEASE.md)：发布与隐私边界说明

## 路线图

- **V1（已完成）**：ASR→TTS 直通 + Web UI + SFT 定制声线
- **V2（已实装）**：情绪层——Silero VAD 降噪 + emotion2vec_plus_large 感知 + Omni-3B 文本润色 + 规则词表纠错；合规轨接入多情绪参考音频切换，已并入 server 实测（延迟 ~12-14s/句，稳态）
- **V3（远期）**：实时对话档

## 许可

MIT（见 [LICENSE](LICENSE)）。第三方模型权重各自遵循其上游许可；角色声线微调产物的权利边界见 VOICE_GUIDE.md，请务必遵守。
