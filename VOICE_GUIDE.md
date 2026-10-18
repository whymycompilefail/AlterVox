# AlterVox 声线制作教程

给 AlterVox 做一个属于你的声音，有两条路线。先看边界，再动手。

## 0. 两条路线与使用边界（务必先读）

| | 路线 A：原创声线 | 路线 B：角色声线微调 |
|---|---|---|
| 声音来源 | VoiceDesign 模型**合成**的音色 | 你喜欢的角色/说话人的录音 |
| 权利状态 | 干净（Apache-2.0 合成音源） | 声音属于表演者（CV/配音演员） |
| 可否公开分发 | ✅ 可以 | ❌ **不可以** |
| 可否商用 | ✅ 可以 | ❌ **不可以** |
| 适合场景 | 直播、视频、对外发布的产品 | 纯自用、本地体验 |

**边界声明**：路线 B 产出的模型（含 LoRA、合并模型、GGUF）与训练语料**仅限本人本地使用**，不得以任何形式分发、上传或商用——声线背后是真实表演者的劳动成果。想分享或商用，请用路线 A。

## 1. 环境准备

- Python 3.11+；运行 venv：`python -m venv .venv && pip install -r requirements.txt`
- 训练 venv（独立，避免污染运行环境）：另建一个 venv，装 `torch (CPU) transformers peft qwen-tts safetensors`
- 底模：`Qwen3-TTS-12Hz-0.6B-Base`（HF 格式，约 1.9GB，下载见 ASSETS.md）
- llama.cpp（含 `llama-tts.exe`、`llama-quantize.exe`、conversion 工具，本仓已收编源码，按 ASSETS.md 构建）
- 全程 **CPU 可跑**：微调约 30 分钟级，推理 RTF ≈ 0.8x

## 2. 路线 A：VoiceDesign 原创声线（合规轨）

1. 用 VoiceDesign 模型按人设描述生成参考音频（性别、音色、气质都可指定）
2. 满意后把该 wav 存为 speaker reference，`config.py` 里指向它
3. **seed 是音色的主导变量**：改 seed = 换人；pitch 建议 0
4. 完事。不需要训练，成本为零

进阶：把生成的音频当作"目标"，也可以走路线 B 的训练流程，把这个人设**焊进权重**（先验更稳，不依赖运行时参考质量）。

## 3. 路线 B：角色声线微调（自用轨）

### 3.1 收集语料

- 目标：**3~10 分钟干净语音**（约 100~200 句）
- 要求：单说话人、无 BGM、无混响、语气多样（陈述/疑问/感叹都有）
- 格式不限，流程会统一转 24kHz 16bit

### 3.2 切句与转写

```
.venv/Scripts/python.exe cut_align.py <你的音频>
```

产出：`voices/sft_dataset/clips/*.wav` + `metadata.csv`（`file|text` 格式）。
自动流程会做 VAD 切句 + ASR 转写，并丢弃过短片段。

### 3.3 人工校对（不可跳过）

ASR 一定会错：同音字、专有名词、英文碎片。逐句对照音频校对 `metadata.csv`，
专有名词务必写对——TTS 会照着文字念，错字会永远留在声线里。

### 3.4 训练

```
.venv-qwentts/Scripts/python.exe sft_train.py --mode xvec --epochs 8
```

- LoRA 挂 talker + code_predictor 的线性层（r16/α32），CPU 每步约 1.3s
- 观察 loss：应从 ~14 降到 ~2.2（talker 收敛很快，sub-talker 慢些）
- 锚点默认选数据集中**最平稳的长句**（8s 上下最佳），可在脚本常量里改
- 结束自动合并并输出试听 `voices/sft_merged/audition_xvec.wav`，先听再往下走

### 3.5 转 GGUF 并量化

```
llama.cpp/conversion 下的 qwen3tts 转换器 → f16 GGUF
llama-quantize f16 → q8_0（约 610MB，与底模同规格）
```

### 3.6 接入

`config.py` 里把 `TTS_BACKBONE` 指向新 GGUF、`TTS_SPEAKER` 指向锚点 wav，重启 server，
用测试句回归：音色对不对、韵律自不自然、随机换几句再听。

## 4. 常见问题

- **语料最少多少？** 100 句 3 分钟起步可出效果，200 句明显更稳
- **能做多个角色吗？** 能，一角色一模型，流程完全复用
- **为什么推理还要给参考音频？** x-vector 负责指方向，权重先验负责声线本体；
  锚点质量影响不大，但选平稳长句最保险
- **训练完不像怎么办？** 先换锚点试听；再考虑加 epoch；语料语气覆盖面比时长更关键
