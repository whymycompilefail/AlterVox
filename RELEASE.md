# AlterVox 开源发布执行手册（RELEASE.md）

> 目标：把 AlterVox 整理成可公开push的仓库，对齐 cpu-only-ai-tuber 的既有发布范式。
> 原则（拍板 2026-10-18）：**MIT 许可；仅发布源代码；CV 语音数据与其衍生品永不入仓；角色声线模型不发布，对外文档一律用"喜欢的角色声线"通用表述，不出现任何具体角色/作品名。**
> 铁律（来自 cpu-only-ai-tuber 的实测教训）：**.gitignore 必须在 `git init` + 首次 `git add` 之前就位**——gitignore 不能撤回已提交的文件，而改写已发布历史不影响已经 clone 的 fork。

---

## 步骤 0：边界确认（已完成，留档）

| 决策 | 内容 |
| --- | --- |
| LICENSE | MIT（与 cpu-only-ai-tuber 一致） |
| 发布物 | 源代码 + 文档；**一切权重/录音/编译产物除外** |
| CV 数据 | `voices/sft_dataset/`（录音+转写）及其衍生模型（sft_merged*、*.gguf）**永不发布** |
| 对外文档 | 不出现具体角色名；教程按"用户自己的目标声线"通用表述（见 VOICE_GUIDE.md） |
| llama.cpp | CPU 特化 fork，源码树**随主仓收编**（同 cpu-only-ai-tuber 2026-10-11 拍板），build/ 排除 |

## 步骤 1：.gitignore 先行

新建 `D:\AlterVox\.gitignore`，内容如下（三类来源：照抄 cpu-only-ai-tuber 的通用块、AlterVox 新增的 CV 数据块、llama.cpp 收编块）：

```gitignore
# ---- 隐私与本机痕迹（永远最先写）----
.workbuddy/
.trae/
*.log
__pycache__/
*.pyc
.venv/
.venv-*/

# ---- CV 语音数据与衍生品（本仓最高危类别）----
# 录音切片+转写=表演者劳动成果；合并模型/LoRA/GGUF=衍生品。全部永不发布。
voices/
cute_girl.wav
cute_girl.mp4

# ---- 第三方权重（既定边界：仅源码，权重一律不发）----
emotion2vec_plus_large/
emotion2vec_base_finetuned/
**/*.gguf
**/*.onnx
**/*.safetensors
**/*.bin

# ---- llama.cpp fork：收编源码，排除三件套 ----
# build/ 含 CMakeCache（Windows 用户名）与 compile_commands.json（本机绝对路径）；
# models/ 是上游测试 vocab（.gguf）；.git.fork-backup/ 是停放的上游历史。
llama.cpp/build/
llama.cpp/build*/
llama.cpp/out/
llama.cpp/models/
llama.cpp/.git.fork-backup/

# ---- 运行产物与评测存档 ----
# v2_poc = 模型选型 bake-off 的测试环境（测试集/实验脚本/报告），只用于决定"用哪个模型"，
# 与项目本体无关，结论已沉淀进 SPEC.md——本地存档，不入仓。
v2_poc/
```

## 步骤 2：代码去本地化

硬编码 `D:\...` 绝对路径全部改为 `config.py` 常量或 CLI 参数，`config.py` 本身改为"相对仓库根 + 环境变量覆盖"：

| 文件 | 处理 |
| --- | --- |
| `config.py` | 模型/服务路径改为相对根目录 + `os.environ.get()` 覆盖；新增 `ALTERVOX_ASSETS` 一类的外部依赖根变量 |
| `cut_align.py` | `ROOT`/`VAD_MODEL` 常量改从 config 导入 |
| `sft_train.py` | `BASE`/`DATA` 等常量改从 config 导入 |
| `test_icl.py` | 同上 |
| `v2_poc/*.py` | PoC 存档性质，仅检查无敏感信息即可，不强求参数化 |

验证：`Grep "D:\\\\" --glob "*.py"` 应零命中（v2_poc 除外可放宽）。

## 步骤 3：文档重写

| 文档 | 动作 |
| --- | --- |
| `README.md` | **新建**：项目定位（ASR→情绪→TTS 全 CPU 管线）、架构图（引 SPEC.md）、快速开始、声线制作教程链接、LICENSE |
| `ASSETS.md` | **新建**：第三方资产来源与下载方式（Qwen3-TTS-Base、emotion2vec_plus_large、Qwen2.5-Omni GGUF、SenseVoice ONNX、MiniCPM/Qwen3.5 备选、llama.cpp 上游基线 commit `c649f60`） |
| `VOICE_GUIDE.md` | 重写：去 D 盘路径 → "git clone 后 pip install -r requirements.txt" 流程；确认无角色名 |
| `requirements.txt` | **新建**：主依赖清单（funasr、sherpa-onnx、soundfile、librosa、pypinyin、psutil、peft、torch-cpu 等） |
| `SPEC.md` | 随仓发布（步骤 6 已拍板：A），但**角色/作品名已全部通用化**——本表仅在本地留存原始记录 |

## 步骤 4：llama.cpp 收编

**复制**（不动原目录——正在运行的服务还指着它）：

```powershell
robocopy <你的llama.cpp源码根> llama.cpp /E /XD build out .git .git.fork-backup models /XF *.log
```

- `/XD` 排除：build（隐私源：CMakeCache 含 Windows 用户名与 compile_commands.json 含本机绝对路径）、上游测试 vocab、停放 git 历史
- 验证：源码侧文件数应 ≈3,300+；构建说明见 ASSETS.md
- CPU 调优的正式补丁与设计文档在 cpu-only-ai-tuber 仓库的 `patches/` 与 `specs/`

## 步骤 5：隐私扫描（push 前最后一道闸）

扫描命令里的占位符（`<你的用户名>`、`<本机资产根>`、`<角色名>`）填你自己的真实值，
**占位符保证名单本身不入仓**：

```powershell
# 1. 用户名/本机路径泄漏
git grep -I "<你的用户名>" || echo OK
git grep -Ii "<本机资产根>" -- "*.py" "*.md" || echo OK
# 2. 媒体/权重误入
git ls-files | Select-String "\.(wav|mp4|gguf|onnx|safetensors)$" || echo OK
# 3. 角色/作品名泄漏（对外文档通用性）：扫描你用过的全部角色名与英文名，
#    名单不上墙——发布前手动执行，零命中才可 push
git grep -i "<角色名>\|<角色英文名>\|<作品名>" -- "*.md" "*.py" || echo OK
# 4. 文件数 sanity（cpu-only-ai-tuber 教训：实测过 72,497 文件事故）
git ls-files | Measure-Object -Line    # 预期 ~4,000-6,000
```

## 步骤 6：SPEC.md 的去留（已拍板：A 随仓发布 + 全面通用化）

- **A（采用）**：随仓发布，技术决策记录是仓库价值的一部分；文中角色/作品名已全部替换为"定制角色声线/喜欢的角色声线"，本地留存原始记录
- ~~B 排除~~：不采用

## 步骤 7：git init 与首次提交

```powershell
git init -b main
git add -A
git status --short | Measure-Object -Line        # 复核文件数与清单
git commit -m "AlterVox: ASR→emotion→TTS full-CPU voice pipeline (V1 done, V2 specced)"
git remote add origin <你的仓库地址>
git push -u origin main
```

push 后用浏览器过一遍仓库根目录与 voices/（应不存在）确认。

---

## 附：与 cpu-only-ai-tuber 的差异备忘

| | cpu-only-ai-tuber | AlterVox |
| --- | --- | --- |
| 最高危隐私 | bilibili cookie、观众弹幕 | **CV 录音与衍生模型** |
| llama.cpp | 收编源码 + patches/ 正式补丁 | 收编源码（patch 维护在彼仓） |
| 权重 | 全排除 | 全排除（新增 emotion2vec/Omni/底模） |
| 运行音频 | ctest/*.wav | voices/（整目录） |
