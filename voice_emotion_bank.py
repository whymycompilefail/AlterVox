#!/usr/bin/env python
"""AlterVox 情绪参考库：合规轨的建库脚本。

流程：
1. VoiceDesign 1.7B 用同一段声线描述 + 不同情绪子句 + 贴合该情绪的台词，
   按多个 seed 批量生成候选（台词与情绪不匹配时，生气这类情绪会念不出来）；
2. 身份预筛：中立候选取 medoid 作基准，余弦 <0.96 或参考音区越界的直接淘汰，
   不进入复验——跨情绪换人必须在这一层挡住，不能让情绪分把它抬进库；
3. 释放 1.7B，用运行时真正在用的 0.6B base backbone 把每条候选克隆到
   「中性句 + 该情绪自己的场景句」上（只测中性句会选出换文本就掉情绪的参考）；
4. 对「克隆产物」做 emotion2vec 打分 + 韵律签名，场景句达标才进库。

为什么筛选对象是克隆产物而不是原始参考：
    VoiceDesign 的设计意图不会被 0.6B 完整继承。原始参考听着有情绪，克隆后
    可能变回中立；反过来，实测里 happy 的 emotion2vec 目标分只有 0.003，
    但输出 F0 中位数 +13%、F0 标准差 +99%——声学上情绪确实发生了。
    因此判定必须同时看「模型标签」和「韵律方向」，只看任何一个都会误杀。

淘汰规则写进 manifest.json；某个情绪没有任何候选达标就不挂载，不硬上。

用法（必须用装了 qwen_tts + funasr 的 venv）：
  $env:ALTERVOX_ASSETS='D:/Ai_edge_deployment'
  D:/Ai_edge_deployment/.venv-qwentts/Scripts/python.exe voice_emotion_bank.py
  # 加 --force 重新生成已有候选
"""

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import librosa
import numpy as np
import soundfile as sf
import torch

import config

torch.set_num_threads(22)

BANK = config.ROOT / "voices" / "emotion_bank"
CAND = BANK / "candidates"
SCREEN = BANK / "screened"
SEL = BANK / "selected"
for d in (CAND, SCREEN, SEL):
    d.mkdir(parents=True, exist_ok=True)

FORCE = "--force" in sys.argv


def _arg(name: str) -> str | None:
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


# 定向补生成：--only=surprise --seeds=1,2,3,4,5,6
# 只对列出的情绪生成新 seed，其余已有候选照常参与复验与挑选。
ONLY = {s.strip() for s in (_arg("only") or "").split(",") if s.strip()}
# 身份门槛收紧后，情绪卡很难在少数 seed 里恰好抽到同一个人：试点实测每个
# 情绪约 1/6 的 seed 能同时过余弦与音区，所以候选量要够。
SEEDS = [int(s) for s in
         (_arg("seeds") or "1,2,3,4,5,6,7,8,9,10,11,12").split(",")
         if s.strip()]

# 固定声线卡：五个情绪必须共用，而且必须写到声学特征级别。
# 只写「二十五岁年轻女性、声音温和自然」时，VoiceDesign 每次都会自己脑补
# 一个具体的人，seed 一变人就变；0.6B 克隆时参考一换人，声音就成了缝合怪。
# 这里刻意不写语速、音量、情绪词——它们由情绪卡负责，写在这里会连带改音区。
VOICE = (
    "固定角色声线，五段情绪必须是同一个人：二十岁出头的成年少女，稚气未脱但已成年；"
    "音色偏薄、偏亮，基频偏高但不尖锐；口腔共鸣小、声音靠前，带一点轻鼻音；"
    "咬字清晰干脆，字尾收得短，不拖尾音；语气清冷寡淡，不撒娇、不嗲、不甜腻；"
    "音域固定在这位角色的自然音区，整段只能用这一个人的声音。"
)

# 生成参考用的台词：必须和情绪贴合。用一句中性的话配「生气」子句时，
# VoiceDesign 只会用中立的语气把它念完，参考里根本没有怒意。
GEN_TEXT = {
    "neutral": "你回来啦，我等你好一会儿了。",
    "happy": "谢谢你！这个礼物我很喜欢。",
    # 原句是两个问句，0.6B 克隆时会把句尾问调抬起来，emotion2vec 全判成
    # 「吃惊」；换成陈述句 + 叹气开头，句尾自然下坠，才带得出难过。
    "sad": "唉……你走了以后，我就只剩下一个人了。",
    "angry": "你这个人怎么这样！",
    # 「你怎么会在这里」本身带着质问语气，实测九个候选里大半被判成生气，
    # 换成明确指向惊喜的句子，情绪才不会被文本语义带偏。
    "surprise": "真的吗？这也太神奇了吧！",
}

INVARIANT = ("只改变语速、力度、气息和音高走向；不要变年龄、不要换音色、"
             "不要换口腔共鸣、不要变成另一个角色。")

# (文件名标识, emotion2vec 标签, 情绪子句)
# 情绪卡只描述表演方式，并重复一遍声线不变——实测这句不变声明能明显压低
# VoiceDesign 自己换人的概率。「发火/吃惊」这类词会被直接理解成抬高音区，
# 所以必须显式写「音高持平或下压」，否则同一张声线卡也会被顶到另一个音区。
EMOTIONS = [
    ("neutral", "中立",
     "保持这个角色清冷平淡的日常说话状态，语速中等，音量偏小。" + INVARIANT),
    ("happy", "开心",
     "同一角色的开心：语气轻快，带一点藏不住的笑意，尾音轻轻上扬，"
     "语速比平时快一点。" + INVARIANT + "不要靠抬高音高来表达开心，"
     "尤其不要变成甜腻的撒娇音。"),
    ("sad", "难过",
     "同一角色的难过：情绪压着，声音发沉、气息变多，语速偏慢，句尾往下掉。"
     + INVARIANT + "不要哭喊。"),
    ("angry", "生气",
     "同一角色的生气：压着嗓子冷冷地发火，音量不比平时大，"
     "字字咬得又短又硬，气息往下沉，句尾收紧。" + INVARIANT
     + "音高只能持平或微微下压，绝对不要提高基频、不要喊、不要尖叫。"),
    ("surprise", "吃惊",
     "同一角色的吃惊：短促地倒吸一口气，声音轻，音量偏小，"
     "句内音高只轻轻一挑，尾音短促收住。" + INVARIANT
     + "音高只微微上扬，绝对不要提高基频、不要尖叫。"),
]

# 复验用的中性台词：同一条参考、同 seed 下只改文本，避免文本语义干扰情绪判断。
# 另加一条该情绪自己的场景台词：参考只在中性句上有情绪、换到场景句就变味，
# 运行时照样听不出来（实测吃惊参考念「真的吗？你怎么会在这里！」被判成生气）。
SCREEN_TEXTS = [
    "你回来啦，我等你好一会儿了。",
    "今天外面下了点雨，路上有点堵。",
]



# ---------------------------------------------------------------- 判定阈值
MIN_EMO_SCORE = 0.45          # emotion2vec 直接认出目标情绪
# 场景句上允许走韵律通道的最低分：低于它说明模型听到的是另一种情绪，
# 靠 F0/RMS 比值硬挂只会得到「音高不对、听着还是像生气」的参考。
MIN_SCENE_FLOOR = 0.30
# 场景句重复合成多条取中位数：同一参考、同一句文本，采样不同时 emotion2vec
# 可能从「开心 0.99」跳到「生气 0.57」，单次复验会误杀好参考。
SCENE_TAKES = 3
MIN_ID_COS = 0.96             # 与身份基准的余弦，防止换人
# 音区硬约束：F0 中位数相对基准的比值。情绪允许合理升降（难过偏低、生气偏高），
# 但翻倍就是把另一个人请进来了——旧库生气 2.04、吃惊 1.64，正是缝合怪的来源。
F0_RATIO = (0.80, 1.35)
REF_DUR = (1.2, 6.0)          # 参考音频时长窗口
OUT_DUR = (1.0, 8.0)          # 克隆产物时长窗口

# 韵律签名：克隆产物 / 选中中性产物 的比值。emotion2vec 对「开心」几乎无感，
# 所以这条是独立通道，阈值按实测可分辨的幅度定，宁缺毋滥。
SIGNATURE = {
    "happy": lambda r: r["f0_med"] >= 1.08 and r["f0_std"] >= 1.20,
    "sad": lambda r: r["f0_med"] <= 0.88 and r["f0_std"] <= 0.88,
    "angry": lambda r: (r["f0_med"] >= 1.08 and r["f0_std"] >= 1.15
                        and r["rms"] >= 1.08),
    "surprise": lambda r: r["f0_std"] >= 1.20 and r["f0_med"] >= 1.03,
}

WROTE_RE = re.compile(r"\[\d+\] wrote (.*?)\s*$")
SPK_OK = "[tts] speaker reference"
SPK_FAIL = "[tts] speaker switch failed"


def log(msg: str):
    print(msg, flush=True)


def stats(path: str) -> dict:
    y, sr = librosa.load(path, sr=None)
    if y.ndim > 1:
        y = y.mean(axis=1)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=600, sr=sr,
                                 frame_length=1024, hop_length=256)
    f = f0[voiced & ~np.isnan(f0)]
    return {
        "dur": round(len(y) / sr, 2),
        "f0_med": round(float(np.median(f)), 1) if len(f) else None,
        "f0_std": round(float(np.std(f)), 1) if len(f) else None,
        "rms": round(float(np.sqrt(np.mean(y ** 2))), 4),
    }


def mean_stat(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key)]
    return float(np.mean(vals)) if vals else None


def compact_pauses(path: str, max_pause_s: float = 0.20,
                   sil_rms: float = 0.01) -> None:
    """把参考音频里的长停顿压短（句中 ≤0.2s，句尾留 0.1s）。

    llama-tts 会把参考的停顿当成韵律一起克隆：吃惊参考自带约 1s 的句中
    停顿，合成同一句话时偶尔会先吐几秒静音再出声，甚至撞满 token 上限。
    压掉长停顿时，压力测试（同文本 12 次）的跑飞次数从 6 次降到 1 次。
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    hop = max(int(sr * 0.01), 1)  # 10ms
    n = len(y) // hop
    if n == 0:
        return
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    silent = rms < sil_rms
    keep = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        limit = 0.10 if j == n else max_pause_s
        if (j - i) * hop / sr > limit:
            keep[i + max(int(limit * sr / hop), 1):j] = False
        i = j
    idx = np.repeat(keep, hop)
    idx = np.concatenate([idx, np.ones(len(y) - len(idx), dtype=bool)])
    if idx.all():
        return
    sf.write(path, y[idx], sr, subtype="PCM_16")
    log(f"[compact] {os.path.basename(path)} {len(y) / sr:.2f}s -> "
        f"{idx.sum() / sr:.2f}s")


# ---------------------------------------------------------------- 1) 生成候选
def generate_candidates() -> list[str]:
    from qwen_tts import Qwen3TTSModel

    log(f"loading VoiceDesign: {config.VOICEDESIGN_MODEL}")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        config.VOICEDESIGN_MODEL, device_map="cpu", dtype=torch.float32)
    log(f"loaded in {time.time() - t0:.0f}s")

    paths = []
    for name, _, clause in EMOTIONS:
        instruct = VOICE + clause
        for seed in SEEDS:
            path = CAND / f"{name}_s{seed}.wav"
            if ONLY and name not in ONLY:
                # 定向补生成模式下，未被点名的情绪只参与已有候选的复验
                if path.exists():
                    paths.append(str(path))
                continue
            paths.append(str(path))
            if path.exists() and not FORCE:
                log(f"[gen] {name:9} s{seed} 已存在，跳过")
                continue
            torch.manual_seed(seed)
            t = time.time()
            try:
                wavs, sr = model.generate_voice_design(
                    text=GEN_TEXT[name], language="Chinese", instruct=instruct)
            except Exception as e:
                log(f"[gen] {name:9} s{seed} FAILED {type(e).__name__}: {e}")
                continue
            wav = np.asarray(wavs[0], dtype=np.float32)
            sf.write(path, wav, sr, subtype="PCM_16")
            log(f"[gen] {name:9} s{seed} {len(wav) / sr:5.2f}s "
                f"({time.time() - t:.0f}s) -> {path.name}")

    del model
    gc.collect()
    live = [p for p in paths if os.path.exists(p)]
    # 复验/打分都基于压缩后的参考，保证入库文件与判定对象一致
    for p in live:
        compact_pauses(p)
    return live


# ---------------------------------------------------------------- 2) 实链复验
def start_engine():
    env = dict(os.environ)
    env["LLAMA_TTS_LOOP"] = "1"
    out = SCREEN / "pipe.wav"
    cmd = [
        config.LLAMA_TTS_BIN,
        "-m", config.TTS_BASE_GGUF,
        "-mm", config.TTS_MMPROJ,
        "--tts-lang", config.TTS_LANG,
        "-t", str(config.TTS_THREADS),
        "-n", str(config.TTS_MAX_NEW_TOKENS),
        "-s", str(config.TTS_SEED),
        "-o", str(out),
    ]
    log("[tts] " + " ".join(cmd))
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8",
        errors="replace", bufsize=1,
        cwd=os.path.dirname(config.LLAMA_TTS_BIN) or None)


def read_until(proc, needle: str) -> str:
    for _ in range(8192):
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("llama-tts 提前退出")
        if SPK_FAIL in line:
            raise RuntimeError(line.strip())
        if needle in line:
            return line
    raise RuntimeError(f"llama-tts 没有输出 {needle}")


def _degenerate_check():
    # pipeline/__init__.py 会连带 import asr（建库 venv 没有 sherpa_onnx），
    # 所以按文件加载运行时同一份体检函数，避免两套判定标准。
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "altervox_tts_check", config.ROOT / "pipeline" / "tts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._is_degenerate


def _screen_take(proc, is_degenerate, text: str, dst, tag: str):
    """合成一条复验音频；跑飞重试一次，仍失败就删文件返回 None。"""
    for attempt in range(2):
        proc.stdin.write(text + "\n")
        proc.stdin.flush()
        m = WROTE_RE.search(read_until(proc, "wrote "))
        os.replace(m.group(1), dst)
        if not is_degenerate(dst, config.TTS_MAX_NEW_TOKENS):
            return str(dst)
        if attempt == 0:
            log(f"[screen] {tag} 跑飞，重试")
            continue
        # 丢弃时删掉坏文件，下次建库会重新复验
        log(f"[screen] {tag} 连续跑飞，丢弃")
        try:
            dst.unlink()
        except OSError:
            pass
    return None


def screen_candidates(refs: list[str]) -> dict:
    is_degenerate = _degenerate_check()
    proc = None
    try:
        outputs = {}
        for ref in refs:
            stem = os.path.splitext(os.path.basename(ref))[0]
            name = stem.split("_s")[0]
            # 复验结果逐文件复用：文本集加了场景句后，老的 t0/t1 仍有效，
            # 只需补合成缺的。采样带随机性，重测会把已达标候选刷掉，
            # 所以只有 --force（且该情绪被点名）才重测。
            force = FORCE and (not ONLY or name in ONLY)
            if force:
                legacy = SCREEN / f"{stem}_t{len(SCREEN_TEXTS)}.wav"
                if legacy.exists():
                    legacy.unlink()

            ndst = [SCREEN / f"{stem}_t{i}.wav"
                    for i in range(len(SCREEN_TEXTS))]
            npend = [i for i, p in enumerate(ndst)
                     if force or not p.exists()
                     or is_degenerate(p, config.TTS_MAX_NEW_TOKENS)]

            scene_text = GEN_TEXT.get(name)
            # 中立的“场景句”就是第一条中性句，不重复复验
            sdst = [] if scene_text in SCREEN_TEXTS else [
                SCREEN / f"{stem}_t{len(SCREEN_TEXTS)}_{k}.wav"
                for k in range(SCENE_TAKES)]
            spend = [k for k, p in enumerate(sdst)
                     if force or not p.exists()
                     or is_degenerate(p, config.TTS_MAX_NEW_TOKENS)]

            if not npend and not spend:
                outputs[ref] = {
                    "neutral": [str(p) for p in ndst],
                    "scene": [str(p) for p in sdst],
                }
                log(f"[screen] {stem:14} 复用已有复验")
                continue
            if proc is None:
                proc = start_engine()
            proc.stdin.write("@spk=" + os.path.abspath(ref) + "\n")
            proc.stdin.flush()
            read_until(proc, SPK_OK)
            for i in npend:
                _screen_take(proc, is_degenerate, SCREEN_TEXTS[i], ndst[i],
                             f"{stem} t{i}")
            for k in spend:
                _screen_take(proc, is_degenerate, scene_text, sdst[k],
                             f"{stem} 场景{k}")
            outputs[ref] = {
                "neutral": [str(p) if p.exists() else None for p in ndst],
                "scene": [str(p) if p.exists() else None for p in sdst],
            }
            log(f"[screen] {stem:14} -> "
                f"{sum(1 for p in outputs[ref]['neutral'] if p)} 中性句，"
                f"{sum(1 for p in outputs[ref]['scene'] if p)} 场景句")
        return outputs
    finally:
        if proc is not None:
            proc.kill()


# ---------------------------------------------------------------- 3) 身份向量
def identity_embeddings(paths: list[str]) -> dict:
    from qwen_tts import Qwen3TTSModel

    log(f"loading speaker encoder: {config.TTS_BASE_MODEL}")
    t0 = time.time()
    base = Qwen3TTSModel.from_pretrained(
        config.TTS_BASE_MODEL, device_map="cpu", dtype=torch.float32)
    log(f"loaded in {time.time() - t0:.0f}s")

    out = {}
    for path in paths:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 24000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        with torch.inference_mode():
            e = base.model.extract_speaker_embedding(audio=wav, sr=24000)
        e = e.detach().float().cpu().numpy().reshape(-1)
        out[path] = e / (np.linalg.norm(e) + 1e-8)

    del base
    gc.collect()
    return out


def identity_gate(refs: list[str], embs: dict, ref_stats: dict):
    """身份预筛：在 0.6B 复验之前，只按参考音频本身判断「是不是同一个人」。

    基准取中立候选的 medoid（与其它中立候选平均余弦最高者），而不是分数最高
    的那条——medoid 才是这组声线的中心。跨情绪换人是参考层的问题，先筛掉能
    省下大量复验时间，也避免情绪分把身份不合格的参考抬进库。
    """
    neutrals = [p for p in refs
                if os.path.basename(p).startswith("neutral_")
                and REF_DUR[0] <= ref_stats[p]["dur"] <= REF_DUR[1]]
    if not neutrals:
        raise SystemExit("没有合格的中立候选，无法建立身份基准")

    if len(neutrals) == 1:
        base_ref, base_sim = neutrals[0], 1.0
    else:
        sims = {p: float(np.mean([embs[p] @ embs[q]
                                  for q in neutrals if q != p]))
                for p in neutrals}
        base_ref = max(neutrals, key=lambda p: sims[p])
        base_sim = sims[base_ref]
    log(f"[gate] 中立 medoid: {os.path.basename(base_ref)} "
        f"(平均余弦 {base_sim:.4f}, {len(neutrals)} 条中立候选)")

    base_emb = embs[base_ref]
    base_f0 = ref_stats[base_ref]["f0_med"]

    gate = {}
    for p in refs:
        cos = float(embs[p] @ base_emb)
        f0 = ref_stats[p]["f0_med"]
        ratio = round(f0 / base_f0, 3) if (f0 and base_f0) else None
        reasons = []
        if cos < MIN_ID_COS:
            reasons.append(f"身份余弦 {cos:.4f} < {MIN_ID_COS}")
        if ratio is None:
            reasons.append("参考 F0 缺失")
        elif not F0_RATIO[0] <= ratio <= F0_RATIO[1]:
            reasons.append(f"参考音区 {ratio} 不在 {F0_RATIO[0]}~{F0_RATIO[1]}")
        gate[p] = {
            "ok": not reasons,
            "cos": round(cos, 4),
            "ref_f0_ratio": ratio,
            "reason": ("身份预筛：" + "；".join(reasons)) if reasons else "",
        }
    return base_ref, gate


# ---------------------------------------------------------------- 4) 打分
def emotion_scores(rows: list[dict]) -> None:
    # 直接按文件加载 pipeline/emotion.py：走 `import pipeline` 会连带 import
    # asr.py，而建库 venv 里没有 sherpa_onnx（只有 funasr/qwen_tts）。
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "altervox_emotion", config.ROOT / "pipeline" / "emotion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    EmotionRecognizer = mod.EmotionRecognizer

    log("loading emotion2vec ...")
    emo = EmotionRecognizer(top_k=9)

    def top_of(path: str) -> dict:
        y, sr = librosa.load(path, sr=None)
        top, _ = emo.recognize(y, sr)
        return {t["label"]: t["score"] for t in top}

    for r in rows:
        outs = r.get("outputs")
        if not outs:
            # 身份预筛/复验阶段就淘汰的候选没有产物，不打分
            continue
        per_text = []
        for out in outs.get("neutral", []):
            if not out:
                per_text.append(None)
                continue
            per_text.append(top_of(out))
        r["scores"] = per_text
        # target_score 只统计中性复验句，保持与中立基准可比；
        # scene_score 是该情绪自己的场景句得分，运行时能不能听出来看这条。
        neutral = [s for s in per_text if s]
        r["target_score"] = round(float(np.mean(
            [s.get(r["label"], 0.0) for s in neutral])), 3) if neutral else 0.0
        # 场景句多条 take 取中位数：单次采样可能把「开心 0.99」念成
        # 「生气 0.57」，用一次结果判定会误杀好参考。
        scene_scores = [round(float(top_of(out).get(r["label"], 0.0)), 3)
                        for out in outs.get("scene", []) if out]
        r["scene_scores"] = scene_scores
        r["scene_score"] = (round(float(np.median(scene_scores)), 3)
                            if scene_scores else None)
        base_scores = neutral
        r["top"] = [
            {"label": lab, "score": round(float(np.mean(
                [s.get(lab, 0.0) for s in base_scores])), 3)}
            for lab in (base_scores[0] if base_scores else {})
        ]
        r["top"].sort(key=lambda x: -x["score"])


# ---------------------------------------------------------------- 主流程
def main():
    log(f"bank: {BANK}")
    log(f"backbone: {config.TTS_BASE_GGUF}")
    log(f"mmproj:  {config.TTS_MMPROJ}")

    refs = generate_candidates()
    if not refs:
        raise SystemExit("没有生成任何候选")

    log("\n=== 身份预筛（参考层）===")
    ref_stats = {p: stats(p) for p in refs}
    embs = identity_embeddings(refs)
    base_ref, gate = identity_gate(refs, embs, ref_stats)
    live = [p for p in refs if gate[p]["ok"]]
    log(f"[gate] 通过 {len(live)}/{len(refs)}，淘汰 {len(refs) - len(live)}")
    if not live:
        raise SystemExit("所有候选都没通过身份门槛，需要调整声线卡")

    log("\n=== 实链复验（0.6B base）===")
    outputs = screen_candidates(live)

    log("\n=== emotion2vec + 韵律 ===")
    rows = []
    for ref in refs:
        stem = os.path.splitext(os.path.basename(ref))[0]
        name, seed_s = stem.split("_s")
        label = next(lb for n, lb, _ in EMOTIONS if n == name)
        g = gate[ref]
        row = {
            "emotion": name,
            "label": label,
            "seed": int(seed_s),
            "ref": os.path.basename(ref),
            "ref_path": ref,
            "ref_stats": ref_stats[ref],
            "cos": g["cos"],
            "ref_f0_ratio": g["ref_f0_ratio"],
        }
        if not g["ok"]:
            row["verdict"] = "rejected"
            row["reason"] = g["reason"]
            rows.append(row)
            log(f"[gate ] {name:9} s{row['seed']} cos={g['cos']} "
                f"音区={g['ref_f0_ratio']} -> rejected（{g['reason']}）")
            continue

        outs_paths = outputs.get(ref) or {}
        # 韵律签名只在同一批中性复验句上比较：场景句会额外拉高情绪候选的
        # F0 比值，而中立基准没有对应句子，混在一起算判定会失真。
        neutral_paths = [p for p in outs_paths.get("neutral", []) if p]
        scene_paths = [p for p in outs_paths.get("scene", []) if p]
        outs = [stats(p) for p in neutral_paths]
        if not outs:
            row["verdict"] = "rejected"
            row["reason"] = "没有复验产物"
            rows.append(row)
            log(f"[skip ] {stem} 没有复验产物")
            continue
        row.update({
            "outputs": outs_paths,
            "out_stats": outs,
            "scene_stats": stats(scene_paths[0]) if scene_paths else None,
            "dur": round(float(np.mean([o["dur"] for o in outs])), 2),
            "f0_med": round(mean_stat(outs, "f0_med"), 1),
            "f0_std": round(mean_stat(outs, "f0_std"), 1),
            "rms": round(mean_stat(outs, "rms"), 4),
        })
        rows.append(row)

    emotion_scores(rows)

    # --- 身份基准就是预筛时的 medoid，不再用「情绪分最高的中立」 ---
    base = next((r for r in rows if r["ref_path"] == base_ref
                 and r.get("outputs")
                 and OUT_DUR[0] <= r["dur"] <= OUT_DUR[1]), None)
    if base is None:
        # medoid 自己复验失败：退回复验成功、离基准最近的中立候选
        pool = [r for r in rows if r["emotion"] == "neutral" and r.get("outputs")
                and OUT_DUR[0] <= r["dur"] <= OUT_DUR[1]]
        if not pool:
            raise SystemExit("没有可用的中立候选，无法建立身份基准")
        base = max(pool, key=lambda r: r["cos"])
        log(f"[base] medoid 复验失败，改用中立 s{base['seed']}"
            f"（cos={base['cos']}）")
    log(f"[base] 中立 s{base['seed']} score={base['target_score']} "
        f"f0_med={base['f0_med']} f0_std={base['f0_std']} rms={base['rms']}")

    # --- 逐条判定 ---
    for r in rows:
        if not r.get("outputs"):
            continue
        if r["emotion"] == "neutral":
            r["verdict"] = "base" if r is base else "candidate"
            continue

        ratios = {
            "f0_med": round(r["f0_med"] / base["f0_med"], 3),
            "f0_std": round(r["f0_std"] / base["f0_std"], 3),
            "rms": round(r["rms"] / base["rms"], 3),
        }
        r["ratios"] = ratios
        sig = SIGNATURE[r["emotion"]](ratios)
        r["signature"] = sig

        reasons = []
        if r["cos"] < MIN_ID_COS:
            reasons.append(f"身份余弦 {r['cos']} < {MIN_ID_COS}")
        if not REF_DUR[0] <= r["ref_stats"]["dur"] <= REF_DUR[1]:
            reasons.append(f"参考时长 {r['ref_stats']['dur']}s 越界")
        if not OUT_DUR[0] <= r["dur"] <= OUT_DUR[1]:
            reasons.append(f"产物时长 {r['dur']}s 越界")
        # 场景句是运行时真正要念的那类话。它不达标时，前面的中性句分数再高
        # 也说明参考一换文本就掉情绪，不能挂。
        if r.get("scene_score") is None:
            reasons.append("场景句复验缺失")
        elif r["scene_score"] < MIN_SCENE_FLOOR:
            reasons.append(f"场景句目标分 {r['scene_score']} "
                           f"< {MIN_SCENE_FLOOR}")

        if reasons:
            r["verdict"] = "rejected"
            r["reason"] = "；".join(reasons)
        elif r["scene_score"] >= MIN_EMO_SCORE:
            r["verdict"] = "score"
        elif sig:
            r["verdict"] = "prosody"
        else:
            r["verdict"] = "rejected"
            r["reason"] = (f"场景句目标分 {r['scene_score']} < {MIN_EMO_SCORE}，"
                           f"韵律未达签名（{ratios}）")

        log(f"[judge] {r['emotion']:9} s{r['seed']} "
            f"中性={r['target_score']:5.3f} 场景={r['scene_score']} "
            f"takes={r.get('scene_scores')} "
            f"cos={r['cos']} sig={sig} -> {r['verdict']}"
            + (f" ({r['reason']})" if r.get("reason") else ""))

    # --- 每个情绪挑一条入库 ---
    best = {"neutral": base}
    rejected = {}
    for name, label, _ in EMOTIONS:
        if name == "neutral":
            continue
        pool = [r for r in rows if r["emotion"] == name
                and r["verdict"] in ("score", "prosody")]
        if not pool:
            pool_all = [r for r in rows if r["emotion"] == name]
            if pool_all:
                b = max(pool_all, key=lambda r: (r.get("scene_score") or 0))
                rejected[label] = (f"best s{b['seed']} "
                                   f"场景={b.get('scene_score')} "
                                   f"中性={b.get('target_score')} "
                                   f"cos={b['cos']} sig={b.get('signature')}"
                                   f"；{b.get('reason', '')}")
            else:
                rejected[label] = "没有复验产物"
            continue
        # 场景句优先；同分先看身份（不能再让情绪分压过「是不是同一个人」），
        # 再看中性句分数与韵律幅度
        best[name] = max(
            pool,
            key=lambda r: (r["scene_score"], r["cos"], r["target_score"],
                           r.get("signature", False)))

    # --- 写 selected/ 与 manifest ---
    for f in SEL.glob("*.wav"):
        f.unlink()
    for name, r in best.items():
        shutil.copyfile(r["ref_path"], SEL / f"{name}.wav")
        log(f"[select] {name:9} s{r['seed']} -> selected/{name}.wav "
            f"({r['verdict']}, 场景={r.get('scene_score')}, "
            f"中性={r['target_score']}, cos={r['cos']})")

    manifest = {
        "version": 2,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": "compliant",
        "voice_description": VOICE,
        "ref_text": GEN_TEXT,
        "backbone": os.path.basename(config.TTS_BASE_GGUF),
        "screen_text": SCREEN_TEXTS,
        "scene_text": {name: GEN_TEXT[name] for name, _, _ in EMOTIONS},
        "thresholds": {
            "min_emo_score": MIN_EMO_SCORE,
            "min_scene_floor": MIN_SCENE_FLOOR,
            "scene_takes": SCENE_TAKES,
            "min_id_cos": MIN_ID_COS,
            "f0_ratio": list(F0_RATIO),
            "ref_dur": REF_DUR,
            "out_dur": OUT_DUR,
            "signature": {
                "happy": "f0_med>=1.08 且 f0_std>=1.20",
                "sad": "f0_med<=0.88 且 f0_std<=0.88",
                "angry": "f0_med>=1.08 且 f0_std>=1.15 且 rms>=1.08",
                "surprise": "f0_std>=1.20 且 f0_med>=1.03",
            },
        },
        "default": {
            "emotion": "neutral",
            "file": "selected/neutral.wav",
            "seed": base["seed"],
            "score": base["target_score"],
        },
        "refs": {
            r["label"]: {
                "file": f"selected/{name}.wav",
                "text": GEN_TEXT[name],
                "seed": r["seed"],
                "score": r["target_score"],
                "scene_score": r.get("scene_score"),
                "scene_scores": r.get("scene_scores"),
                "cos": r["cos"],
                "ref_f0_ratio": r.get("ref_f0_ratio"),
                "mode": r["verdict"],
                "ratios": r.get("ratios"),
                "top": r["top"],
            }
            for name, r in best.items() if name != "neutral"
        },
        "rejected": rejected,
        "candidates": [
            {
                "emotion": r["emotion"],
                "seed": r["seed"],
                "ref": r["ref"],
                "ref_stats": r["ref_stats"],
                "out_stats": r.get("out_stats"),
                "scene_stats": r.get("scene_stats"),
                "target_score": r.get("target_score"),
                "scene_score": r.get("scene_score"),
                "scene_scores": r.get("scene_scores"),
                "cos": r["cos"],
                "ref_f0_ratio": r.get("ref_f0_ratio"),
                "ratios": r.get("ratios"),
                "signature": r.get("signature"),
                "verdict": r["verdict"],
                "reason": r.get("reason"),
            }
            for r in rows
        ],
    }
    with open(BANK / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log("\n=== 结果 ===")
    for name, r in best.items():
        log(f"  {name:9} selected/{name}.wav  "
            f"({r['verdict']}, 场景={r.get('scene_score')}, "
            f"中性={r['target_score']}, cos={r['cos']})")
    for label, why in rejected.items():
        log(f"  {label:9} 未挂载：{why}")
    log(f"\nmanifest -> {BANK / 'manifest.json'}")


if __name__ == "__main__":
    main()
