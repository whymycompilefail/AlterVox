"""Qwen3-TTS 引擎（llama.cpp llama-tts 常驻进程），迁移自 Open-LLM-VTuber 的 qwen3_tts.py。

保留原实现的三个关键结论：
1. 常驻进程（LLAMA_TTS_LOOP=1）省掉每次 ~3.4s 的模型加载；vocoder ~3-5s 是逐次固定税，省不掉。
2. 启动时先烧一次 warmup，把首次 ~2s 的预热开销挪出用户的第一句话。
3. stdin/stdout 协议是严格的一问一答，必须整把锁串行化。

音频规格： llama-tts 输出 wav。seed/pitch 等音色参数在 config.py，与虚拟主播管线保持一致。
"""

import atexit
import os
import re
import shutil
import subprocess
import threading
import wave

import numpy as np

import config

# `0.13.901.111 I [0] wrote <tmp>/tts_pipe_0.wav`
WROTE_RE = re.compile(r"\[\d+\] wrote (.*?)\s*$")
# tts.cpp 的控制行回执（与合成结果走同一个 stdout）
SPEAKER_OK = "[tts] speaker reference"
SPEAKER_FAIL = "[tts] speaker switch failed"
SEED_OK = "[tts] seed ->"
WARMUP_TEXT = "嗯，我在。"

# Qwen3-TTS 声学帧率：12.5 token/秒，用来把时长换算成生成 token 数
TOKENS_PER_SEC = 12.5
# 产物体检：低于该 RMS 视为静音帧；静音帧占比超过阈值判定为跑飞
SILENCE_RMS = 0.01
MAX_SILENT_RATIO = 0.65
# 开口前的静音超过该值，听感就是"半天没反应"，同样算跑飞
MAX_LEAD_SILENCE = 1.2
# 跑飞重试次数：常驻进程在固定 seed 下完全确定，必须换 seed 才是新采样轨迹
SYNTH_ATTEMPTS = 3


def _is_degenerate(path: str, max_new_tokens: int) -> bool:
    """判断产物是否"跑飞"。

    情绪参考音频偶尔会让模型不吐 EOS：要么一直生成到 token 上限（时长
    约等于 max_new_tokens/12.5），要么先吐几秒静音再说话。两者听感都是
    "半天没反应"，必须拦下来重试。
    """
    with wave.open(os.fspath(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        width, raw = w.getsampwidth(), w.readframes(n)
    if max_new_tokens > 0 and n / sr >= max_new_tokens / TOKENS_PER_SEC * 0.98:
        return True
    if width != 2 or not raw:
        return False
    y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    frame = max(sr // 25, 1)  # 40ms
    usable = len(y) - len(y) % frame
    if usable == 0:
        return True
    rms = np.sqrt((y[:usable].reshape(-1, frame) ** 2).mean(axis=1))
    silent = rms < SILENCE_RMS
    if float(silent.mean()) > MAX_SILENT_RATIO:
        return True
    voiced = np.flatnonzero(~silent)
    if voiced.size == 0:
        return True
    return float(voiced[0] * frame) / sr > MAX_LEAD_SILENCE


class Qwen3TTS:
    _singleton = None  # 进程级单例：常驻引擎全局只有一份

    def __init__(
        self,
        llama_tts_bin: str = config.LLAMA_TTS_BIN,
        backbone_model: str = config.TTS_BACKBONE,
        mmproj_model: str = config.TTS_MMPROJ,
        speaker_file: str = config.TTS_SPEAKER,
        lang: str = config.TTS_LANG,
        threads: int = config.TTS_THREADS,
        seed: int = config.TTS_SEED,
        max_new_tokens: int = config.TTS_MAX_NEW_TOKENS,
        tmp_dir: str = config.TTS_TMP_DIR,
    ):
        if Qwen3TTS._singleton is not None:
            self.__dict__ = Qwen3TTS._singleton.__dict__
            return

        self.llama_tts_bin = llama_tts_bin
        self.backbone_model = backbone_model
        self.mmproj_model = mmproj_model
        self.speaker_file = os.path.abspath(speaker_file) if speaker_file else None
        self.lang = lang
        self.threads = threads
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        # 必须绝对路径：常驻进程的 cwd 在 binary 旁边，相对路径会写到别处
        self.tmp_dir = os.path.abspath(tmp_dir)
        self._proc = None
        self._lock = threading.Lock()
        # 引擎当前持有的参考音频；None 表示无参考
        self._speaker = self.speaker_file
        # 常驻进程当前的采样 seed（@seed= 切换后与构造参数不同）
        self._seed = self.seed

        for p in (llama_tts_bin, backbone_model, mmproj_model):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Qwen3-TTS: 路径不存在 {p}")
        if speaker_file and not os.path.exists(speaker_file):
            raise FileNotFoundError(f"Qwen3-TTS: 音色参考不存在 {speaker_file}")

        os.makedirs(self.tmp_dir, exist_ok=True)
        self._spawn()
        atexit.register(self._shutdown)
        # 首次合成有 ~2s 额外预热，在后台烧掉，不占用户第一句
        threading.Thread(target=self._warmup, daemon=True).start()

        Qwen3TTS._singleton = self

    def _warmup(self):
        try:
            with self._lock:
                path = self._synth_once(WARMUP_TEXT)
            try:
                os.remove(path)
            except OSError:
                pass
            print("[tts] warmup done, resident engine ready")
        except Exception as e:
            print(f"[tts] warmup failed (harmless): {e}")

    # ------------------------------------------------------------------ 进程

    def _spawn(self):
        out_base = os.path.join(self.tmp_dir, "tts_pipe.wav")
        cmd = [
            self.llama_tts_bin,
            "-m", self.backbone_model,
            "-mm", self.mmproj_model,
            "--tts-lang", self.lang,
            "-t", str(self.threads),
            "-o", out_base,
        ]
        if self.speaker_file:
            cmd += ["--tts-speaker-file", self.speaker_file]
        if self.max_new_tokens > 0:
            cmd += ["-n", str(self.max_new_tokens)]
        if self.seed >= 0:
            cmd += ["-s", str(self.seed)]

        env = dict(os.environ)
        env["LLAMA_TTS_LOOP"] = "1"
        print(f"[tts] starting resident llama-tts ({self.threads} threads, seed={self.seed})")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=os.path.dirname(self.llama_tts_bin) or None,
        )

    def _restart(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
        self._spawn()
        # 新进程只带启动参数里的默认参考
        self._speaker = self.speaker_file
        self._seed = self.seed

    def _set_speaker(self, speaker_file: str | None):
        """让常驻进程换一条参考音频。引擎每句都会重新编码参考，切换无额外开销。"""
        if speaker_file == self._speaker:
            return
        if self._proc is None or self._proc.poll() is not None:
            self._restart()

        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write("@spk=" + (speaker_file or "") + "\n")
        self._proc.stdin.flush()

        for _ in range(4096):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("llama-tts 在切换参考音频时退出")
            if SPEAKER_FAIL in line:
                raise RuntimeError(f"参考音频切换失败: {speaker_file}")
            if SPEAKER_OK in line:
                self._speaker = speaker_file
                return
        raise RuntimeError("llama-tts 没有回应参考音频切换")

    def _set_seed(self, seed: int):
        """切换常驻进程的采样 seed（同时重置 backbone 采样器和音频码 RNG）。

        即使 seed 没变也要发：warmup 之后重建一次采样器，首句才不会踩在
        warmup 留下的轨迹上。
        """
        if seed < 0:
            return
        if self._proc is None or self._proc.poll() is not None:
            self._restart()

        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(f"@seed={seed}\n")
        self._proc.stdin.flush()

        for _ in range(4096):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("llama-tts 在切换 seed 时退出")
            if SEED_OK in line:
                self._seed = seed
                return
        raise RuntimeError("llama-tts 没有回应 seed 切换")

    def _synth_once(self, text: str) -> str:
        """发一行 prompt 给常驻进程，等它报告写出的 wav 路径。"""
        if self._proc is None or self._proc.poll() is not None:
            self._restart()

        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(text + "\n")
        self._proc.stdin.flush()

        for _ in range(4096):
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("llama-tts 在合成中途退出")
            m = WROTE_RE.search(line)
            if m:
                return m.group(1)
            if "loop mode" in line:
                continue
        raise RuntimeError("llama-tts 没有为该 prompt 产出输出行")

    def _synth_checked(self, text: str, target: str | None) -> str:
        """按目标参考合成；产物跑飞时换 seed 重试，仍失败才退回默认音色。

        常驻进程在固定 seed 下是完全确定性的：同一条参考、同一句文本如果
        落在坏轨迹上（先吐几秒静音、或干脆不吐 EOS），重试多少次结果都一
        样。所以每次重试都要切一个不同的 seed，才是真正的新采样轨迹。直接
        退回默认音色会让这句情绪消失，因此默认音色是最后一档。
        """
        self._set_speaker(target)
        src = ""
        for i in range(SYNTH_ATTEMPTS):
            self._set_seed(self.seed + i)
            src = self._synth_once(text)
            if not _is_degenerate(src, self.max_new_tokens):
                return src
            print(f"[tts] degenerate output with {target or 'no reference'} "
                  f"(seed {self._seed}, take {i + 1}/{SYNTH_ATTEMPTS})")
        if target == self.speaker_file:
            return src
        print("[tts] retry with default voice")
        self._set_speaker(self.speaker_file)
        self._set_seed(self.seed)
        return self._synth_once(text)

    # ------------------------------------------------------------------ 公开

    def synth_to_file(self, text: str, speaker_file: str | None = None) -> str:
        """合成一段话，返回本地 wav 路径。内部串行化，线程安全。

        speaker_file 为情绪参考音频；空则用启动时的默认音色。
        """
        # 协议是每行一个 prompt，换行会错位
        clean = " ".join(text.split())
        if not clean:
            raise ValueError("文本为空，跳过合成")
        target = os.path.abspath(speaker_file) if speaker_file else self.speaker_file

        with self._lock:
            try:
                src = self._synth_checked(clean, target)
            except Exception:
                # 常驻进程死掉不应毒化后续所有句子
                self._restart()
                src = self._synth_checked(clean, target)

        if not os.path.exists(src):
            raise FileNotFoundError(f"llama-tts 报告了 {src} 但文件不存在")
        return src

    def synth_cleanup(self, path: str):
        try:
            os.remove(path)
        except OSError:
            pass

    def _shutdown(self):
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
