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

import config

# `0.13.901.111 I [0] wrote <tmp>/tts_pipe_0.wav`
WROTE_RE = re.compile(r"\[\d+\] wrote (.*?)\s*$")
WARMUP_TEXT = "嗯，我在。"


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
        self.speaker_file = speaker_file
        self.lang = lang
        self.threads = threads
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        # 必须绝对路径：常驻进程的 cwd 在 binary 旁边，相对路径会写到别处
        self.tmp_dir = os.path.abspath(tmp_dir)
        self._proc = None
        self._lock = threading.Lock()

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

    # ------------------------------------------------------------------ 公开

    def synth_to_file(self, text: str) -> str:
        """合成一段话，返回本地 wav 路径。内部串行化，线程安全。"""
        # 协议是每行一个 prompt，换行会错位
        clean = " ".join(text.split())
        if not clean:
            raise ValueError("文本为空，跳过合成")

        with self._lock:
            try:
                src = self._synth_once(clean)
            except Exception:
                # 常驻进程死掉不应毒化后续所有句子
                self._restart()
                src = self._synth_once(clean)

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
