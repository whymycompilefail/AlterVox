"""Omni-3B 文本通道纠错润色器（llama-server 常驻），五模型 bake-off 冠军的定稿移植。

实测结论（SPEC 5.0+ / 未决 #7-#9）：
- Omni 作为音频单体被否决（23s/句、情绪感知塌缩）；作为文本通道纠错润色器胜出
  （7/10、零腐化）——只消费文本与情绪分布，不自己听。
- "情绪的接口是文本本身"：输出带语气词/标点的台词直喂 TTS；max_tokens 上限
  是失控防线（虚拟主播实测过 40.96s 的笑声失控案例）。
- prompt 核心先验：说话人只说中文——转写里的英文单词都是识别错误，按发音还原。
"""

import atexit
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

import config
from .corrector import GLOSSARY_TEXT

SYSTEM_PROMPT = """你负责修复语音识别器的输出，并按情绪证据给台词加表现力。识别器经常出错：把中文词写成同音错字，或把中文词识别成英文。你的唯一工作是把错误修复成通顺、有逻辑的中文。
重要前提：说话人只会说中文，从不中英夹杂。因此转写里出现的英文单词都是识别错误，必须根据发音和语境还原成中文（例如 boss 多半是"老板"，check 多半是"查"）。
词汇表列出本场景的专有名词，转写中发音接近词汇表的词如有错误应替换为词表写法。
情绪证据来自专用情绪识别模型（标签+概率）。按证据给台词加表现力：句首/句尾添加语气词、调整标点（！？……~）；证据以中立为主时只微调句尾标点，不加戏。
除修复错误与情绪标记外：不要改写句式、不要增删实词。
只输出修复后的文本，不要解释、不要引号。
示例一（英文是识别错误）：
转写：你身上有key in
词汇表：刻印
情绪证据：惊讶(0.6) 中立(0.3)
输出：你身上有刻印？
示例二（发音接近词汇表的错字）：
转写：就像驯珠的脑子一样，空空如也
词汇表：蕈猪
情绪证据：中立(0.9)
输出：就像蕈猪的脑子一样，空空如也。
示例三（没有错误、情绪强）：
转写：这个礼物我真的很喜欢
情绪证据：开心(1.0)
输出：这个礼物我真的很喜欢！"""

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class OmniRewriter:
    _singleton = None  # 进程级单例：常驻 llama-server 全局只有一份

    def __init__(
        self,
        bin_path: str = config.LLAMA_SERVER_BIN,
        model: str = config.OMNI_MODEL,
        mmproj: str = config.OMNI_MMPROJ,
        host: str = config.OMNI_HOST,
        port: int = config.OMNI_PORT,
        threads: int = config.OMNI_THREADS,
    ):
        if OmniRewriter._singleton is not None:
            self.__dict__ = OmniRewriter._singleton.__dict__
            return

        self.bin_path = bin_path
        self.model = model
        self.mmproj = mmproj
        self.host = host
        self.port = port
        self.threads = threads
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self._health = f"http://{host}:{port}/health"
        self._ready = threading.Event()
        self._proc = None
        self._lock = threading.Lock()

        for p in (bin_path, model, mmproj):
            if not os.path.exists(p):
                raise FileNotFoundError(f"OmniRewriter: 路径不存在 {p}")

        self._spawn()
        atexit.register(self._shutdown)
        OmniRewriter._singleton = self

    # ------------------------------------------------------------------ 进程

    def _spawn(self):
        cmd = [
            self.bin_path,
            "-m", self.model,
            "--mmproj", self.mmproj,
            "--host", self.host,
            "--port", str(self.port),
            "-t", str(self.threads),
            "-ngl", "0",
        ]
        print(f"[omni] starting resident llama-server :{self.port} "
              f"({self.threads} threads)")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(self.bin_path) or None,
        )
        # 加载约 0.5-1 分钟，后台等 /health，不挡 server 启动
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _wait_ready(self, timeout: float = 900):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                print("[omni] llama-server 进程退出，放弃等待")
                return
            try:
                with urllib.request.urlopen(self._health, timeout=2) as r:
                    if r.status == 200:
                        self._ready.set()
                        print("[omni] ready (text channel)")
                        return
            except Exception:
                pass
            time.sleep(1)

    def _shutdown(self):
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------ 公开

    def rewrite(self, asr_text: str, evidence: str) -> tuple[str, float]:
        """返回 (润色文本, 耗时秒)。服务未就绪/失败时抛异常，由调用方兜底。"""
        body = json.dumps({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"转写：{asr_text}\n词汇表：{GLOSSARY_TEXT}\n"
                    f"情绪证据：{evidence or '中立(1.0)'}\n输出修复后的文本。"},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }).encode("utf-8")

        with self._lock:
            if not self._ready.wait(timeout=900):
                raise RuntimeError("Omni llama-server 未就绪")
            req = urllib.request.Request(self.url, data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            dt = time.perf_counter() - t0
        text = THINK_RE.sub("", resp["choices"][0]["message"]["content"]).strip()
        return text, dt
