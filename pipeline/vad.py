"""Silero VAD 人声过滤（sherpa-onnx），对齐 Open-LLM-VTuber 的服务端降噪思路。

前端能量 VAD 只负责断句（便宜、够用）；人声与否由这里用神经 VAD 裁决：
键盘声、音乐、环境噪声整体剔除，只把人声段拼回给 ASR——垃圾音频在进管线
之前就被拒绝，不再排队拖响应，也不会喂出 ASR 幻觉。

每次调用独立建 VAD 实例（模型只有 2MB，初始化毫秒级），天然线程安全。
"""

import os

import numpy as np
import sherpa_onnx

import config


class VoiceFilter:
    def __init__(
        self,
        model: str = config.ASR_VAD_MODEL,
        sample_rate: int = config.ASR_SAMPLE_RATE,
        threshold: float = 0.5,      # 人声概率阈值（Open-LLM-VTuber 用 0.4，0.5 更保守）
        min_silence_s: float = 0.8,  # 与前端断句静音窗一致
        min_speech_s: float = 0.25,
        max_speech_s: float = 20.0,  # 超长人声切分，避免整段排队
    ):
        if not os.path.isfile(model):
            raise FileNotFoundError(f"Silero VAD: 模型不存在 {model}")
        self.model = model
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_s = min_silence_s
        self.min_speech_s = min_speech_s
        self.max_speech_s = max_speech_s

    def speech_only(self, audio, sr: int | None = None) -> np.ndarray:
        """返回只含人声段的 float32 mono；无有效人声时返回长度 0 数组。"""
        audio = np.asarray(audio, dtype=np.float32)
        if sr is not None and sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        scfg = sherpa_onnx.SileroVadModelConfig(
            model=self.model,
            threshold=self.threshold,
            min_silence_duration=self.min_silence_s,
            min_speech_duration=self.min_speech_s,
            max_speech_duration=self.max_speech_s,
            window_size=512,
        )
        vad = sherpa_onnx.VoiceActivityDetector(
            sherpa_onnx.VadModelConfig(silero_vad=scfg, sample_rate=self.sample_rate),
            buffer_size_in_seconds=60,
        )
        # 必须分块流式喂入：一次性整段 accept 只会被处理出一小段，
        # 0.5s 一块喂入实测人声保留 ~95%，噪声/纯音全拒
        W = int(self.sample_rate * 0.5)
        for i in range(0, len(audio), W):
            vad.accept_waveform(audio[i:i + W])
        vad.flush()

        chunks = []
        first_start = None
        while not vad.empty():
            seg = vad.front
            if first_start is None:
                first_start = int(seg.start)
            chunks.append(np.asarray(seg.samples, dtype=np.float32))
            vad.pop()
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(chunks)
        # 段前预留 150ms 原始音频：VAD 会削掉起音的软 onset（实测吃过"不"字）
        if first_start > 0:
            pre = int(0.15 * self.sample_rate)
            head = audio[max(0, first_start - pre):first_start]
            out = np.concatenate([head, out])
        return out
