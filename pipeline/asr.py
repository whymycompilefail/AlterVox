"""SenseVoice ASR（sherpa-onnx int8），迁移自 Open-LLM-VTuber 的 sherpa_onnx_asr.py。

只保留 sense_voice 路径：本项目只需要"语音→文字"。
"""

import os

import numpy as np
import sherpa_onnx

import config


class SenseVoiceASR:
    def __init__(
        self,
        model_dir: str = config.SENSEVOICE_DIR,
        sample_rate: int = config.ASR_SAMPLE_RATE,
        num_threads: int = config.ASR_NUM_THREADS,
        use_itn: bool = config.ASR_USE_ITN,
    ):
        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        for p in (model_path, tokens_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"SenseVoice: 文件不存在 {p}")

        self.sample_rate = sample_rate
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            use_itn=use_itn,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono，采样率 = self.sample_rate。"""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self.recognizer.decode_streams([stream])
        return stream.result.text
