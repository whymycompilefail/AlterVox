"""emotion2vec_plus_large 情绪感知（funasr），PoC run_emotion.py 的服务化移植。

实测 0.2~0.4s/句，速度与 base 同档；标签为双语格式（"开心/happy"），对外只取
中文半截。低强度样本会出混合分布（如 低落 → neutral .38/surprised .36/happy .22），
交给润色层解读；残余塌缩类（俏皮 → neutral .975）由 LLM prompt 兜底规则处理。
"""

import os
import tempfile
import time

import soundfile as sf

import config


class EmotionRecognizer:
    def __init__(self, model_dir: str = config.EMOTION_MODEL_DIR, top_k: int = 3):
        from funasr import AutoModel
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"emotion2vec: 模型目录不存在 {model_dir}")
        self.top_k = top_k
        self.model = AutoModel(model=model_dir, disable_update=True)

    def recognize(self, audio, sample_rate: int = config.ASR_SAMPLE_RATE):
        """audio: float32 mono。返回 (top 标签列表, 耗时秒)。

        标签项形如 {"label": "开心", "score": 0.61}，按概率降序。
        """
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, sample_rate, subtype="PCM_16")
            t0 = time.perf_counter()
            res = self.model.generate(input=path, granularity="utterance",
                                      extract_embedding=False)
            dt = time.perf_counter() - t0
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        pairs = sorted(zip(res[0]["labels"], res[0]["scores"]), key=lambda x: -x[1])
        top = [{"label": str(l).split("/")[0], "score": round(float(s), 3)}
               for l, s in pairs[:self.top_k]]
        return top, dt
