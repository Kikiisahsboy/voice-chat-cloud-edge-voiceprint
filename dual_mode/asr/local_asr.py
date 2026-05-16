# -*- coding: utf-8 -*-
"""本地 Vosk ASR 引擎（离线语音识别）。"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LocalASREngine:
    """Vosk 本地语音识别，支持流式输入。"""

    def __init__(self, model_path: str, sample_rate: int = 16000):
        import vosk

        logger.info("加载 Vosk 模型: %s", model_path)
        self._model = vosk.Model(model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
        self._recognizer.SetWords(False)
        self._sample_rate = sample_rate
        logger.info("Vosk 本地 ASR 就绪")

    def accept_waveform(self, data: bytes) -> Optional[dict]:
        """
        接受原始 PCM 音频数据，返回识别结果。

        Args:
            data: int16 PCM 音频字节。

        Returns:
            dict with keys "partial" or "text", or None.
        """
        if self._recognizer.AcceptWaveform(data):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").strip().replace(" ", "")
            if text:
                return {"text": text}
        else:
            partial = json.loads(self._recognizer.PartialResult())
            text = partial.get("partial", "").strip()
            if text:
                return {"partial": text}
        return None

    def reset(self):
        """重置识别器状态（新一轮对话前调用）。"""
        self._recognizer.Reset()

    def final_result(self) -> Optional[dict]:
        """获取最终结果（语音结束后调用）。"""
        result = json.loads(self._recognizer.FinalResult())
        text = result.get("text", "").strip().replace(" ", "")
        if text:
            return {"text": text}
        return None
