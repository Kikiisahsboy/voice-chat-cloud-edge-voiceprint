# -*- coding: utf-8 -*-
"""
Wake word detection using openwakeword (Windows/Linux/macOS compatible).

openwakeword is a pure Python ONNX-based wake word engine with pre-trained
models for "alexa", "hey_mycroft", "hey_jarvis", etc.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Pre-trained models available in openwakeword
_PRETRAINED = [
    "alexa",
    "hey_mycroft",
    "hey_jarvis",
    "ok_nabu",
    "hey_computer",
    "xiao_ai",      # 小爱同学
]


class WakeWordDetector:
    """
    Wake word detector backed by openwakeword.

    Usage:
        detector = WakeWordDetector(model="alexa", threshold=0.5)
        while True:
            audio = get_audio_chunk()  # 16kHz int16 bytes
            if detector.detect(audio):
                print("Wake word detected!")
    """

    def __init__(
        self,
        model: str = "alexa",
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        try:
            from openwakeword.model import Model
        except ImportError:
            raise ImportError(
                "需要 openwakeword: pip install openwakeword"
            )

        if model not in _PRETRAINED:
            logger.warning(
                "模型 '%s' 不在预训练列表中 %s，尝试加载...", model, _PRETRAINED)

        self.model_name = model
        self.threshold = threshold
        self.sample_rate = sample_rate

        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        logger.info("唤醒词模型 '%s' 已加载", model)

    def detect(self, audio_bytes: bytes) -> bool:
        """
        Check if wake word is present in the audio chunk.

        Args:
            audio_bytes: Raw int16 PCM audio at sample_rate.

        Returns True if wake word score exceeds threshold.
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        prediction = self._model.predict(audio)
        if self.model_name in prediction:
            score = float(prediction[self.model_name])
            if score >= self.threshold:
                logger.info("唤醒词检测到! (score=%.3f)", score)
                return True
        return False

    def reset(self):
        """Reset the detector's internal state between uses."""
        self._model.reset()
