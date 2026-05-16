# -*- coding: utf-8 -*-
"""
SpeechBrain ECAPA-TDNN 声纹提取引擎（Windows 兼容）。

SpeechBrain 会自动下载并缓存模型到 huggingface hub 缓存目录。
Windows 下通过环境变量禁用 symlink 以避免权限问题。
"""

import logging
import os

# 必须在 import speechbrain 前设置
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_SYMLINKS"] = "0"

import numpy as np

logger = logging.getLogger(__name__)


class SpeechBrainEngine:
    """Speaker embedding extractor using SpeechBrain ECAPA-TDNN."""

    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError("需要 speechbrain: pip install speechbrain")

        logger.info("加载 SpeechBrain 模型: %s ...", model_id)

        # 不指定 savedir，让 SpeechBrain 使用 huggingface hub 缓存
        # 这样避免 Windows symlink 权限问题
        try:
            self._model = EncoderClassifier.from_hparams(
                source=model_id,
                run_opts={"device": "cpu"},
            )
        except Exception:
            # 如果默认路径有问题，尝试用绝对路径缓存
            savedir = os.path.abspath("pretrained_models/spkrec-ecapa-voxceleb")
            os.makedirs(savedir, exist_ok=True)
            self._model = EncoderClassifier.from_hparams(
                source=model_id,
                savedir=savedir,
                run_opts={"device": "cpu"},
            )

        logger.info("SpeechBrain 模型加载完毕")

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        import torch

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        # SpeechBrain 需要 PyTorch tensor
        tensor = torch.from_numpy(audio).to(self._model.device)
        embedding = self._model.encode_batch(tensor)
        emb = embedding.squeeze().detach().cpu().numpy()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @property
    def embedding_dim(self) -> int:
        return 192
