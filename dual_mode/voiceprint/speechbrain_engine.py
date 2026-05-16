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

    @staticmethod
    def _patch_torch_amp():
        """修复 torch 2.3+ 与 speechbrain 1.1.0 的 amp API 不兼容。"""
        import torch
        # speechbrain 调用 torch.amp.custom_fwd(device_type=...) 但 torch 2.3+
        # 的 custom_fwd 签名不同。映射到正确的 API。
        if hasattr(torch, 'amp') and not hasattr(torch.amp, 'custom_fwd'):
            # torch 2.4+ 移除了 custom_fwd，用 autocast 模式替代
            try:
                # 尝试用 torch.cuda.amp.custom_fwd 作为替代
                torch.amp.custom_fwd = torch.cuda.amp.custom_fwd
                torch.amp.custom_bwd = torch.cuda.amp.custom_bwd
            except Exception:
                pass

    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb"):
        try:
            # 修复 torch 2.3+ 与 speechbrain 1.1.0 的兼容问题
            self._patch_torch_amp()
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
