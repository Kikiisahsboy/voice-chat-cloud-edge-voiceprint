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
    def _patch_speechbrain():
        """修复 torch 2.3+ 与 speechbrain 1.1.0 的 amp API 不兼容。"""
        import torch
        from torch.cuda.amp import custom_fwd as _cuda_custom_fwd
        from torch.cuda.amp import custom_bwd as _cuda_custom_bwd

        # torch 2.3 的 custom_fwd 签名变成 (fwd=None, *, cast_inputs=None)
        # 不再接受 device_type 参数。需要包装以兼容 speechbrain。
        def _wrapped_custom_fwd(fwd=None, *, device_type=None, **kwargs):
            # 忽略 device_type，speechbrain 传但我们不认
            if fwd is None:
                return lambda f: _wrapped_custom_fwd(f, device_type=device_type, **kwargs)
            return _cuda_custom_fwd(fwd, **kwargs)

        def _wrapped_custom_bwd(bwd=None, *, device_type=None, **kwargs):
            if bwd is None:
                return lambda f: _wrapped_custom_bwd(f, device_type=device_type, **kwargs)
            return _cuda_custom_bwd(bwd, **kwargs)

        torch.amp.custom_fwd = _wrapped_custom_fwd
        torch.amp.custom_bwd = _wrapped_custom_bwd
        logger.debug("SpeechBrain autocast 补丁已应用")

    def __init__(self, model_id: str = "speechbrain/spkrec-ecapa-voxceleb"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError("需要 speechbrain: pip install speechbrain")

        # 修复 torch 2.3+ 兼容性（必须在加载模型之前）
        self._patch_speechbrain()

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
