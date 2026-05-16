# -*- coding: utf-8 -*-
"""
WeSpeaker engine wrapper for voiceprint embedding extraction.

WeSpeaker provides state-of-the-art Chinese speaker recognition.
Model: ECAPA-TDNN with 512-dim embeddings.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class WeSpeakerEngine:
    """Wrapper around WeSpeaker for speaker embedding extraction."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the WeSpeaker model.

        Args:
            model_path: Local path to the model or ModelScope model ID.
            device: 'cpu' or 'cuda'.
        """
        self.model_path = model_path
        self.device = device
        self._model: Optional[object] = None
        self._load_model()

    def _load_model(self):
        """Load the WeSpeaker model. Supports local path and ModelScope download."""
        try:
            from wespeaker import Model
        except ImportError:
            raise ImportError(
                "WeSpeaker is required. Install with: pip install wespeaker")

        if os.path.isdir(self.model_path):
            # Local model directory
            logger.info("Loading WeSpeaker from local path: %s", self.model_path)
            self._model = Model(self.model_path, device=self.device)
        else:
            # Assume it's a ModelScope model ID
            logger.info("Loading WeSpeaker from ModelScope: %s", self.model_path)
            from modelscope.hub.snapshot_download import snapshot_download

            local_dir = snapshot_download(
                self.model_path,
                cache_dir="models/wespeaker",
            )
            self._model = Model(local_dir, device=self.device)
            self.model_path = local_dir

        logger.info("WeSpeaker model loaded successfully.")

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Float32 numpy array, shape (n_samples,), mono.
            sample_rate: Audio sample rate (should be 16000).

        Returns:
            Normalized 512-dim embedding vector.
        """
        if self._model is None:
            raise RuntimeError("WeSpeaker model not loaded.")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure 16kHz — the model expects 16k
        if sample_rate != 16000:
            logger.warning(
                "WeSpeaker expects 16kHz audio, got %dHz. Resampling not automatic.",
                sample_rate)

        # WeSpeaker Model.extract_embedding expects (n_samples,) float32 array
        embedding = self._model.extract_embedding(audio)
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @property
    def embedding_dim(self) -> int:
        return 512
