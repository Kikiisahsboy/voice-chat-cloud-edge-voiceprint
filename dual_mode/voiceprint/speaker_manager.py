# -*- coding: utf-8 -*-
"""
Speaker manager for voiceprint enrollment and identification.

Stores speaker voiceprint templates as .npy files and performs
cosine-similarity-based identification.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerManager:
    """Manages speaker enrollment, storage, and identification."""

    def __init__(
        self,
        enrollment_dir: str,
        embedding_dim: int = 512,
        threshold: float = 0.65,
    ):
        self.enrollment_dir = enrollment_dir
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self._templates: Dict[str, np.ndarray] = {}

        os.makedirs(enrollment_dir, exist_ok=True)
        self._load_all_templates()

    def _template_path(self, name: str) -> str:
        safe = "".join(c for c in name if c.isalnum() or c in "._-")
        return os.path.join(self.enrollment_dir, f"{safe}.npy")

    def _load_all_templates(self):
        """Load all enrolled speaker templates from disk."""
        self._templates.clear()
        if not os.path.isdir(self.enrollment_dir):
            return
        for fname in os.listdir(self.enrollment_dir):
            if fname.endswith(".npy"):
                name = fname[:-4]
                path = os.path.join(self.enrollment_dir, fname)
                try:
                    emb = np.load(path)
                    if emb.shape == (self.embedding_dim,):
                        self._templates[name] = emb
                        logger.info("Loaded speaker template: %s", name)
                    else:
                        logger.warning(
                            "Skipping %s: expected shape (%d,), got %s",
                            fname, self.embedding_dim, emb.shape)
                except Exception as e:
                    logger.error("Failed to load template %s: %s", fname, e)

    def enroll(self, name: str, embeddings: List[np.ndarray]) -> bool:
        """
        Enroll a new speaker.

        Args:
            name: Speaker name/identifier.
            embeddings: List of 512-dim embedding vectors from multiple utterances.

        Returns:
            True on success.
        """
        if not embeddings:
            logger.error("No embeddings provided for enrollment.")
            return False

        # Average multiple embeddings for a robust template
        stacked = np.stack(embeddings, axis=0)
        template = np.mean(stacked, axis=0)

        # L2 normalize the averaged template
        norm = np.linalg.norm(template)
        if norm > 0:
            template = template / norm

        path = self._template_path(name)
        np.save(path, template)
        self._templates[name] = template
        logger.info(
            "Enrolled speaker '%s' with %d utterance(s) → %s",
            name, len(embeddings), path)
        return True

    def identify(self, embedding: np.ndarray) -> Optional[str]:
        """
        Identify a speaker from an embedding.

        Args:
            embedding: 512-dim normalized embedding.

        Returns:
            Speaker name if matched, None otherwise.
        """
        if not self._templates:
            return None

        best_name = None
        best_score = -1.0

        for name, template in self._templates.items():
            score = float(np.dot(embedding, template))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.threshold:
            logger.info(
                "Speaker identified: '%s' (score=%.3f)", best_name, best_score)
            return best_name

        logger.info(
            "Unknown speaker (best='%s', score=%.3f < threshold=%.3f)",
            best_name, best_score, self.threshold)
        return None

    def remove(self, name: str) -> bool:
        """Remove an enrolled speaker."""
        path = self._template_path(name)
        if os.path.exists(path):
            os.remove(path)
        if name in self._templates:
            del self._templates[name]
        logger.info("Removed speaker: %s", name)
        return True

    def list_enrolled(self) -> List[str]:
        return sorted(self._templates.keys())

    @property
    def speaker_count(self) -> int:
        return len(self._templates)
