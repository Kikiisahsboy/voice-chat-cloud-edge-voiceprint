# -*- coding: utf-8 -*-
"""声纹管理器：注册、识别、存储。"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerManager:
    def __init__(
        self,
        enrollment_dir: str = "enrolled_speakers/",
        embedding_dim: int = 192,
        threshold: float = 0.55,
    ):
        self.enrollment_dir = enrollment_dir
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self._templates: Dict[str, np.ndarray] = {}

        os.makedirs(enrollment_dir, exist_ok=True)
        self._load_all()

    def _path(self, name: str) -> str:
        safe = "".join(c for c in name if c.isalnum() or c in "._-")
        return os.path.join(self.enrollment_dir, f"{safe}.npy")

    def _load_all(self):
        self._templates.clear()
        if not os.path.isdir(self.enrollment_dir):
            return
        for fn in os.listdir(self.enrollment_dir):
            if fn.endswith(".npy"):
                name = fn[:-4]
                try:
                    emb = np.load(os.path.join(self.enrollment_dir, fn))
                    if emb.shape == (self.embedding_dim,):
                        self._templates[name] = emb
                        logger.info("已加载声纹: %s", name)
                except Exception as e:
                    logger.error("加载声纹失败 %s: %s", fn, e)

    def enroll(self, name: str, embeddings: List[np.ndarray]) -> bool:
        if not embeddings:
            logger.error("没有提供嵌入向量")
            return False

        template = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = np.linalg.norm(template)
        if norm > 0:
            template = template / norm

        np.save(self._path(name), template)
        self._templates[name] = template
        logger.info("已注册说话人 '%s' (%d 条语音)", name, len(embeddings))
        return True

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        识别说话人。
        返回 (说话人姓名或None, 最高相似度分数)。
        """
        if not self._templates:
            return None, 0.0

        best_name, best_score = None, -1.0
        for name, tpl in self._templates.items():
            score = float(np.dot(embedding, tpl))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.threshold and best_name:
            logger.info("识别说话人: '%s' (score=%.3f)", best_name, best_score)
            return best_name, best_score

        logger.info("未知说话人 (best='%s', score=%.3f)", best_name, best_score)
        return None, best_score

    def remove(self, name: str) -> bool:
        p = self._path(name)
        if os.path.exists(p):
            os.remove(p)
        self._templates.pop(name, None)
        logger.info("已删除说话人: %s", name)
        return True

    def list_enrolled(self) -> List[str]:
        return sorted(self._templates.keys())

    @property
    def speaker_count(self) -> int:
        return len(self._templates)
