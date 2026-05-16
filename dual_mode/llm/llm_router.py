# -*- coding: utf-8 -*-
"""
LLM Router — Cloud-first dual-mode orchestration.

Strategy:
  1. Try cloud TTS server (combined LLM+TTS) in a thread with timeout.
  2. If successful → audio played, return "cloud".
  3. If timeout or error → fall back to local Ollama + local TTS, return "local".
  4. Next turn always retries the cloud path (no permanent lock-in).
"""

import logging
import queue
import threading
import time
from typing import Tuple

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM requests to cloud or local backend based on availability."""

    def __init__(
        self,
        tts_client,               # Original TTSStreamClient for cloud
        local_ollama,             # LocalOllamaClient
        local_tts,                # LocalTTSClient
        cloud_timeout: float = 10.0,
    ):
        self._tts_client = tts_client
        self._local_ollama = local_ollama
        self._local_tts = local_tts
        self._cloud_timeout = cloud_timeout

        # Stats
        self.cloud_successes = 0
        self.local_fallbacks = 0
        self.consecutive_cloud_failures = 0

    def get_response_and_play(self, user_text: str) -> Tuple[str, str]:
        """
        Get AI response and play audio. Tries cloud first, falls back to local.

        Args:
            user_text: The ASR transcript to send to the LLM.

        Returns:
            Tuple of (response_text, source) where source is "cloud" or "local".
            On cloud success, response_text may be empty (TTS server handles text internally).
        """
        # Try cloud path
        result_queue: queue.Queue = queue.Queue()

        def _try_cloud():
            try:
                self._tts_client.send_text_and_play_audio(user_text)
                result_queue.put(("ok", ""))
            except Exception as e:
                result_queue.put(("error", str(e)))

        cloud_thread = threading.Thread(target=_try_cloud, daemon=True)
        cloud_thread.start()
        cloud_thread.join(timeout=self._cloud_timeout)

        if cloud_thread.is_alive():
            # Timeout — cloud took too long
            logger.warning(
                "Cloud TTS timed out after %.1fs, falling back to local.",
                self._cloud_timeout)
            self._force_close_cloud()
            cloud_thread.join(timeout=2.0)
            return self._fallback_to_local(user_text)

        try:
            status, detail = result_queue.get_nowait()
        except queue.Empty:
            logger.warning("Cloud thread produced no result, falling back.")
            return self._fallback_to_local(user_text)

        if status == "ok":
            self.cloud_successes += 1
            self.consecutive_cloud_failures = 0
            logger.info("Cloud LLM response successful.")
            return ("", "cloud")

        # Cloud errored out
        logger.warning("Cloud TTS error: %s. Falling back to local.", detail)
        return self._fallback_to_local(user_text)

    def _fallback_to_local(self, user_text: str) -> Tuple[str, str]:
        """Run local LLM + local TTS."""
        self.local_fallbacks += 1
        self.consecutive_cloud_failures += 1
        logger.info("Running local LLM (%s)...", self._local_ollama.model)

        try:
            response_text = self._local_ollama.generate_full(user_text)
            if not response_text:
                response_text = "抱歉，我暂时无法回答这个问题。"
        except Exception as e:
            logger.error("Local Ollama failed: %s", e)
            response_text = "抱歉，本地服务暂时不可用，请稍后重试。"

        logger.info("Local LLM response: %s", response_text[:80])

        try:
            self._local_tts.speak(response_text)
        except Exception as e:
            logger.error("Local TTS failed: %s", e)

        return (response_text, "local")

    def _force_close_cloud(self):
        """Force close the cloud TTS connection."""
        try:
            self._tts_client.close()
        except Exception:
            pass

    def reconnect_cloud(self) -> bool:
        """Reconnect to cloud TTS server (called before each turn)."""
        try:
            return self._tts_client.connect()
        except Exception as e:
            logger.warning("Cloud TTS reconnect failed: %s", e)
            return False

    @property
    def is_cloud_healthy(self) -> bool:
        """Heuristic: cloud was successful last time."""
        return self.consecutive_cloud_failures == 0
