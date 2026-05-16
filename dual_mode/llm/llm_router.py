# -*- coding: utf-8 -*-
"""
LLM 路由器 — 云端优先，超时/失败自动切本地。

Cloud path: 连接云端 TTS 服务器 (LLM+TTS 一体)，播放返回的音频。
Local path: 本地 Ollama → edge-tts。
"""

import logging
import queue
import socket
import struct
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pyaudio
    _pyaudio_ok = True
except ImportError:
    pyaudio = None
    _pyaudio_ok = False


class CloudTTSClient:
    """云端 TTS TCP 客户端（LLM+TTS 一体服务）。"""

    def __init__(self, host: str, port: int, sample_rate: int = 24000):
        if not _pyaudio_ok:
            raise ImportError("需要 pyaudio: pip install pyaudio")
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self._sock: Optional[socket.socket] = None
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._tts_timeout = 10.0

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(self._tts_timeout)
            logger.info("云端 TTS 已连接 %s:%d", self.host, self.port)
            return True
        except Exception as e:
            logger.warning("云端 TTS 连接失败: %s", e)
            self._sock = None
            return False

    def send_text_and_play(self, text: str) -> None:
        """发送文本到云端 TTS 并实时播放返回的音频。"""
        if not self._sock:
            raise ConnectionError("未连接云端 TTS")

        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        self._sock.sendall(f"{text}|".encode('utf-8'))

        stream: Optional[pyaudio.Stream] = None

        while True:
            header = self._sock.recv(4)
            if not header:
                break

            len_bytes = self._sock.recv(4)
            if not len_bytes:
                break
            payload_len = struct.unpack('>I', len_bytes)[0]

            payload = b''
            while len(payload) < payload_len:
                chunk = self._sock.recv(payload_len - len(payload))
                if not chunk:
                    raise ConnectionError("TTS 连接断开")
                payload += chunk

            if header == b'TEXT':
                print(payload.decode('utf-8'), end="", flush=True)
            elif header == b'AUDO':
                if stream is None:
                    stream = self._pyaudio.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.sample_rate,
                        output=True,
                        frames_per_buffer=4096,
                    )
                stream.write(payload)
            elif header == b'EOT ':
                print()
                break

        if stream:
            time.sleep(0.3)  # 等待缓冲区排空
            stream.stop_stream()
            stream.close()

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


class LLMRouter:
    """云边双模路由：云端优先，超时切本地。"""

    def __init__(
        self,
        cloud_tts: CloudTTSClient,
        local_ollama,      # LocalOllamaClient
        local_tts,          # LocalTTSClient
        cloud_timeout: float = 10.0,
    ):
        self._cloud_tts = cloud_tts
        self._local_ollama = local_ollama
        self._local_tts = local_tts
        self._cloud_timeout = cloud_timeout

        self.cloud_successes = 0
        self.local_fallbacks = 0
        self.consecutive_failures = 0

    def get_response_and_play(self, user_text: str) -> Tuple[str, str]:
        """
        获取 AI 回复并播放语音。
        返回 (response_text, source)，source 为 "cloud" 或 "local"。
        """
        # 尝试云端
        if self._cloud_tts._sock or self._cloud_tts.connect():
            result_q: queue.Queue = queue.Queue()

            def _try():
                try:
                    self._cloud_tts.send_text_and_play(user_text)
                    result_q.put(("ok", ""))
                except Exception as e:
                    result_q.put(("error", str(e)))

            t = threading.Thread(target=_try, daemon=True)
            t.start()
            t.join(timeout=self._cloud_timeout)

            if t.is_alive():
                logger.warning("云端 TTS 超时 (%.1fs)，切到本地", self._cloud_timeout)
                self._cloud_tts.close()
                return self._fallback_local(user_text)

            try:
                status, detail = result_q.get_nowait()
            except queue.Empty:
                return self._fallback_local(user_text)

            if status == "ok":
                self.cloud_successes += 1
                self.consecutive_failures = 0
                return ("", "cloud")

            logger.warning("云端 TTS 错误: %s，切到本地", detail)
            self._cloud_tts.close()

        return self._fallback_local(user_text)

    def _fallback_local(self, user_text: str) -> Tuple[str, str]:
        self.local_fallbacks += 1
        self.consecutive_failures += 1

        try:
            response = self._local_ollama.generate_full(user_text)
            if not response:
                response = "抱歉，我暂时无法回答这个问题。"
        except Exception as e:
            logger.error("本地 Ollama 失败: %s", e)
            response = "抱歉，本地服务暂时不可用，请稍后重试。"

        logger.info("本地 LLM 回复: %s", response[:80])

        try:
            self._local_tts.speak(response)
        except Exception as e:
            logger.error("本地 TTS 失败: %s", e)

        return (response, "local")
