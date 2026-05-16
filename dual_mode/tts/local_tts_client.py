# -*- coding: utf-8 -*-
"""本地 TTS 客户端（edge-tts），无需 ffmpeg。"""

import asyncio
import io
import logging
import os
import tempfile
import threading
import wave
from typing import Optional

try:
    import pyaudio
    _pyaudio_available = True
except ImportError:
    pyaudio = None
    _pyaudio_available = False

import edge_tts

logger = logging.getLogger(__name__)


class LocalTTSClient:
    """使用 Microsoft Edge TTS，输出 PCM 通过 PyAudio 播放。"""

    VOICE = "zh-CN-XiaoxiaoNeural"

    def __init__(self, sample_rate: int = 24000):
        if not _pyaudio_available:
            raise ImportError("需要 pyaudio: pip install pyaudio")

        self.sample_rate = sample_rate
        self._pyaudio: Optional[pyaudio.PyAudio] = None

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _synthesize(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, self.VOICE)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def _mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """将 MP3 解码为 PCM 16-bit mono，使用 pydub + ffmpeg。"""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("需要 pydub + ffmpeg: pip install pydub 并安装 ffmpeg")

        seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        seg = seg.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
        return seg.raw_data

    def _save_and_play_mp3(self, mp3_data: bytes):
        """备选方案：保存 MP3 到临时文件，用 PyAudio 播放 WAV。"""
        # 尝试用 pydub（需要 ffmpeg）
        try:
            pcm = self._mp3_to_pcm(mp3_data)
            self._play_pcm(pcm)
            return
        except ImportError:
            logger.debug("pydub/ffmpeg 不可用，尝试 soundfile...")
        except Exception as e:
            logger.debug("pydub 解码失败: %s", e)

        # 备选：保存为临时 MP3，用 wave 模块播放（需要先转 WAV）
        # 或者直接用 os.startfile 让系统默认播放器播放
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(mp3_data)
            tmp.close()
            os.startfile(tmp.name)
            # 等播放完
            import time
            est = max(2.0, len(mp3_data) / 2000)  # rough estimate
            time.sleep(est)
            os.unlink(tmp.name)
        except Exception as e:
            logger.error("所有播放方式均失败: %s", e)

    def _play_pcm(self, pcm: bytes):
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=4096,
        )

        chunk = 4096 * 2
        for i in range(0, len(pcm), chunk):
            stream.write(pcm[i:i + chunk])

        stream.stop_stream()
        stream.close()
        logger.info("本地 TTS 播放完毕")

    def speak(self, text: str):
        if not text.strip():
            return

        logger.info("本地 TTS 合成: %s", text[:50])

        future = asyncio.run_coroutine_threadsafe(self._synthesize(text), self._loop)
        try:
            mp3_data = future.result(timeout=30)
        except Exception as e:
            logger.error("TTS 合成失败: %s", e)
            raise

        self._save_and_play_mp3(mp3_data)

    def close(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
