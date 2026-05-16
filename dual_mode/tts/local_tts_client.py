# -*- coding: utf-8 -*-
"""
本地 TTS（edge-tts），云端不可用时的兜底方案。
使用 Microsoft Edge 免费 TTS API，中文质量好。
"""

import asyncio
import io
import logging
import tempfile
import threading

logger = logging.getLogger(__name__)

try:
    import edge_tts
    _edge_tts_ok = True
except ImportError:
    edge_tts = None
    _edge_tts_ok = False

try:
    import pyaudio
    _pyaudio_ok = True
except ImportError:
    pyaudio = None
    _pyaudio_ok = False


class LocalTTSClient:
    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

    def __init__(self, voice: str = DEFAULT_VOICE, sample_rate: int = 24000):
        if not _edge_tts_ok:
            raise ImportError("需要 edge_tts: pip install edge-tts")
        if not _pyaudio_ok:
            raise ImportError("需要 pyaudio: pip install pyaudio")

        self.voice = voice
        self.sample_rate = sample_rate
        self._pyaudio: Optional[pyaudio.PyAudio] = None

        # Background event loop for async edge-tts
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _synthesize(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, self.voice)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def _mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("需要 pydub: pip install pydub")

        seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        seg = seg.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
        return seg.raw_data

    def speak(self, text: str) -> None:
        """合成并播放语音。"""
        if not text.strip():
            return

        logger.info("本地 TTS 合成: %s", text[:50])

        future = asyncio.run_coroutine_threadsafe(self._synthesize(text), self._loop)
        try:
            mp3_data = future.result(timeout=30)
        except Exception as e:
            logger.error("TTS 合成失败: %s", e)
            raise

        pcm = self._mp3_to_pcm(mp3_data)

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

    def close(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
