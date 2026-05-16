# -*- coding: utf-8 -*-
"""
Local TTS client using Microsoft Edge TTS (edge-tts).

Provides speech synthesis when cloud TTS is unavailable.
Uses the free Microsoft Edge TTS API with Chinese neural voices.
"""

import asyncio
import io
import logging
import tempfile
import threading
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)


class LocalTTSClient:
    """Edge-TTS based local speech synthesis."""

    # Good Chinese voices from edge-tts
    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = 24000,
        buffer_size: int = 4096,
    ):
        self.voice = voice
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._start_event_loop()

    def _start_event_loop(self):
        """Start a background asyncio event loop for edge-tts calls."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_pyaudio(self):
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

    async def _synthesize(self, text: str) -> bytes:
        """Async: synthesize text to MP3 bytes using edge-tts."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def _mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """Convert MP3 bytes to PCM int16 bytes at target sample rate."""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub is required for audio format conversion. "
                "Install with: pip install pydub"
            )

        seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        seg = seg.set_frame_rate(self.sample_rate).set_channels(
            1).set_sample_width(2)
        return seg.raw_data

    def speak(self, text: str) -> None:
        """
        Synthesize text and play audio via PyAudio.
        Blocks until playback completes.
        """
        if not text.strip():
            return

        logger.info("Local TTS synthesizing: %s", text[:50])

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._synthesize(text), self._loop)
            mp3_data = future.result(timeout=30)
        except Exception as e:
            logger.error("Local TTS synthesis failed: %s", e)
            raise

        try:
            pcm_data = self._mp3_to_pcm(mp3_data)
        except ImportError as e:
            logger.warning("pydub not available, playing MP3 directly: %s", e)
            # Fallback: save to temp file and play with a system player
            self._play_mp3_fallback(mp3_data)
            return

        self._ensure_pyaudio()
        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.buffer_size,
        )

        chunk_size = self.buffer_size * 2  # 2 bytes per sample for int16
        offset = 0
        while offset < len(pcm_data):
            chunk = pcm_data[offset:offset + chunk_size]
            stream.write(chunk)
            offset += chunk_size

        stream.stop_stream()
        stream.close()
        logger.info("Local TTS playback complete.")

    def _play_mp3_fallback(self, mp3_data: bytes):
        """Fallback: save MP3 to temp file and play with pyaudio."""
        self._ensure_pyaudio()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        try:
            tmp.write(mp3_data)
            tmp.close()

            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(tmp.name, format="mp3")
                seg = seg.set_frame_rate(self.sample_rate).set_channels(
                    1).set_sample_width(2)
                pcm = seg.raw_data
            except ImportError:
                logger.error("pydub not available for fallback playback")
                return

            stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
            )
            chunk_size = self.buffer_size * 2
            offset = 0
            while offset < len(pcm):
                stream.write(pcm[offset:offset + chunk_size])
                offset += chunk_size
            stream.stop_stream()
            stream.close()
        finally:
            import os
            os.unlink(tmp.name)

    def close(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
