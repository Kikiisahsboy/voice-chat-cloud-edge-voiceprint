# -*- coding: utf-8 -*-
"""
Parallel PyAudio capture for voiceprint audio collection.

Opens a second PyAudio input stream alongside AudioStreamClient.
On Linux with ALSA, the dsnoop plugin allows multiple processes/streams
to capture from the same hardware device simultaneously.
"""

import logging
import threading
from typing import Optional

import numpy as np
import pyaudio

logger = logging.getLogger(__name__)


class UtteranceCapture:
    """
    Captures raw audio from a parallel PyAudio stream for voiceprint extraction.

    Usage:
        capture = UtteranceCapture(sample_rate=16000)
        capture.start_capture()
        # ... user is speaking ...
        audio = capture.stop_capture()
        # audio is a float32 numpy array
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 4096,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer

        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._buffer: list = []  # Accumulates audio bytes
        self._lock = threading.Lock()
        self._capturing = False
        self._available = True  # Set to False if dual-stream not supported

    @property
    def is_available(self) -> bool:
        return self._available

    def start_capture(self):
        """Start capturing audio from a second PyAudio input stream."""
        if not self._available:
            return

        with self._lock:
            self._buffer.clear()

        try:
            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self._audio_callback,
            )
            self._capturing = True
            self._stream.start_stream()
            logger.info("UtteranceCapture started (parallel audio stream).")
        except OSError as e:
            logger.warning(
                "Cannot open second audio stream (dsnoop may not be available): %s. "
                "Voiceprint recognition will be disabled.", e)
            self._available = False
            self._cleanup_audio()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self._capturing:
            with self._lock:
                self._buffer.append(in_data)
        return (None, pyaudio.paContinue)

    def stop_capture(self) -> Optional[np.ndarray]:
        """
        Stop capturing and return accumulated audio as float32 numpy array.

        Returns:
            Float32 numpy array (mono) or None if no audio captured.
        """
        self._capturing = False

        if self._stream:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.warning("Error closing capture stream: %s", e)
            self._stream = None

        with self._lock:
            chunks = list(self._buffer)
            self._buffer.clear()

        self._cleanup_audio()

        if not chunks:
            logger.debug("No audio captured in this utterance.")
            return None

        raw = b"".join(chunks)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / self.sample_rate
        logger.info("UtteranceCapture stopped: %.2fs of audio captured.", duration)
        return audio

    def get_duration(self) -> float:
        """Get current duration of captured audio in seconds."""
        with self._lock:
            total_bytes = sum(len(c) for c in self._buffer)
        return total_bytes / (self.sample_rate * self.channels * 2)

    def clear(self):
        """Reset the audio buffer without stopping the stream."""
        with self._lock:
            self._buffer.clear()

    def _cleanup_audio(self):
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
