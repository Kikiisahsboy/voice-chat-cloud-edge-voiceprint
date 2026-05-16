# -*- coding: utf-8 -*-
"""
Windows-compatible audio capture with VAD, noise reduction, and ASR streaming.

Replaces the original AudioStreamClient (which depended on Linux Snowboy).
Uses PyAudio + WebRTC VAD + noisereduce — all cross-platform.
"""

import collections
import json
import logging
import queue
import socket
import struct
import threading
import time
from typing import Any, Dict, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Lazy hardware imports ---
_pyaudio = None
_webrtcvad = None
_nr = None

def _ensure_deps():
    global _pyaudio, _webrtcvad, _nr
    if _pyaudio is None:
        try:
            import pyaudio as _pyaudio
        except ImportError:
            raise ImportError("需要 PyAudio: pip install pyaudio")
    if _webrtcvad is None:
        try:
            import webrtcvad as _webrtcvad
        except ImportError:
            raise ImportError("需要 webrtcvad: pip install webrtcvad")
    if _nr is None:
        try:
            import noisereduce as _nr
        except ImportError:
            logger.warning("noisereduce 未安装，降噪功能禁用")


class AudioCapture:
    """
    Cross-platform audio capture with VAD and ASR streaming.

    Usage (context manager):
        with AudioCapture(asr_host, asr_port) as cap:
            cap.calibrate_noise()
            # Wake word detection
            cap.wait_for_wake_word(wake_word_callback)
            # Listen and get ASR results
            for result in cap.listen_and_stream():
                ...
    """

    def __init__(
        self,
        asr_host: str = "127.0.0.1",
        asr_port: int = 12306,
        sample_rate: int = 16000,
        frames_per_buffer: int = 2048,
        vad_aggressiveness: int = 3,
        vad_frame_ms: int = 30,
        silence_timeout_s: float = 1.5,
        speech_start_threshold: float = 0.5,
        speech_end_threshold: float = 0.3,
        listen_timeout_s: float = 10.0,
    ):
        _ensure_deps()

        self.asr_host = asr_host
        self.asr_port = asr_port
        self.sample_rate = sample_rate
        self.frames_per_buffer = frames_per_buffer
        self.silence_timeout_s = silence_timeout_s
        self.speech_start_threshold = speech_start_threshold
        self.speech_end_threshold = speech_end_threshold
        self.listen_timeout_s = listen_timeout_s

        # VAD
        if vad_frame_ms not in (10, 20, 30):
            raise ValueError("vad_frame_ms 必须是 10, 20, 或 30")
        self.vad = _webrtcvad.Vad(vad_aggressiveness)
        self.vad_frame_ms = vad_frame_ms
        self.vad_frame_samples = int(sample_rate * vad_frame_ms / 1000)
        self.vad_frame_bytes = self.vad_frame_samples * 2  # int16

        # State
        self._audio: Optional[_pyaudio.PyAudio] = None
        self._stream: Optional[_pyaudio.Stream] = None
        self._sock: Optional[socket.socket] = None
        self._noise_profile: Optional[np.ndarray] = None
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._read_thread: Optional[threading.Thread] = None
        self._noisereduce_available = _nr is not None

    # ===== Context Manager =====

    def __enter__(self):
        self._audio = _pyaudio.PyAudio()
        self._start_stream()
        return self

    def __exit__(self, *args):
        self._stop_stream()
        if self._audio:
            self._audio.terminate()
            self._audio = None
        self.disconnect_asr()

    # ===== Stream Management =====

    def _start_stream(self):
        if self._stream and self._stream.is_active():
            return
        self._stream = self._audio.open(
            format=_pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        logger.info("音频流已启动 (%.1fkHz)", self.sample_rate / 1000)

    def _stop_stream(self):
        self._running = False
        if self._read_thread:
            self._read_thread.join(timeout=1.0)
            self._read_thread = None
        if self._stream:
            if self._stream.is_active():
                self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info("音频流已停止")

    def _read_loop(self):
        while self._running:
            try:
                data = self._stream.read(self.frames_per_buffer, exception_on_overflow=False)
                if self._running:
                    self._queue.put(data)
            except Exception as e:
                logger.error("音频读取错误: %s", e)
                break

    # ===== Noise Calibration =====

    def calibrate_noise(self, duration: float = 1.0):
        """采集环境噪声用于降噪校准。"""
        if not self._noisereduce_available:
            logger.info("降噪未启用（noisereduce 未安装）")
            return
        logger.info("正在采集环境噪声...（请保持安静 %.1f 秒）", duration)
        frames = []
        num = int(self.sample_rate / self.frames_per_buffer * duration)
        for _ in range(num):
            try:
                data = self._queue.get(timeout=1.0)
                frames.append(np.frombuffer(data, dtype=np.int16))
            except queue.Empty:
                break
        if frames:
            self._noise_profile = np.concatenate(frames)
            logger.info("噪声校准完成")
        else:
            logger.warning("噪声采集失败")

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        if self._noise_profile is not None and self._noisereduce_available:
            return _nr.reduce_noise(
                y=audio, sr=self.sample_rate,
                y_noise=self._noise_profile, prop_decrease=0.7)
        return audio

    # ===== ASR Connection =====

    def connect_asr(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2.0)
            self._sock.connect((self.asr_host, self.asr_port))
            self._sock.settimeout(0.1)
            logger.info("已连接 ASR 服务器 %s:%d", self.asr_host, self.asr_port)
            return True
        except Exception as e:
            logger.error("连接 ASR 失败: %s", e)
            self._sock = None
            return False

    def disconnect_asr(self):
        if self._sock:
            try:
                self._sock.sendall(struct.pack('>I', 0))
            except Exception:
                pass
            self._sock.close()
            self._sock = None

    # ===== Wake Word Detection =====

    def wait_for_wake_word(self, wake_word_detector) -> bool:
        """
        Block until wake word is detected.

        Args:
            wake_word_detector: Object with a `detect(audio_bytes) -> bool` method.

        Returns True when wake word is detected.
        """
        logger.info("等待唤醒词...")
        while self._running:
            try:
                data = self._queue.get(timeout=0.1)
                if wake_word_detector.detect(data):
                    logger.info("唤醒词检测到!")
                    return True
            except queue.Empty:
                continue
        return False

    # ===== Listen + Stream to ASR =====

    def listen_and_stream(self) -> Generator[Optional[Dict[str, Any]], None, None]:
        """
        Listen for speech, stream to ASR, yield recognition results.

        Yields dicts with "partial" or "text" keys.
        """
        if not self._sock:
            logger.error("未连接 ASR 服务器")
            return

        # VAD state tracking
        voiced_history = collections.deque(maxlen=15)  # ~450ms padding
        is_speaking = False
        speech_frames = 0
        silence_frames = 0
        send_buffer = bytearray()
        last_activity = time.time()
        last_heartbeat = time.time()

        # Chunk size for sending to ASR
        send_chunk_frames = max(1, 300 // self.vad_frame_ms)  # ~300ms chunks
        send_chunk_bytes = self.vad_frame_bytes * send_chunk_frames
        min_speech_frames = max(1, int(0.5 / (self.vad_frame_ms / 1000)))  # 0.5s min
        max_silence_frames = max(1, int(self.silence_timeout_s / (self.vad_frame_ms / 1000)))

        vad_buffer = b''

        logger.info("开始监听用户语音...")

        # Initialize accumulation
        if hasattr(self, '_accumulate') and self._accumulate:
            self._accumulated_audio = bytearray()

        try:
            while True:
                # Timeout check
                if time.time() - last_activity > self.listen_timeout_s:
                    logger.info("聆听超时")
                    if len(send_buffer) > 0:
                        self._send_asr(send_buffer)
                    return

                # Heartbeat (only when not speaking)
                if not is_speaking and time.time() - last_heartbeat > 5.0:
                    try:
                        self._sock.sendall(struct.pack('>I', 0))
                        last_heartbeat = time.time()
                    except Exception:
                        logger.error("心跳发送失败")
                        return

                # Get audio data
                try:
                    data = self._queue.get(timeout=0.05)
                except queue.Empty:
                    # Check for ASR result
                    result = self._recv_asr_result()
                    if result:
                        last_activity = time.time()
                        yield result
                        if result.get("text"):
                            return
                    continue

                # Accumulate raw audio for voiceprint
                if hasattr(self, '_accumulate') and self._accumulate:
                    self._accumulated_audio.extend(data)

                # Process audio
                audio_np = np.frombuffer(data, dtype=np.int16)
                audio_np = self._reduce_noise(audio_np)
                vad_buffer += audio_np.tobytes()

                # VAD frame processing
                while len(vad_buffer) >= self.vad_frame_bytes:
                    frame = vad_buffer[:self.vad_frame_bytes]
                    vad_buffer = vad_buffer[self.vad_frame_bytes:]

                    is_voice = self.vad.is_speech(frame, self.sample_rate)
                    voiced_history.append(is_voice)

                    if is_voice:
                        speech_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        speech_frames = 0

                    voice_ratio = sum(voiced_history) / len(voiced_history)

                    if not is_speaking:
                        if voice_ratio >= self.speech_start_threshold and speech_frames >= 3:
                            is_speaking = True
                            logger.info("语音开始")
                            last_activity = time.time()
                    else:
                        if voice_ratio <= self.speech_end_threshold and silence_frames >= max_silence_frames:
                            is_speaking = False
                            logger.info("语音结束")
                            if len(send_buffer) > 0:
                                self._send_asr(send_buffer)
                                send_buffer = bytearray()

                # Send audio while speaking
                if is_speaking:
                    send_buffer.extend(audio_np.tobytes())
                    while len(send_buffer) >= send_chunk_bytes:
                        chunk = bytes(send_buffer[:send_chunk_bytes])
                        send_buffer = send_buffer[send_chunk_bytes:]
                        self._send_asr(chunk)

        except (ConnectionResetError, BrokenPipeError, socket.error) as e:
            logger.error("ASR 连接断开: %s", e)
            self.disconnect_asr()
            raise

    def _send_asr(self, data: bytes):
        if self._sock and len(data) > 0:
            self._sock.sendall(struct.pack('>I', len(data)) + data)

    def _recv_asr_result(self) -> Optional[Dict[str, Any]]:
        if not self._sock:
            return None
        try:
            self._sock.settimeout(0.01)
            len_bytes = self._sock.recv(4)
            if not len_bytes:
                raise ConnectionResetError("ASR 连接关闭")
            payload_len = struct.unpack('>I', len_bytes)[0]
            if payload_len == 0:
                return None
            payload = b''
            while len(payload) < payload_len:
                chunk = self._sock.recv(payload_len - len(payload))
                if not chunk:
                    raise ConnectionResetError("ASR 连接断开")
                payload += chunk
            return json.loads(payload.decode('utf-8'))
        except socket.timeout:
            return None

    # ===== Raw audio accumulation (for voiceprint) =====

    def accumulate_audio(self, enable: bool = True):
        """Enable/disable raw audio accumulation for voiceprint capture."""
        self._accumulate = enable
        if enable:
            self._accumulated_audio = bytearray()

    def get_accumulated_audio(self) -> Optional[np.ndarray]:
        """Return accumulated raw audio as float32 numpy array and reset."""
        if not hasattr(self, '_accumulated_audio') or not self._accumulated_audio:
            return None
        raw = bytes(self._accumulated_audio)
        self._accumulated_audio = bytearray()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
