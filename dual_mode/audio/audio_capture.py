# -*- coding: utf-8 -*-
"""
Windows-compatible audio capture with VAD, noise reduction, and dual-mode ASR.

Supports both cloud ASR (TCP) and local ASR (Vosk). Falls back to local
when cloud is unreachable.
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
    """Cross-platform audio capture with VAD + dual-mode ASR."""

    def __init__(
        self,
        asr_host: str = "127.0.0.1",
        asr_port: int = 12306,
        sample_rate: int = 16000,
        frames_per_buffer: int = 2048,
        vad_aggressiveness: int = 3,
        silence_timeout_s: float = 1.5,
        listen_timeout_s: float = 10.0,
        local_asr=None,  # LocalASREngine instance (optional)
    ):
        _ensure_deps()

        self.asr_host = asr_host
        self.asr_port = asr_port
        self.sample_rate = sample_rate
        self.frames_per_buffer = frames_per_buffer
        self.silence_timeout_s = silence_timeout_s
        self.listen_timeout_s = listen_timeout_s

        # VAD — fixed 30ms frames for WebRTC
        self.vad = _webrtcvad.Vad(vad_aggressiveness)
        self.vad_frame_ms = 30
        self.vad_frame_samples = int(sample_rate * 30 / 1000)
        self.vad_frame_bytes = self.vad_frame_samples * 2

        # ASR mode
        self._local_asr = local_asr
        self._use_cloud_asr = False

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

    # ===== Stream =====

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

    # ===== Noise =====

    def calibrate_noise(self, duration: float = 1.0):
        if not self._noisereduce_available:
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
        """尝试云端 ASR，不可达则切本地 Vosk。"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2.0)
            self._sock.connect((self.asr_host, self.asr_port))
            self._sock.settimeout(0.1)
            self._use_cloud_asr = True
            logger.info("已连接云端 ASR %s:%d", self.asr_host, self.asr_port)
            return True
        except Exception as e:
            logger.warning("云端 ASR 不可用 (%s)", e)
            self._sock = None

        if self._local_asr is not None:
            self._use_cloud_asr = False
            self._local_asr.reset()
            logger.info("已切换到本地 Vosk ASR")
            return True

        logger.error("无可用的 ASR")
        return False

    @property
    def is_cloud_asr(self) -> bool:
        return self._use_cloud_asr

    def disconnect_asr(self):
        if self._sock:
            try:
                self._sock.sendall(struct.pack('>I', 0))
            except Exception:
                pass
            self._sock.close()
            self._sock = None
        if self._local_asr:
            self._local_asr.reset()

    # ===== Wake Word =====

    def wait_for_wake_word(self, wake_word_detector) -> bool:
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

    # ===== ASR helpers =====

    def _send_asr(self, data: bytes):
        if self._use_cloud_asr and self._sock and len(data) > 0:
            self._sock.sendall(struct.pack('>I', len(data)) + data)

    def _send_local_asr(self, data: bytes) -> Optional[dict]:
        if not self._use_cloud_asr and self._local_asr and data:
            return self._local_asr.accept_waveform(bytes(data))
        return None

    def _recv_asr(self) -> Optional[dict]:
        if self._use_cloud_asr:
            return self._recv_cloud_asr()
        return None

    def _recv_cloud_asr(self) -> Optional[dict]:
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

    def _local_final_result(self) -> Optional[dict]:
        if not self._use_cloud_asr and self._local_asr:
            return self._local_asr.final_result()
        return None

    # ===== Listen + Stream =====

    def listen_and_stream(self) -> Generator[Optional[Dict[str, Any]], None, None]:
        """Listen for speech, stream to ASR (cloud or local), yield results."""

        # VAD state
        voiced_history = collections.deque(maxlen=15)
        send_buffer = bytearray()
        last_activity = time.time()
        last_heartbeat = time.time()
        is_speaking = False
        speech_start_hits = 0
        silence_hits = 0

        send_chunk_bytes = self.vad_frame_bytes * 10  # ~300ms
        silence_hits_needed = int(self.silence_timeout_s / (self.vad_frame_ms / 1000))

        label = "云端" if self._use_cloud_asr else "本地Vosk"
        logger.info("开始监听... (%s)", label)

        if hasattr(self, '_accumulate') and self._accumulate:
            self._accumulated_audio = bytearray()

        vad_buffer = b''

        try:
            while True:
                if time.time() - last_activity > self.listen_timeout_s:
                    logger.info("聆听超时")
                    if len(send_buffer) > 0 and self._use_cloud_asr:
                        self._send_asr(send_buffer)
                    if not self._use_cloud_asr:
                        final = self._local_final_result()
                        if final:
                            yield final
                    return

                # Heartbeat (cloud only)
                if not is_speaking and self._use_cloud_asr and time.time() - last_heartbeat > 5.0:
                    try:
                        self._sock.sendall(struct.pack('>I', 0))
                        last_heartbeat = time.time()
                    except Exception:
                        logger.error("心跳发送失败")
                        return

                # Get audio
                try:
                    data = self._queue.get(timeout=0.05)
                except queue.Empty:
                    # Check cloud ASR results
                    if self._use_cloud_asr:
                        result = self._recv_cloud_asr()
                        if result:
                            last_activity = time.time()
                            yield result
                            if result.get("text"):
                                return
                    continue

                # Voiceprint accumulation
                if hasattr(self, '_accumulate') and self._accumulate:
                    self._accumulated_audio.extend(data)

                audio_np = np.frombuffer(data, dtype=np.int16)
                audio_np = self._reduce_noise(audio_np)
                vad_buffer += audio_np.tobytes()

                # VAD frame-by-frame
                while len(vad_buffer) >= self.vad_frame_bytes:
                    frame = vad_buffer[:self.vad_frame_bytes]
                    vad_buffer = vad_buffer[self.vad_frame_bytes:]

                    is_voice = self.vad.is_speech(frame, self.sample_rate)
                    voiced_history.append(is_voice)
                    voice_ratio = sum(voiced_history) / len(voiced_history)

                    if is_voice:
                        speech_start_hits += 1
                        silence_hits = 0
                    else:
                        silence_hits += 1
                        speech_start_hits = 0

                    if not is_speaking:
                        if voice_ratio >= 0.5 and speech_start_hits >= 3:
                            is_speaking = True
                            logger.info("语音开始")
                            last_activity = time.time()
                    else:
                        # Send while speaking
                        send_buffer.extend(frame)
                        last_activity = time.time()

                        # Send to ASR in chunks
                        while len(send_buffer) >= send_chunk_bytes:
                            chunk = bytes(send_buffer[:send_chunk_bytes])
                            send_buffer = send_buffer[send_chunk_bytes:]
                            self._send_asr(chunk)
                            # Local ASR: send and yield results immediately
                            local_r = self._send_local_asr(chunk)
                            if local_r:
                                yield local_r

                        # End detection
                        if silence_hits >= silence_hits_needed:
                            logger.info("语音结束")
                            if len(send_buffer) > 0:
                                self._send_asr(bytes(send_buffer))
                                local_r = self._send_local_asr(bytes(send_buffer))
                                if local_r:
                                    yield local_r

                            if self._use_cloud_asr:
                                self._send_asr(b'\x00')  # EOS
                                while True:
                                    r = self._recv_cloud_asr()
                                    if r:
                                        yield r
                                        if r.get("text"):
                                            return
                                    else:
                                        break
                            else:
                                final = self._local_final_result()
                                if final:
                                    yield final
                            return

        except (ConnectionResetError, BrokenPipeError) as e:
            logger.error("ASR 连接断开: %s", e)
            raise

    # ===== Raw audio accumulation (for voiceprint) =====

    def accumulate_audio(self, enable: bool = True):
        self._accumulate = enable
        if enable:
            self._accumulated_audio = bytearray()

    def get_accumulated_audio(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_accumulated_audio') or not self._accumulated_audio:
            return None
        raw = bytes(self._accumulated_audio)
        self._accumulated_audio = bytearray()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
