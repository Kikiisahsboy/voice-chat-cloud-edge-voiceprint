# -*- coding: utf-8 -*-
"""本地流式 TTS 客户端 — edge-tts 流式合成 + miniaudio 解码 + PyAudio 播放。"""

import asyncio
import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pyaudio
    _pyaudio_ok = True
except ImportError:
    pyaudio = None
    _pyaudio_ok = False

import edge_tts
import miniaudio


class StreamingLocalTTS:
    """流式 TTS：边合成边播放，支持随时打断。"""

    VOICE = "zh-CN-XiaoxiaoNeural"
    STREAM_CHUNK = 1024  # MP3 解码后的 PCM 帧大小

    def __init__(self, sample_rate: int = 24000):
        if not _pyaudio_ok:
            raise ImportError("需要 pyaudio: pip install pyaudio")

        self.sample_rate = sample_rate
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stop_event = threading.Event()
        self._playing = False

        # Async event loop
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _stream_tts(self, text: str, pcm_queue: queue.Queue):
        """异步流式获取 edge-tts 音频块，解码后放入队列。"""
        communicate = edge_tts.Communicate(text, self.VOICE)
        mp3_buffer = bytearray()

        async for chunk in communicate.stream():
            if self._stop_event.is_set():
                break
            if chunk["type"] == "audio":
                mp3_buffer.extend(chunk["data"])
                # 每积累一定量就解码
                if len(mp3_buffer) >= 4096:
                    pcm = self._decode_mp3(bytes(mp3_buffer))
                    if pcm:
                        pcm_queue.put(pcm)
                    mp3_buffer.clear()

        # 解析剩余
        if mp3_buffer and not self._stop_event.is_set():
            pcm = self._decode_mp3(bytes(mp3_buffer))
            if pcm:
                pcm_queue.put(pcm)

        pcm_queue.put(None)  # 结束标记

    def _decode_mp3(self, data: bytes) -> Optional[bytes]:
        """用 miniaudio 解码 MP3 为 int16 PCM。"""
        try:
            decoded = miniaudio.decode(data, output_format=miniaudio.SampleFormat.SIGNED16,
                                        nchannels=1, sample_rate=self.sample_rate)
            return decoded.samples.tobytes()
        except Exception as e:
            logger.debug("MP3 解码失败: %s", e)
            return None

    def speak(self, text: str) -> bool:
        """
        流式合成并播放。返回 True 表示正常完成，False 表示被打断。
        """
        if not text.strip():
            return True

        logger.info("本地 TTS 流式合成: %s", text[:50])
        self._stop_event.clear()
        self._playing = True

        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.STREAM_CHUNK,
        )

        pcm_queue: queue.Queue = queue.Queue()

        # 启动异步合成
        future = asyncio.run_coroutine_threadsafe(
            self._stream_tts(text, pcm_queue), self._loop)

        first_chunk = True
        interrupted = False

        try:
            while True:
                if self._stop_event.is_set():
                    interrupted = True
                    break
                try:
                    pcm = pcm_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if pcm is None:  # 结束
                    break
                if first_chunk:
                    logger.info("TTS 开始播放")
                    first_chunk = False
                stream.write(pcm)
        finally:
            stream.stop_stream()
            stream.close()
            future.cancel()
            self._playing = False

        if interrupted:
            logger.info("TTS 被用户打断")
            return False
        logger.info("TTS 播放完毕")
        return True

    def stop(self):
        """打断当前播放。"""
        if self._playing:
            self._stop_event.set()
            logger.info("TTS 打断信号已发送")

    def close(self):
        self.stop()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
