# -*- coding: utf-8 -*-
"""
云边双模语音助手 + 声纹识别 (Windows 兼容版)

Usage:
  python orchestrator.py                        # 正常对话
  python orchestrator.py --enroll <姓名>        # 注册声纹
  python orchestrator.py --list-speakers        # 列出已注册说话人
"""

import logging
import os
import sys
import threading
import time
import wave
from enum import Enum, auto
from typing import Optional

# ── 项目路径 ──────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dual_mode.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── 懒加载依赖 ────────────────────────────────────────
_pyaudio = None
_AudioCapture = None
_WakeWordDetector = None
_CloudTTSClient = None
_LLMRouter = None
_LocalOllamaClient = None
_LocalTTSClient = None
_SpeechBrainEngine = None
_SpeakerManager = None
_LocalASREngine = None

_import_errors: dict = {}

# PyAudio (硬件依赖)
try:
    import pyaudio as _pyaudio
except ImportError as e:
    _import_errors['pyaudio'] = str(e)

# 音频捕获
try:
    from dual_mode.audio.audio_capture import AudioCapture as _AudioCapture
except ImportError as e:
    _import_errors['audio_capture'] = str(e)

# 唤醒词
try:
    from dual_mode.audio.wake_word import WakeWordDetector as _WakeWordDetector
except ImportError as e:
    _import_errors['wake_word'] = str(e)

# TTS 路由
try:
    from dual_mode.llm.llm_router import CloudTTSClient as _CloudTTSClient
    from dual_mode.llm.llm_router import LLMRouter as _LLMRouter
except ImportError as e:
    _import_errors['llm_router'] = str(e)

# 本地 LLM
try:
    from dual_mode.llm.local_ollama_client import LocalOllamaClient as _LocalOllamaClient
except ImportError as e:
    _import_errors['local_ollama'] = str(e)

# 本地 TTS
try:
    from dual_mode.tts.local_tts_client import LocalTTSClient as _LocalTTSClient
except ImportError as e:
    _import_errors['local_tts'] = str(e)

# 本地 ASR
try:
    from dual_mode.asr.local_asr import LocalASREngine as _LocalASREngine
except ImportError as e:
    _import_errors['local_asr'] = str(e)

# 声纹
try:
    from dual_mode.voiceprint.speechbrain_engine import SpeechBrainEngine as _SpeechBrainEngine
    from dual_mode.voiceprint.speaker_manager import SpeakerManager as _SpeakerManager
except ImportError as e:
    _import_errors['voiceprint'] = str(e)


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    SPEAKING = auto()


class DualModeOrchestrator:
    """Windows 兼容的云边双模语音助手。"""

    def __init__(self, config_path: str):
        cfg = ConfigLoader(config_path)
        self.cfg = cfg

        # ── 唤醒词 ──────────────────────────────
        self.wake_detector = None
        if cfg.wake_word.enabled and _WakeWordDetector:
            try:
                self.wake_detector = _WakeWordDetector(
                    model=cfg.wake_word.model,
                    threshold=cfg.wake_word.threshold,
                )
                logger.info("唤醒词已启用 (%s)", cfg.wake_word.model)
            except Exception as e:
                logger.warning("唤醒词加载失败: %s", e)

        # ── 本地 ASR ────────────────────────────
        self.local_asr = None
        if _LocalASREngine and os.path.isdir(cfg.asr.local.model_path):
            try:
                self.local_asr = _LocalASREngine(
                    model_path=cfg.asr.local.model_path,
                    sample_rate=cfg.audio.sample_rate,
                )
                logger.info("本地 ASR 就绪")
            except Exception as e:
                logger.warning("本地 ASR 初始化失败: %s", e)

        # ── 音频捕获 ────────────────────────────
        self.audio_capture = None
        if _AudioCapture:
            self.audio_capture = _AudioCapture(
                asr_host=cfg.servers.cloud.asr.host,
                asr_port=cfg.servers.cloud.asr.port,
                sample_rate=cfg.audio.sample_rate,
                frames_per_buffer=cfg.audio.frames_per_buffer,
                silence_timeout_s=cfg.audio.silence_timeout_seconds,
                listen_timeout_s=cfg.conversation.listen_timeout_seconds,
                local_asr=self.local_asr,  # 云端不可用时自动切换
            )

        # ── 云端 TTS ────────────────────────────
        self.cloud_tts = None
        if _CloudTTSClient:
            self.cloud_tts = _CloudTTSClient(
                host=cfg.servers.cloud.tts.host,
                port=cfg.servers.cloud.tts.port,
                sample_rate=cfg.audio.tts_sample_rate,
            )

        # ── 本地 LLM ────────────────────────────
        self.local_ollama = None
        if _LocalOllamaClient:
            prompt = ""
            prompt_path = "dual_mode/prompts/local_fallback.txt"
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            try:
                self.local_ollama = _LocalOllamaClient(
                    base_url=cfg.servers.local.ollama.base_url,
                    model=cfg.llm.local.model,
                    system_prompt=prompt,
                    temperature=cfg.llm.local.temperature,
                    num_predict=cfg.llm.local.num_predict,
                    timeout=cfg.llm.local.timeout_seconds,
                )
            except Exception as e:
                logger.warning("本地 Ollama 初始化失败: %s", e)

        # ── 本地 TTS ────────────────────────────
        self.local_tts = None
        if _LocalTTSClient:
            try:
                self.local_tts = _LocalTTSClient(
                    sample_rate=cfg.audio.tts_sample_rate,
                )
            except Exception as e:
                logger.warning("本地 TTS 初始化失败: %s", e)

        # ── LLM 路由 ────────────────────────────
        self.llm_router = None
        if _LLMRouter and self.cloud_tts and self.local_ollama and self.local_tts:
            self.llm_router = _LLMRouter(
                cloud_tts=self.cloud_tts,
                local_ollama=self.local_ollama,
                local_tts=self.local_tts,
                cloud_timeout=cfg.llm.cloud.tts_timeout_seconds,
            )

        # ── 声纹识别 ────────────────────────────
        self.speaker_manager = None
        self.speaker_engine = None
        if cfg.voiceprint.enabled and _SpeakerManager and _SpeechBrainEngine:
            try:
                self.speaker_engine = _SpeechBrainEngine(
                    model_id=cfg.voiceprint.model)
                self.speaker_manager = _SpeakerManager(
                    enrollment_dir=cfg.voiceprint.enrollment_dir,
                    embedding_dim=cfg.voiceprint.embedding_dim,
                    threshold=cfg.voiceprint.identification_threshold,
                )
                logger.info("声纹识别已启用 (%d 人已注册)",
                            self.speaker_manager.speaker_count)
            except Exception as e:
                logger.warning("声纹初始化失败: %s", e)

        # ── 音频播放器 ──────────────────────────
        self._player = _pyaudio.PyAudio() if _pyaudio else None

        # ── 状态 ────────────────────────────────
        self.last_text = ""
        self.last_speaker: Optional[str] = None

    # ==================================================================

    def _play_wav(self, path: str):
        if not self._player or not os.path.exists(path):
            return
        try:
            with wave.open(path, 'rb') as wf:
                s = self._player.open(
                    format=self._player.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )
                data = wf.readframes(1024)
                while data:
                    s.write(data)
                    data = wf.readframes(1024)
                s.stop_stream()
                s.close()
        except Exception as e:
            logger.error("播放失败: %s", e)

    def _identify_async(self, audio):
        if audio is None or self.speaker_manager is None or self.speaker_engine is None:
            return
        dur = len(audio) / self.cfg.audio.sample_rate
        if dur < self.cfg.voiceprint.min_audio_duration_seconds:
            return

        def _run():
            try:
                emb = self.speaker_engine.extract_embedding(
                    audio, self.cfg.audio.sample_rate)
                name, score = self.speaker_manager.identify(emb)
                self.last_speaker = name
                if name:
                    logger.info(">>> 说话人: %s (%.3f) <<<", name, score)
                else:
                    logger.info(">>> 说话人: 未知 <<<")
            except Exception as e:
                logger.error("声纹识别错误: %s", e)

        threading.Thread(target=_run, daemon=True).start()

    # ==================================================================
    # 注册声纹
    # ==================================================================

    def enroll_speaker(self, name: str, num: int = 3):
        if self.speaker_engine is None or self.speaker_manager is None:
            print("声纹识别未启用，请检查依赖。")
            return
        if self.audio_capture is None:
            print("音频捕获未启用。")
            return

        print(f"\n=== 声纹注册: {name} ===")
        print(f"请在听到提示后说话（共 {num} 次）\n")

        embeddings = []
        for i in range(num):
            input(f"[{i+1}/{num}] 按 Enter 开始录音...")

            # 录制一段时间的音频（不需要连接 ASR）
            frames = []
            with self.audio_capture as cap:
                cap.calibrate_noise(duration=0.5)
                cap._start_stream()
                print("   录音中... 说一句话")
                timeout = self.cfg.conversation.listen_timeout_seconds
                start = time.time()
                while time.time() - start < timeout:
                    try:
                        data = cap._queue.get(timeout=0.1)
                        frames.append(data)
                    except Exception:
                        pass
                cap._stop_stream()
                cap.disconnect_asr()

            if not frames:
                print("   未捕获到音频，跳过")
                continue

            import numpy as np
            raw = b"".join(frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            dur = len(audio) / self.cfg.audio.sample_rate
            print(f"   录制完成 ({dur:.1f}s)")

            emb = self.speaker_engine.extract_embedding(
                audio, self.cfg.audio.sample_rate)
            embeddings.append(emb)

        if embeddings:
            self.speaker_manager.enroll(name, embeddings)
            print(f"\n✓ '{name}' 声纹注册成功！")
        else:
            print("\n✗ 注册失败")

    # ==================================================================
    # 对话主循环
    # ==================================================================

    def start_conversation(self):
        if self.audio_capture is None:
            self._print_deps_error()
            return

        print("\n" + "=" * 55)
        print("  云边双模语音助手 (Windows)")
        if self.wake_detector:
            print(f"  唤醒词: {self.cfg.wake_word.model}")
        else:
            print("  唤醒词: 已禁用（按 Enter 开始说话）")
        if self.speaker_manager and self.speaker_manager.speaker_count > 0:
            names = ", ".join(self.speaker_manager.list_enrolled())
            print(f"  已注册说话人: {names}")
        print(f"  云端: {self.cfg.servers.cloud.tts.host}")
        print(f"  本地: {self.cfg.servers.local.ollama.base_url}")
        print("=" * 55)

        with self.audio_capture as cap:
            cap.calibrate_noise()
            if not cap.connect_asr():
                print("所有 ASR 均不可用，退出。")
                return

            mode = "云端" if cap.is_cloud_asr else "本地 Vosk"
            print(f"ASR 模式: {mode}")

            state = AssistantState.IDLE

            while True:
                # ── IDLE: 等待唤醒词 ──────────────────
                if state == AssistantState.IDLE:
                    print("\n[待机] 等待唤醒...")
                    if self.wake_detector:
                        self.wake_detector.reset()
                        found = cap.wait_for_wake_word(self.wake_detector)
                    else:
                        input("按 Enter 开始说话...")
                        found = True

                    if found:
                        self._play_wav(self.cfg.sounds.ding)
                        state = AssistantState.LISTENING

                # ── LISTENING: 听用户说话 ─────────────
                elif state == AssistantState.LISTENING:
                    print(f"\n[聆听中] (超时 {self.cfg.conversation.listen_timeout_seconds:.0f}s)...")

                    # 启用声纹音频累积
                    cap.accumulate_audio(True)

                    final_text = ""
                    try:
                        for result in cap.listen_and_stream():
                            if not result:
                                continue
                            print("\r" + " " * 80 + "\r", end="")

                            if result.get("text"):
                                final_text = result["text"].strip().replace(" ", "")
                                logger.info("ASR 最终: '%s'", final_text)
                                break
                            elif result.get("partial"):
                                p = result.get("partial", "").strip()
                                if p:
                                    print(f"  {p}", end="", flush=True)
                    except (ConnectionResetError, BrokenPipeError) as e:
                        logger.error("ASR 连接断开: %s，重连...", e)
                        if not cap.connect_asr():
                            break
                        continue

                    # 获取累积音频用于声纹识别
                    vp_audio = cap.get_accumulated_audio()
                    if vp_audio is not None:
                        self._identify_async(vp_audio)

                    if final_text:
                        if final_text in self.cfg.conversation.exit_phrases:
                            logger.info("退出命令")
                            self.last_text = "再见！"
                            state = AssistantState.SPEAKING
                        else:
                            self.last_text = final_text
                            state = AssistantState.SPEAKING
                    else:
                        print("\n[超时] 返回待机")
                        self._play_wav(self.cfg.sounds.dong)
                        state = AssistantState.IDLE

                # ── SPEAKING: LLM + TTS ───────────────
                elif state == AssistantState.SPEAKING:
                    label = "本地" if self.llm_router and self.llm_router.consecutive_failures > 0 else "云端"
                    print(f"\n[播报中: {label}]...")

                    if self.llm_router:
                        try:
                            resp_text, source = self.llm_router.get_response_and_play(
                                self.last_text)
                            if source == "local":
                                print(f"AI: {resp_text}")
                        except Exception as e:
                            logger.error("路由错误: %s", e)
                    else:
                        logger.error("LLM 路由不可用")

                    if self.last_text == "再见！":
                        break
                    state = AssistantState.LISTENING

        self._close()

    def _print_deps_error(self):
        print("\n" + "=" * 55)
        print("  缺少运行依赖")
        print("=" * 55)
        for k, v in _import_errors.items():
            print(f"  [{k}] {v}")
        print()
        print("  安装命令:")
        print("    pip install pyaudio webrtcvad noisereduce openwakeword")
        print("    pip install requests edge-tts pydub speechbrain pyyaml")
        print()

    def _close(self):
        if self.cloud_tts:
            self.cloud_tts.close()
        if self.local_tts:
            self.local_tts.close()
        if self._player:
            self._player.terminate()


# ==========================================================================
# 入口
# ==========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="云边双模语音助手")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--enroll", metavar="NAME", help="注册新说话人")
    parser.add_argument("--list-speakers", action="store_true", help="列出已注册说话人")
    args = parser.parse_args()

    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    orch = DualModeOrchestrator(config_path)

    if args.list_speakers:
        if orch.speaker_manager:
            names = orch.speaker_manager.list_enrolled()
            if names:
                print(f"已注册说话人 ({len(names)}):")
                for n in names:
                    print(f"  - {n}")
            else:
                print("尚未注册任何说话人。")
        else:
            print("声纹识别未启用。")
        return

    if args.enroll:
        orch.enroll_speaker(args.enroll)
        return

    orch.start_conversation()


if __name__ == "__main__":
    main()
