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

# ── 懒加载 ────────────────────────────────────────────
_pyaudio = None
_AudioCapture, _WakeWordDetector = None, None
_CloudTTSClient, _LLMRouter = None, None
_LocalOllamaClient, _LocalTTSClient = None, None
_SpeechBrainEngine, _SpeakerManager = None, None
_LocalASREngine = None
_import_errors: dict = {}

try:
    import pyaudio as _pyaudio
except ImportError as e:
    _import_errors['pyaudio'] = str(e)
try:
    from dual_mode.audio.audio_capture import AudioCapture as _AudioCapture
except ImportError as e:
    _import_errors['audio_capture'] = str(e)
try:
    from dual_mode.audio.wake_word import WakeWordDetector as _WakeWordDetector
except ImportError as e:
    _import_errors['wake_word'] = str(e)
try:
    from dual_mode.llm.llm_router import CloudTTSClient as _CloudTTSClient
    from dual_mode.llm.llm_router import LLMRouter as _LLMRouter
except ImportError as e:
    _import_errors['llm_router'] = str(e)
try:
    from dual_mode.llm.local_ollama_client import LocalOllamaClient as _LocalOllamaClient
except ImportError as e:
    _import_errors['local_ollama'] = str(e)
try:
    from dual_mode.tts.local_tts_client import StreamingLocalTTS as _LocalTTSClient
except ImportError as e:
    _import_errors['local_tts'] = str(e)
try:
    from dual_mode.asr.local_asr import LocalASREngine as _LocalASREngine
except ImportError as e:
    _import_errors['local_asr'] = str(e)
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

    def __init__(self, config_path: str):
        cfg = ConfigLoader(config_path)
        self.cfg = cfg

        # ── 唤醒词 ──────────────────────────
        self.wake_detector = None
        if cfg.wake_word.enabled and _WakeWordDetector:
            try:
                self.wake_detector = _WakeWordDetector(
                    model=cfg.wake_word.model,
                    threshold=cfg.wake_word.threshold,
                )
            except Exception as e:
                logger.warning("唤醒词加载失败: %s", e)

        # ── 本地 ASR ────────────────────────
        self.local_asr = None
        if _LocalASREngine and os.path.isdir(cfg.asr.local.model_path):
            try:
                self.local_asr = _LocalASREngine(
                    model_path=cfg.asr.local.model_path,
                    sample_rate=cfg.audio.sample_rate,
                )
            except Exception as e:
                logger.warning("本地 ASR 初始化失败: %s", e)

        # ── 音频捕获 ────────────────────────
        self.audio_capture = None
        if _AudioCapture:
            self.audio_capture = _AudioCapture(
                asr_host=cfg.servers.cloud.asr.host,
                asr_port=cfg.servers.cloud.asr.port,
                sample_rate=cfg.audio.sample_rate,
                frames_per_buffer=cfg.audio.frames_per_buffer,
                silence_timeout_s=cfg.audio.silence_timeout_seconds,
                listen_timeout_s=cfg.conversation.listen_timeout_seconds,
                local_asr=self.local_asr,
            )

        # ── 云端 TTS ────────────────────────
        self.cloud_tts = None
        if _CloudTTSClient:
            self.cloud_tts = _CloudTTSClient(
                host=cfg.servers.cloud.tts.host,
                port=cfg.servers.cloud.tts.port,
                sample_rate=cfg.audio.tts_sample_rate,
            )

        # ── 本地 LLM ────────────────────────
        self.local_ollama = None
        if _LocalOllamaClient:
            prompt = ""
            pp = "dual_mode/prompts/local_fallback.txt"
            if os.path.exists(pp):
                with open(pp, 'r', encoding='utf-8') as f:
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

        # ── 本地 TTS ────────────────────────
        self.local_tts = None
        if _LocalTTSClient:
            try:
                self.local_tts = _LocalTTSClient(
                    sample_rate=cfg.audio.tts_sample_rate,
                )
            except Exception as e:
                logger.warning("本地 TTS 初始化失败: %s", e)

        # ── LLM 路由 ────────────────────────
        self.llm_router = None
        if _LLMRouter and self.cloud_tts and self.local_ollama and self.local_tts:
            self.llm_router = _LLMRouter(
                cloud_tts=self.cloud_tts,
                local_ollama=self.local_ollama,
                local_tts=self.local_tts,
                cloud_timeout=cfg.llm.cloud.tts_timeout_seconds,
            )

        # ── 声纹 ────────────────────────────
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
            except Exception as e:
                logger.warning("声纹初始化失败: %s", e)

        self._player = _pyaudio.PyAudio() if _pyaudio else None
        self.mode = "local"       # cloud | local
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
                emb = self.speaker_engine.extract_embedding(audio, self.cfg.audio.sample_rate)
                name, score = self.speaker_manager.identify(emb)
                self.last_speaker = name
                if name:
                    logger.info(">>> 说话人: %s (%.3f) <<<", name, score)
                else:
                    logger.info(">>> 说话人: 未知 <<<")
            except Exception as e:
                logger.error("声纹识别错误: %s", e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=2.0)  # 等声纹结果出来再进入 SPEAKING

    def _build_llm_text(self, raw_text: str) -> str:
        """构建发给 LLM 的文本，包含说话人信息。"""
        if self.last_speaker:
            return f"{self.last_speaker}说：{raw_text}"
        return raw_text

    # ==================================================================
    # 注册声纹
    # ==================================================================

    # 声纹注册朗读文本（语音学覆盖多样性）
    ENROLL_SENTENCES = [
        "你好，我是来注册声纹的，今天天气真不错。",
        "语音识别可以帮助我们更好地与机器交互。",
        "床前明月光，疑是地上霜，举头望明月，低头思故乡。",
        "人工智能正在改变世界的每一个角落。",
        "春天来了，花儿开了，小鸟在枝头唱歌。",
    ]

    def enroll_speaker(self, name: str, num: int = 3):
        if self.speaker_engine is None or self.speaker_manager is None:
            print("声纹识别未启用。")
            return
        if self.audio_capture is None:
            print("音频捕获未启用。")
            return

        print(f"\n=== 声纹注册: {name} ===")
        print("请朗读以下句子，每句读完后按 Enter 结束。\n")

        sentences = self.ENROLL_SENTENCES[:num] if num <= len(self.ENROLL_SENTENCES) else \
            self.ENROLL_SENTENCES * (num // len(self.ENROLL_SENTENCES) + 1)
        sentences = sentences[:num]

        embeddings = []
        for i, text in enumerate(sentences):
            print(f"  [{i+1}/{num}] 请朗读:")
            print(f"  \033[1;36m\"{text}\"\033[0m")
            input(f"  准备好了按 Enter 开始录音...")

            frames = []
            with self.audio_capture as cap:
                cap.calibrate_noise(duration=0.5)
                cap.flush()
                cap._start_stream()
                print(f"  \033[32m录音中... 读完后按 Enter 结束\033[0m")
                stop_flag = threading.Event()
                def _wait():
                    input()
                    stop_flag.set()
                threading.Thread(target=_wait, daemon=True).start()
                while not stop_flag.is_set():
                    try:
                        data = cap._queue.get(timeout=0.1)
                        frames.append(data)
                    except Exception:
                        pass
                cap._stop_stream()
                cap.disconnect_asr()

            if not frames:
                print("  未捕获到音频，跳过")
                continue

            import numpy as np
            raw = b"".join(frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            dur = len(audio) / self.cfg.audio.sample_rate
            if dur < 1.0:
                print(f"  录制时长过短 ({dur:.1f}s < 1s)，请重试")
                continue
            print(f"  录制完成 ({dur:.1f}s)")
            emb = self.speaker_engine.extract_embedding(audio, self.cfg.audio.sample_rate)
            embeddings.append(emb)

        if embeddings:
            self.speaker_manager.enroll(name, embeddings)
            print(f"\n✓ '{name}' 声纹注册成功！({len(embeddings)} 条样本)")
        else:
            print("\n✗ 注册失败：未收集到足够长度的语音样本")

    # ==================================================================
    # 模式选择
    # ==================================================================

    def _select_mode(self):
        """启动前选择云/本地模式。"""
        print("\n" + "=" * 45)
        print("  选择运行模式:")
        print("    [1] 云端模式（ASR + LLM + CosyVoice2）")
        print("    [2] 本地模式（Vosk + Ollama + edge-tts）")
        print("    [3] 自动（云端优先，不可用切本地）")
        print("=" * 45)

        try:
            choice = input("请输入 (1/2/3，默认 3): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "3"

        use_cloud = False
        if choice == "1":
            use_cloud = True
            self.mode = "cloud"
            print("→ 云端模式\n")
        elif choice == "2":
            use_cloud = False
            self.mode = "local"
            print("→ 本地模式\n")
        else:
            use_cloud = True   # 先试云端
            self.mode = "auto"
            print("→ 自动模式（云端优先）\n")

        # 连接 ASR
        if use_cloud:
            # 强制先试云端
            ok = self.audio_capture.connect_asr()
            if ok and self.audio_capture.is_cloud_asr:
                logger.info("✓ 云端 ASR 连接成功")
            elif ok and not self.audio_capture.is_cloud_asr:
                logger.info("→ 云端 ASR 不可达，已切本地 Vosk")
                self.mode = "local"
            else:
                logger.info("→ ASR 连接失败，使用本地 Vosk")
                self.mode = "local"
                # 强制本地连接
                self.audio_capture._use_cloud_asr = False
                if self.audio_capture._local_asr:
                    self.audio_capture._local_asr.reset()

        # 连接云端 TTS（如果云端模式）
        if self.mode == "cloud" and self.cloud_tts:
            if self.cloud_tts.connect():
                logger.info("✓ 云端 TTS 连接成功")
            else:
                logger.info("→ 云端 TTS 不可达，切本地")
                self.mode = "local"
        elif self.mode == "local":
            if self.local_ollama:
                if self.local_ollama.health_check():
                    logger.info("✓ 本地 Ollama (%s) 就绪", self.local_ollama.model)
                else:
                    logger.warning("⚠ 本地 Ollama 未就绪")

        # 最终确认
        if not self.audio_capture._use_cloud_asr and not self.audio_capture._local_asr:
            print("所有 ASR 均不可用，无法启动。")
            return False
        return True

    # ==================================================================
    # 按键触发录音（替代固定超时）
    # ==================================================================

    def _listen_with_key(self, cap) -> Optional[str]:
        """
        按 Enter 开始说话，再按 Enter 结束。
        返回 ASR 识别文本，或 None。
        """
        # 清空旧音频缓存，避免识别到之前的语音
        cap.flush()
        input("按 Enter 开始说话...")

        cap.accumulate_audio(True)

        # 在单独线程中运行 listen_and_stream
        results = []
        done = threading.Event()

        def _listen():
            try:
                for r in cap.listen_and_stream():
                    if r:
                        results.append(r)
                        if r.get("text"):
                            break
            except Exception as e:
                logger.error("ASR 错误: %s", e)
            done.set()

        listen_thread = threading.Thread(target=_listen, daemon=True)
        listen_thread.start()

        # 等待用户按 Enter 结束或自动结束
        def _wait_key():
            try:
                input()
            except Exception:
                pass
            # 用户按了 Enter → 手动结束
            logger.info("用户手动结束录音")

        key_thread = threading.Thread(target=_wait_key, daemon=True)
        key_thread.start()

        # 等 listen 完成或用户按 Enter
        while not done.is_set():
            if not key_thread.is_alive():
                # 用户按了 Enter → 还需要给一点时间让 ASR 出结果
                key_thread.join(timeout=0)
                time.sleep(0.5)  # 等最后的识别
                break
            done.wait(timeout=0.1)

        vp_audio = cap.get_accumulated_audio()
        if vp_audio is not None:
            self._identify_async(vp_audio)

        # 取最终结果
        final = ""
        for r in reversed(results):
            if r and r.get("text"):
                final = r["text"].strip().replace(" ", "")
                break
        if not final and results:
            # 取最后一个 partial
            for r in reversed(results):
                if r and r.get("partial"):
                    final = r["partial"].strip()
                    break

        return final or None

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
            print("  操作: 按 Enter 开始说话，再按 Enter 结束")
        if self.speaker_manager and self.speaker_manager.speaker_count > 0:
            print(f"  已注册: {', '.join(self.speaker_manager.list_enrolled())}")
        print(f"  模式: {'云端 (CosyVoice2)' if self.mode == 'cloud' else '本地 (Vosk+Ollama+edge-tts)'}")
        print("  按 Enter 打断 AI 说话 | 说 '退出/再见' 结束")
        print("=" * 55)

        # 选择模式
        if not self._select_mode():
            return

        with self.audio_capture as cap:
            cap.calibrate_noise()

            state = AssistantState.IDLE

            while True:
                # ── IDLE ─────────────────────────────
                if state == AssistantState.IDLE:
                    if self.wake_detector:
                        print("\n[待机] 等待唤醒词...")
                        self.wake_detector.reset()
                        found = cap.wait_for_wake_word(self.wake_detector)
                        if found:
                            self._play_wav(self.cfg.sounds.ding)
                            state = AssistantState.LISTENING
                    else:
                        state = AssistantState.LISTENING

                # ── LISTENING ────────────────────────
                elif state == AssistantState.LISTENING:
                    self.last_speaker = None
                    final_text = self._listen_with_key(cap)

                    if final_text:
                        logger.info("ASR 最终: '%s'", final_text)
                        if final_text in self.cfg.conversation.exit_phrases:
                            self.last_text = "再见！"
                        else:
                            # 附上说话人身份
                            self.last_text = self._build_llm_text(final_text)
                        state = AssistantState.SPEAKING
                    else:
                        print("[未识别到语音]")
                        state = AssistantState.IDLE

                # ── SPEAKING ─────────────────────────
                elif state == AssistantState.SPEAKING:
                    label = "云端" if (self.llm_router and self.llm_router.consecutive_failures == 0
                                       and self.mode != "local") else "本地"
                    print(f"\n[播报: {label}] (按 Enter 打断)...")

                    interrupt = threading.Event()

                    def _wait_interrupt():
                        try:
                            input()
                            interrupt.set()
                        except Exception:
                            pass

                    threading.Thread(target=_wait_interrupt, daemon=True).start()

                    if self.llm_router:
                        resp_text = ""
                        result_done = threading.Event()

                        def _speak():
                            nonlocal resp_text
                            try:
                                resp_text, _ = self.llm_router.get_response_and_play(
                                    self.last_text)
                            except Exception as e:
                                logger.error("路由错误: %s", e)
                            result_done.set()

                        t = threading.Thread(target=_speak, daemon=True)
                        t.start()

                        while not result_done.is_set():
                            if interrupt.is_set():
                                logger.info("用户打断!")
                                self.llm_router.stop()
                                result_done.wait(timeout=0.5)
                                break
                            result_done.wait(timeout=0.1)

                        if resp_text:
                            print(f"AI: {resp_text}")
                    else:
                        logger.error("LLM 路由不可用")
                        time.sleep(0.5)

                    if self.last_text == "再见！":
                        break
                    state = AssistantState.IDLE

        self._close()

    def _print_deps_error(self):
        print("\n" + "=" * 55)
        print("  缺少运行依赖")
        print("=" * 55)
        for k, v in _import_errors.items():
            print(f"  [{k}] {v}")
        print()
        print("  安装: pip install pyaudio webrtcvad noisereduce openwakeword")
        print("        pip install requests edge-tts speechbrain pyyaml miniaudio")
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
                print("尚未注册。")
        else:
            print("声纹识别未启用。")
        return
    if args.enroll:
        orch.enroll_speaker(args.enroll)
        return

    orch.start_conversation()


if __name__ == "__main__":
    main()
