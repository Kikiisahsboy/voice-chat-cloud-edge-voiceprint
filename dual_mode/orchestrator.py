# -*- coding: utf-8 -*-
"""
Dual-Mode Voice Assistant with Voiceprint Recognition.

Features:
  - Cloud-first LLM with automatic fallback to local Ollama
  - Multi-user speaker identification via WeSpeaker
  - All configuration in config.yaml

Usage:
  python dual_mode/orchestrator.py                     # Normal run
  python dual_mode/orchestrator.py --enroll <name>     # Enroll a speaker
  python dual_mode/orchestrator.py --list-speakers      # List enrolled speakers
"""

import logging
import os
import sys
import threading
import time
import wave
from enum import Enum, auto
from typing import Optional

import pyaudio

# Ensure we can import from sibling client/ directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from client.audio_stream_client import AudioStreamClient
from client.tts_client import TTSStreamClient

from dual_mode.utils.config_loader import ConfigLoader
from dual_mode.llm.local_ollama_client import LocalOllamaClient
from dual_mode.llm.llm_router import LLMRouter
from dual_mode.tts.local_tts_client import LocalTTSClient
from dual_mode.voiceprint.wespeaker_engine import WeSpeakerEngine
from dual_mode.voiceprint.speaker_manager import SpeakerManager
from dual_mode.audio.utterance_capture import UtteranceCapture

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class AssistantState(Enum):
    IDLE = auto()
    LISTENING_TO_USER = auto()
    ASSISTANT_SPEAKING = auto()


class DualModeOrchestrator:
    """Voice assistant with cloud/edge dual-mode LLM and voiceprint recognition."""

    def __init__(self, config_path: str):
        config = ConfigLoader(config_path)
        self.config = config

        # --- Audio stream client (reused from original) ---
        self.audio_stream_client = AudioStreamClient(
            asr_server_ip=config.servers.cloud.asr.host,
            asr_server_port=config.servers.cloud.asr.port,
            sample_rate=config.audio.asr_sample_rate,
            hotword_model_path=config.wake_word.model_path,
            hotword_sensitivity=config.wake_word.sensitivity,
            listen_turn_timeout_s=config.conversation.timeout_seconds,
        )

        # --- Cloud TTS client (reused from original) ---
        self.cloud_tts_client = TTSStreamClient(
            server_ip=config.servers.cloud.tts.host,
            server_port=config.servers.cloud.tts.port,
            sample_rate=config.audio.tts_sample_rate,
        )

        # --- Local LLM ---
        prompt_path = config.llm.local.get(
            'system_prompt_path', 'dual_mode/prompts/local_fallback.txt')
        system_prompt = ""
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()

        self.local_ollama = LocalOllamaClient(
            base_url=config.servers.local.ollama.base_url,
            model=config.llm.local.model,
            system_prompt=system_prompt,
            temperature=config.llm.local.temperature,
            num_predict=config.llm.local.num_predict,
            timeout=config.llm.local.timeout_seconds,
        )

        # --- Local TTS ---
        self.local_tts = LocalTTSClient(
            sample_rate=config.audio.tts_sample_rate,
        )

        # --- LLM Router ---
        self.llm_router = LLMRouter(
            tts_client=self.cloud_tts_client,
            local_ollama=self.local_ollama,
            local_tts=self.local_tts,
            cloud_timeout=config.llm.cloud.tts_timeout_seconds,
        )

        # --- Voiceprint ---
        self.speaker_manager: Optional[SpeakerManager] = None
        self.wespeaker_engine: Optional[WeSpeakerEngine] = None
        self.utterance_capture: Optional[UtteranceCapture] = None

        if config.voiceprint.enabled:
            try:
                self.wespeaker_engine = WeSpeakerEngine(
                    model_path=config.voiceprint.model_path,
                    device="cpu",
                )
                self.speaker_manager = SpeakerManager(
                    enrollment_dir=config.voiceprint.enrollment_dir,
                    embedding_dim=config.voiceprint.embedding_dim,
                    threshold=config.voiceprint.identification_threshold,
                )
                self.utterance_capture = UtteranceCapture(
                    sample_rate=config.audio.asr_sample_rate,
                    channels=config.audio.channels,
                    frames_per_buffer=config.audio.frames_per_buffer,
                )
                if self.utterance_capture.is_available:
                    logger.info(
                        "Voiceprint recognition enabled (%d enrolled).",
                        self.speaker_manager.speaker_count)
                else:
                    logger.warning(
                        "Voiceprint capture not available on this device.")
            except ImportError as e:
                logger.warning(
                    "Voiceprint dependencies missing: %s. Disabling voiceprint.", e)
            except Exception as e:
                logger.warning(
                    "Failed to initialize voiceprint: %s. Disabling.", e)

        # --- Audio player ---
        self._pyaudio_player = pyaudio.PyAudio()

        # --- State ---
        self.last_transcript = ""
        self.last_speaker: Optional[str] = None

    # =========================================================================
    # Sound effects
    # =========================================================================

    def _play_sound(self, file_path: str, suspend_mic: bool = True):
        if not os.path.exists(file_path):
            logger.warning("Sound file not found: %s", file_path)
            return
        if suspend_mic:
            self.audio_stream_client.suspend_audio_capture()
        try:
            with wave.open(file_path, 'rb') as wf:
                stream = self._pyaudio_player.open(
                    format=self._pyaudio_player.get_format_from_width(
                        wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
                stream.stop_stream()
                stream.close()
        except Exception as e:
            logger.error("Play sound failed: %s", e)
        finally:
            if suspend_mic:
                self.audio_stream_client.resume_audio_capture()

    # =========================================================================
    # Voiceprint helpers
    # =========================================================================

    def _identify_speaker_async(self, audio):
        """Run speaker identification in a daemon thread."""
        if audio is None or self.speaker_manager is None:
            return
        duration = len(audio) / self.config.audio.asr_sample_rate
        min_dur = self.config.voiceprint.min_audio_duration_seconds
        if duration < min_dur:
            logger.debug(
                "Audio too short for voiceprint (%.2fs < %.2fs)", duration, min_dur)
            return

        def _run():
            try:
                emb = self.wespeaker_engine.extract_embedding(
                    audio, self.config.audio.asr_sample_rate)
                speaker = self.speaker_manager.identify(emb)
                self.last_speaker = speaker
                if speaker:
                    logger.info(">>> Identified speaker: %s <<<", speaker)
                else:
                    logger.info(">>> Speaker: UNKNOWN <<<")
            except Exception as e:
                logger.error("Speaker identification error: %s", e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # =========================================================================
    # Enrollment mode
    # =========================================================================

    def enroll_speaker(self, name: str, num_samples: int = 3):
        """Interactive speaker enrollment."""
        if self.speaker_manager is None or self.wespeaker_engine is None:
            print("Voiceprint is not available. Check dependencies.")
            return

        print(f"\n=== 声纹注册: {name} ===")
        print(f"请对着麦克风说 2-3 句话（共 {num_samples} 次采样）\n")

        embeddings = []
        for i in range(num_samples):
            input(f"按 Enter 开始第 {i+1}/{num_samples} 次录音...")

            cap = UtteranceCapture(
                sample_rate=self.config.audio.asr_sample_rate,
                channels=self.config.audio.channels,
            )
            if not cap.is_available:
                print("无法打开麦克风，请检查音频设备。")
                return

            cap.start_capture()
            print(f"录音中...（说一句话，然后等待 {self.config.conversation.timeout_seconds} 秒）")

            timeout = self.config.conversation.timeout_seconds
            start = time.time()
            while time.time() - start < timeout:
                time.sleep(0.1)

            audio = cap.stop_capture()
            if audio is None:
                print("未捕获到音频，请重试。")
                continue

            duration = len(audio) / self.config.audio.asr_sample_rate
            print(f"录制完成，时长: {duration:.1f} 秒")

            emb = self.wespeaker_engine.extract_embedding(
                audio, self.config.audio.asr_sample_rate)
            embeddings.append(emb)
            print(f"  第 {i+1} 次采样完成。")

        if embeddings:
            self.speaker_manager.enroll(name, embeddings)
            print(f"\n✓ 说话人 '{name}' 注册成功！")
        else:
            print("\n✗ 注册失败：未收集到任何有效语音样本。")

    # =========================================================================
    # Main conversation loop
    # =========================================================================

    def start_conversation(self):
        """Main conversation loop with dual-mode and voiceprint."""
        print("\n" + "=" * 55)
        print("   Dual-Mode Voice Assistant (Cloud + Edge)")
        if self.speaker_manager and self.speaker_manager.speaker_count > 0:
            enrolled = ", ".join(self.speaker_manager.list_enrolled())
            print(f"   Enrolled speakers: {enrolled}")
        print("   Say '退出' or '再见' to exit.")
        print("=" * 55)

        # Connect to cloud TTS
        if not self.cloud_tts_client.connect():
            logger.warning(
                "Cannot connect to cloud TTS. Will use local-only mode.")
        else:
            logger.info("Cloud TTS connected.")

        # Check local Ollama
        if not self.local_ollama.health_check():
            logger.warning(
                "Local Ollama (%s) not available. "
                "Local fallback will use error message only.",
                self.local_ollama.model)
        else:
            logger.info("Local Ollama (%s) is ready.", self.local_ollama.model)

        try:
            with self.audio_stream_client as client:
                if not client.connect_to_asr_server():
                    logger.error("Cannot connect to ASR server. Exiting.")
                    return

                state = AssistantState.IDLE

                while True:
                    if state == AssistantState.IDLE:
                        print("\n" + "⚪" * 25)
                        print("Status: IDLE — waiting for wake word...")
                        print("⚪" * 25)

                        hotword_found = client.listen_for_hotword()
                        if hotword_found:
                            self._play_sound(
                                self.config.sounds.ding)
                            state = AssistantState.LISTENING_TO_USER
                        else:
                            logger.error("Wake word detection failed. Exiting.")
                            break

                    elif state == AssistantState.LISTENING_TO_USER:
                        print("\n" + "\U0001f7e2" * 25)
                        timeout = self.config.conversation.timeout_seconds
                        print(f"Status: LISTENING... (timeout={timeout:.0f}s)")
                        print("\U0001f7e2" * 25)

                        # Start parallel voiceprint capture
                        if self.utterance_capture and self.utterance_capture.is_available:
                            self.utterance_capture.start_capture()

                        final_transcript = ""
                        try:
                            for result_tuple in client.listen_and_stream():
                                if not result_tuple:
                                    continue
                                result, _ = result_tuple
                                print("\r" + " " * 80 + "\r", end="")

                                if "text" in result and result.get("text"):
                                    final_transcript = result["text"].strip().replace(
                                        " ", "")
                                    logger.info(
                                        "ASR final: '%s'", final_transcript)
                                    break
                                elif "partial" in result:
                                    partial = result.get('partial', '').strip()
                                    if partial:
                                        print(
                                            f"Partial: {partial}", end="", flush=True)

                        except (ConnectionResetError, BrokenPipeError) as e:
                            logger.error(
                                "ASR connection lost: %s. Reconnecting...", e)
                            client.resume_audio_capture()
                            if not client.connect_to_asr_server():
                                logger.error("ASR reconnect failed. Exiting.")
                                break
                            continue

                        # Stop voiceprint capture
                        voiceprint_audio = None
                        if self.utterance_capture and self.utterance_capture.is_available:
                            voiceprint_audio = self.utterance_capture.stop_capture()

                        if final_transcript:
                            # Run voiceprint identification asynchronously
                            if voiceprint_audio is not None:
                                self._identify_speaker_async(voiceprint_audio)

                            if final_transcript in self.config.conversation.exit_phrases:
                                logger.info("Exit phrase detected.")
                                self.last_transcript = "再见！"
                                state = AssistantState.ASSISTANT_SPEAKING
                            else:
                                self.last_transcript = final_transcript
                                state = AssistantState.ASSISTANT_SPEAKING
                        else:
                            print("\n" + "\U0001f534" * 25)
                            print("Conversation timeout. Returning to IDLE.")
                            print("\U0001f534" * 25)
                            self._play_sound(
                                self.config.sounds.dong, suspend_mic=False)
                            state = AssistantState.IDLE

                    elif state == AssistantState.ASSISTANT_SPEAKING:
                        print("\n" + "\U0001f535" * 25)
                        source_label = "CLOUD"
                        if self.llm_router.consecutive_cloud_failures > 0:
                            source_label = "LOCAL (fallback)"
                        print(f"Status: SPEAKING... [{source_label}]")
                        print("\U0001f535" * 25)

                        client.suspend_audio_capture()
                        try:
                            response_text, source = self.llm_router.get_response_and_play(
                                self.last_transcript)
                            if source == "local":
                                logger.info(
                                    "Response: %s", response_text[:80])
                        except Exception as e:
                            logger.error(
                                "LLM routing error: %s", e)
                        finally:
                            client.resume_audio_capture()

                        if self.last_transcript == "再见！":
                            break

                        state = AssistantState.LISTENING_TO_USER

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        except Exception as e:
            logger.critical("Fatal error in conversation loop: %s", e, exc_info=True)
        finally:
            self.close()
            print("\nProgram ended.")

    def close(self):
        """Clean up all resources."""
        logger.info("Shutting down...")
        self.cloud_tts_client.close()
        self.local_tts.close()
        if self.audio_stream_client:
            self.audio_stream_client.resume_audio_capture()
            self.audio_stream_client.close_asr_connection()
        self._pyaudio_player.terminate()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dual-Mode Voice Assistant with Voiceprint")
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: dual_mode/config.yaml)")
    parser.add_argument(
        "--enroll", metavar="NAME",
        help="Enroll a new speaker with the given name")
    parser.add_argument(
        "--list-speakers", action="store_true",
        help="List enrolled speakers and exit")
    args = parser.parse_args()

    # Resolve config path
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    orch = DualModeOrchestrator(config_path)

    if args.list_speakers:
        if orch.speaker_manager:
            enrolled = orch.speaker_manager.list_enrolled()
            if enrolled:
                print(f"Enrolled speakers ({len(enrolled)}):")
                for name in enrolled:
                    print(f"  - {name}")
            else:
                print("No speakers enrolled.")
        else:
            print("Voiceprint is not enabled.")
        return

    if args.enroll:
        orch.enroll_speaker(args.enroll)
        return

    orch.start_conversation()


if __name__ == "__main__":
    main()
