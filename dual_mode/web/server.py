# -*- coding: utf-8 -*-
"""Flask Web 服务 — 为前端聊天界面提供 API。"""

import base64
import io
import logging
import os
import sys
import tempfile
import threading

# 项目路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from dual_mode.utils.config_loader import ConfigLoader
from dual_mode.llm.local_ollama_client import LocalOllamaClient
from dual_mode.tts.local_tts_client import StreamingLocalTTS
from dual_mode.asr.local_asr import LocalASREngine
from dual_mode.voiceprint.speechbrain_engine import SpeechBrainEngine
from dual_mode.voiceprint.speaker_manager import SpeakerManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ── 全局状态 ──────────────────────────────────────────
_config = None
_local_ollama: LocalOllamaClient = None
_local_tts: StreamingLocalTTS = None
_local_asr: LocalASREngine = None
_speaker_engine: SpeechBrainEngine = None
_speaker_manager: SpeakerManager = None
_mode = "local"  # cloud | local

# 服务端口
WEB_PORT = 8080


def init_services(config_path: str):
    global _config, _local_ollama, _local_tts, _local_asr
    global _speaker_engine, _speaker_manager

    _config = ConfigLoader(config_path)
    sample_rate = _config.audio.sample_rate
    tts_rate = _config.audio.tts_sample_rate

    # 本地 LLM
    prompt = ""
    pp = os.path.join(_PROJECT_ROOT, "dual_mode/prompts/local_fallback.txt")
    if os.path.exists(pp):
        with open(pp, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

    _local_ollama = LocalOllamaClient(
        base_url=_config.servers.local.ollama.base_url,
        model=_config.llm.local.model,
        system_prompt=prompt,
        temperature=_config.llm.local.temperature,
        num_predict=_config.llm.local.num_predict,
        timeout=_config.llm.local.timeout_seconds,
    )

    _local_tts = StreamingLocalTTS(sample_rate=tts_rate)

    # 本地 ASR
    asr_path = _config.asr.local.model_path
    if os.path.isdir(asr_path):
        _local_asr = LocalASREngine(model_path=asr_path, sample_rate=sample_rate)
        logger.info("本地 ASR 就绪")
    else:
        logger.warning("本地 ASR 模型路径不存在: %s", asr_path)

    # 声纹
    if _config.voiceprint.enabled:
        try:
            _speaker_engine = SpeechBrainEngine(model_id=_config.voiceprint.model)
            _speaker_manager = SpeakerManager(
                enrollment_dir=_config.voiceprint.enrollment_dir,
                embedding_dim=_config.voiceprint.embedding_dim,
                threshold=_config.voiceprint.identification_threshold,
            )
            logger.info("声纹就绪 (%d 人)", _speaker_manager.speaker_count)
        except Exception as e:
            logger.warning("声纹初始化失败: %s", e)

    logger.info("Web 服务初始化完成，模式: %s", _mode)


# ═══════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════

@app.route("/api/chat/text", methods=["POST"])
def chat_text():
    """文本聊天：text → LLM → TTS → text+audio。"""
    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({"error": "text is required"}), 400

    # LLM
    try:
        response = _local_ollama.generate_full(user_text)
        if not response:
            response = "抱歉，我暂时无法回答。"
    except Exception as e:
        logger.error("LLM 错误: %s", e)
        response = "服务暂时不可用。"

    # TTS → 音频
    audio_b64 = _tts_to_base64(response)

    return jsonify({
        "text": response,
        "audio": audio_b64,
        "audio_mime": "audio/mp3",
    })


@app.route("/api/chat/voice", methods=["POST"])
def chat_voice():
    """语音聊天：audio → ASR → LLM → TTS → text+audio。"""
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    sample_rate = _config.audio.sample_rate

    # ASR
    user_text = ""
    if _local_asr:
        import numpy as np
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        _local_asr.reset()
        chunks = _split_audio(audio_np, sample_rate, 0.3)
        for chunk in chunks:
            chunk_bytes = (chunk * 32768.0).astype(np.int16).tobytes()
            r = _local_asr.accept_waveform(chunk_bytes)
            if r and r.get("text"):
                user_text = r["text"]
                break
        if not user_text:
            final = _local_asr.final_result()
            if final and final.get("text"):
                user_text = final["text"]
        user_text = user_text.strip().replace(" ", "")

    if not user_text:
        return jsonify({"error": "未能识别语音", "text": "", "audio": None})

    logger.info("ASR: %s", user_text)

    # 声纹识别
    speaker_name = None
    if _speaker_manager and _speaker_engine:
        try:
            import numpy as np
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            dur = len(audio_np) / sample_rate
            min_dur = _config.voiceprint.min_audio_duration_seconds
            if dur >= min_dur:
                emb = _speaker_engine.extract_embedding(audio_np, sample_rate)
                name, score = _speaker_manager.identify(emb)
                if name:
                    speaker_name = name
                    logger.info("声纹: %s (%.3f)", name, score)
        except Exception as e:
            logger.debug("声纹跳过: %s", e)

    # 构建 LLM 输入
    llm_input = f"{speaker_name}说：{user_text}" if speaker_name else user_text

    # LLM
    try:
        response = _local_ollama.generate_full(llm_input)
        if not response:
            response = "抱歉，我暂时无法回答。"
    except Exception as e:
        logger.error("LLM 错误: %s", e)
        response = "服务暂时不可用。"

    # TTS
    audio_b64 = _tts_to_base64(response)

    return jsonify({
        "user_text": user_text,
        "speaker": speaker_name,
        "text": response,
        "audio": audio_b64,
        "audio_mime": "audio/mp3",
    })


@app.route("/api/speakers", methods=["GET"])
def list_speakers():
    if _speaker_manager:
        return jsonify({"speakers": _speaker_manager.list_enrolled()})
    return jsonify({"speakers": []})


@app.route("/api/speakers/enroll", methods=["POST"])
def enroll_speaker():
    if not _speaker_manager or not _speaker_engine:
        return jsonify({"error": "声纹未启用"}), 400

    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    if "audio_files" not in request.files:
        # 返回提示文本
        sentences = [
            "你好，我是来注册声纹的，今天天气真不错。",
            "语音识别可以帮助我们更好地与机器交互。",
            "床前明月光，疑是地上霜，举头望明月，低头思故乡。",
        ]
        return jsonify({"need_audio": True, "sentences": sentences, "count": 3})

    # 处理上传的音频
    audio_files = request.files.getlist("audio_files")
    if len(audio_files) < 2:
        return jsonify({"error": "至少需要2段音频"}), 400

    sr = _config.audio.sample_rate
    embeddings = []
    for af in audio_files:
        raw = af.read()
        import numpy as np
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        dur = len(audio) / sr
        if dur < 1.0:
            continue
        emb = _speaker_engine.extract_embedding(audio, sr)
        embeddings.append(emb)

    if embeddings:
        _speaker_manager.enroll(name, embeddings)
        return jsonify({"ok": True, "name": name, "samples": len(embeddings)})
    return jsonify({"error": "音频长度不足"}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "mode": _mode,
        "asr": _local_asr is not None,
        "llm": _local_ollama is not None,
        "tts": _local_tts is not None,
        "voiceprint": _speaker_manager is not None,
        "speakers": _speaker_manager.list_enrolled() if _speaker_manager else [],
    })


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ═══════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════

def _tts_to_base64(text: str) -> str:
    """用 edge-tts 合成语音并返回 base64 MP3。"""
    import asyncio
    try:
        import edge_tts
        loop = asyncio.new_event_loop()
        chunks = []

        async def _run():
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])

        loop.run_until_complete(_run())
        loop.close()
        if chunks:
            return base64.b64encode(b"".join(chunks)).decode("utf-8")
    except Exception as e:
        logger.error("TTS 合成失败: %s", e)
    return ""


def _split_audio(audio, sr: int, chunk_s: float = 0.3):
    """将音频切分为固定长度块。"""
    chunk_size = int(sr * chunk_s)
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i + chunk_size]


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Web Chat Server")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--port", type=int, default=8080, help="端口")
    args = parser.parse_args()

    config_path = args.config or os.path.join(
        _PROJECT_ROOT, "dual_mode", "config.yaml")
    init_services(config_path)

    print(f"\n  Web 聊天服务已启动: http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
