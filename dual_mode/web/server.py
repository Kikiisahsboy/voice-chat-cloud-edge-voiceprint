# -*- coding: utf-8 -*-
"""Flask Web 服务 — 为前端聊天界面提供 API，支持 SSE 流式输出。"""

import base64
import io
import json
import logging
import os
import sys
import threading
import wave

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
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

_config = None
_local_ollama: LocalOllamaClient = None
_local_tts: StreamingLocalTTS = None
_local_asr: LocalASREngine = None
_speaker_engine: SpeechBrainEngine = None
_speaker_manager: SpeakerManager = None
_mode = "local"


def init_services(config_path: str):
    global _config, _local_ollama, _local_tts, _local_asr
    global _speaker_engine, _speaker_manager

    _config = ConfigLoader(config_path)
    sr = _config.audio.sample_rate

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

    _local_tts = StreamingLocalTTS(sample_rate=_config.audio.tts_sample_rate)

    asr_path = _config.asr.local.model_path
    if os.path.isdir(asr_path):
        _local_asr = LocalASREngine(model_path=asr_path, sample_rate=sr)
        logger.info("本地 ASR 就绪")

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

    logger.info("Web 服务初始化完成")


# ═══════════════════════════════════════════════════════
#  SSE 流式聊天
# ═══════════════════════════════════════════════════════

@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """SSE 流式聊天：text → 逐句推送 LLM 回复 → 最后推 TTS 音频。"""
    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    speaker = data.get("speaker", "").strip()
    if not user_text:
        return jsonify({"error": "text is required"}), 400

    llm_input = f"{speaker}说：{user_text}" if speaker else user_text

    def generate():
        full_response = ""
        try:
            for sentence in _local_ollama.generate_stream(llm_input):
                if sentence:
                    full_response += sentence
                    yield f"data: {json.dumps({'type': 'text', 'data': sentence})}\n\n"
        except Exception as e:
            logger.error("LLM 流式错误: %s", e)
            yield f"data: {json.dumps({'type': 'text', 'data': '抱歉，服务暂时不可用。'})}\n\n"
            full_response = "抱歉，服务暂时不可用。"

        # TTS
        if full_response.strip():
            try:
                audio_b64 = _tts_sync(full_response)
                if audio_b64:
                    yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64, 'mime': 'audio/mp3'})}\n\n"
            except Exception as e:
                logger.error("TTS 错误: %s", e)

        yield "data: {\"type\": \"done\"}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })


# ═══════════════════════════════════════════════════════
#  语音聊天（接收 WAV/PCM，返回流式 SSE）
# ═══════════════════════════════════════════════════════

@app.route("/api/chat/voice/stream", methods=["POST"])
def chat_voice_stream():
    """语音聊天 SSE 流式。"""
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_file = request.files["audio"]
    raw = audio_file.read()
    sr = _config.audio.sample_rate

    # 解析 WAV 文件
    audio_np, actual_sr = _wav_to_pcm(raw)
    if audio_np is None:
        return jsonify({"error": "无法解析音频文件"}), 400

    # 重采样如果采样率不同
    if actual_sr != sr and actual_sr > 0:
        try:
            import scipy.signal
            gcd = __import__('math').gcd(actual_sr, sr)
            audio_np = scipy.signal.resample_poly(audio_np.astype('float64'), sr // gcd, actual_sr // gcd).astype('float32')
        except Exception:
            pass

    # ASR
    user_text = ""
    if _local_asr:
        _local_asr.reset()
        chunks = _split_audio(audio_np, sr, 0.4)
        for chunk in chunks:
            chunk_bytes = (chunk * 32767.0).astype('<i2').tobytes()
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
        return jsonify({"error": "未能识别语音", "text": ""}), 200

    logger.info("ASR: %s", user_text)

    # 声纹
    speaker = ""
    if _speaker_manager and _speaker_engine:
        try:
            dur = len(audio_np) / sr
            min_dur = _config.voiceprint.min_audio_duration_seconds
            if dur >= min_dur:
                emb = _speaker_engine.extract_embedding(audio_np, sr)
                name, score = _speaker_manager.identify(emb)
                if name:
                    speaker = name
        except Exception as e:
            logger.debug("声纹跳过: %s", e)

    # 构建 LLM 输入
    llm_input = f"{speaker}说：{user_text}" if speaker else user_text

    def generate():
        # 先发送 ASR 结果和声纹
        yield f"data: {json.dumps({'type': 'user_text', 'data': user_text, 'speaker': speaker})}\n\n"

        full_response = ""
        try:
            for sentence in _local_ollama.generate_stream(llm_input):
                if sentence:
                    full_response += sentence
                    yield f"data: {json.dumps({'type': 'text', 'data': sentence})}\n\n"
        except Exception as e:
            logger.error("LLM 流式错误: %s", e)
            yield f"data: {json.dumps({'type': 'text', 'data': '抱歉，服务暂时不可用。'})}\n\n"
            full_response = "抱歉，服务暂时不可用。"

        if full_response.strip():
            try:
                audio_b64 = _tts_sync(full_response)
                if audio_b64:
                    yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64, 'mime': 'audio/mp3'})}\n\n"
            except Exception as e:
                logger.error("TTS 错误: %s", e)

        yield "data: {\"type\": \"done\"}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════
#  说话人 API
# ═══════════════════════════════════════════════════════

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
        sentences = [
            "你好，我是来注册声纹的，今天天气真不错。",
            "语音识别可以帮助我们更好地与机器交互。",
            "床前明月光，疑是地上霜，举头望明月，低头思故乡。",
        ]
        return jsonify({"need_audio": True, "sentences": sentences, "count": 3})

    audio_files = request.files.getlist("audio_files")
    if len(audio_files) < 2:
        return jsonify({"error": "至少需要2段音频"}), 400

    sr = _config.audio.sample_rate
    embeddings = []
    for af in audio_files:
        raw = af.read()
        audio_np, actual_sr = _wav_to_pcm(raw)
        if audio_np is None:
            continue
        if actual_sr != sr and actual_sr > 0:
            try:
                import scipy.signal
                from math import gcd
                audio_np = scipy.signal.resample_poly(
                    audio_np.astype('float64'), sr // gcd(actual_sr, sr),
                    actual_sr // gcd(actual_sr, sr)).astype('float32')
            except Exception:
                pass
        dur = len(audio_np) / sr
        if dur < 1.0:
            continue
        emb = _speaker_engine.extract_embedding(audio_np, sr)
        embeddings.append(emb)

    if embeddings:
        _speaker_manager.enroll(name, embeddings)
        return jsonify({"ok": True, "name": name, "samples": len(embeddings)})
    return jsonify({"error": "音频长度不足"}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True, "mode": _mode,
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
#  辅助
# ═══════════════════════════════════════════════════════

def _tts_sync(text: str) -> str:
    """同步 TTS 合成，返回 base64 MP3。"""
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
        logger.error("TTS 失败: %s", e)
    return ""


def _wav_to_pcm(data: bytes):
    """将 WAV 字节解码为 float32 PCM (+ sample_rate)。"""
    try:
        with io.BytesIO(data) as f:
            with wave.open(f, 'rb') as wf:
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
                import numpy as np
                w = wf.getsampwidth()
                if w == 2:
                    pcm = np.frombuffer(raw, dtype='<i2')
                elif w == 4:
                    pcm = np.frombuffer(raw, dtype='<i4')
                else:
                    return None, 0
                return pcm.astype(np.float32) / 32767.0, sr
    except Exception as e:
        logger.error("WAV 解析失败: %s", e)
        return None, 0


def _split_audio(audio, sr: int, chunk_s: float = 0.3):
    chunk_size = int(sr * chunk_s)
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i + chunk_size]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Web Chat Server")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    config_path = args.config or os.path.join(_PROJECT_ROOT, "dual_mode", "config.yaml")
    init_services(config_path)

    print(f"\n  Web 聊天: http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
