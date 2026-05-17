# -*- coding: utf-8 -*-
"""Flask Web 服务 — 为前端聊天界面提供 API。"""

import base64
import io
import logging
import os
import sys
import tempfile
import threading
import json
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import time
import asyncio
import subprocess
import numpy as np

# 项目路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
        try:
            _local_asr = LocalASREngine(model_path=asr_path, sample_rate=sample_rate)
            logger.info("本地 ASR 就绪")
        except Exception as e:
            logger.error(f"本地 ASR 初始化失败: {e}")
            _local_asr = None
    else:
        logger.warning("本地 ASR 模型路径不存在: %s", asr_path)
        _local_asr = None

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
    content_type = audio_file.content_type
    audio_bytes = audio_file.read()
    sample_rate = _config.audio.sample_rate

    logger.info(f"收到音频文件: 文件名={audio_file.filename}, 类型={content_type}, 大小={len(audio_bytes)}字节")

    # ASR
    user_text = ""
    if _local_asr:
        try:
            # 将WebM格式转换为WAV格式
            audio_np = convert_webm_to_wav_array(audio_bytes, sample_rate)
            
            if audio_np is None or len(audio_np) == 0:
                logger.error("音频转换失败或结果为空")
                return jsonify({"error": "音频转换失败", "text": "", "audio": None})
            
            logger.info(f"转换后音频: 长度={len(audio_np)}, 采样率={sample_rate}, 值范围=[{audio_np.min()}, {audio_np.max()}]")
            
            # 重置ASR引擎
            _local_asr.reset()
            
            # 转 int16 PCM 送 Vosk
            raw = (audio_np * 32768.0).astype(np.int16).tobytes()
            _local_asr.accept_waveform(raw)
            result = _local_asr.final_result()
            
            if result and result.get("text"):
                user_text = result["text"].strip()
                logger.info(f"ASR原始结果: '{user_text}'")
                
                # 清理文本（移除空格、特殊字符等）
                user_text = clean_asr_text(user_text)
                logger.info(f"ASR清理后结果: '{user_text}'")
            else:
                logger.warning("ASR返回空结果")
                
        except Exception as e:
            logger.error(f"ASR处理错误: {e}", exc_info=True)
            user_text = ""
    else:
        logger.warning("ASR引擎未初始化")
        user_text = ""

    if not user_text or len(user_text) < 1:  # 至少1个字符
        logger.warning(f"ASR未能识别有效语音: '{user_text}'")
        return jsonify({"error": "未能识别有效语音内容", "text": "", "audio": None})

    # 声纹识别
    speaker_name = None
    if _speaker_manager and _speaker_engine:
        try:
            # 重新转换音频用于声纹识别
            audio_np = convert_webm_to_wav_array(audio_bytes, sample_rate)
            if audio_np is not None and len(audio_np) > 0:
                dur = len(audio_np) / sample_rate
                min_dur = _config.voiceprint.min_audio_duration_seconds
                if dur >= min_dur:
                    emb = _speaker_engine.extract_embedding(audio_np, sample_rate)
                    name, score = _speaker_manager.identify(emb)
                    if name and score > _config.voiceprint.identification_threshold:
                        speaker_name = name
                        logger.info("声纹识别: %s (置信度: %.3f)", name, score)
        except Exception as e:
            logger.debug("声纹识别跳过: %s", e)

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


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """流式聊天：支持文本和语音输入的流式响应。"""
    content_type = request.content_type
    
    # 处理语音输入
    if 'multipart/form-data' in content_type:
        if "audio" not in request.files:
            def generate_error():
                yield json.dumps({"type": "error", "content": "audio file is required"}).encode() + b'\n'
            return Response(generate_error(), mimetype='text/plain')

        audio_file = request.files["audio"]
        content_type = audio_file.content_type
        audio_bytes = audio_file.read()
        sample_rate = _config.audio.sample_rate

        logger.info(f"流式处理音频: 文件名={audio_file.filename}, 类型={content_type}, 大小={len(audio_bytes)}字节")

        # ASR
        user_text = ""
        if _local_asr:
            try:
                # 将WebM格式转换为WAV格式
                audio_np = convert_webm_to_wav_array(audio_bytes, sample_rate)
                
                if audio_np is None or len(audio_np) == 0:
                    logger.error("流式音频转换失败或结果为空")
                    def generate_error():
                        yield json.dumps({"type": "error", "content": "音频转换失败"}).encode() + b'\n'
                    return Response(generate_error(), mimetype='text/plain')
                
                logger.info(f"流式转换后音频: 长度={len(audio_np)}, 采样率={sample_rate}")
                
                # 重置ASR引擎
                _local_asr.reset()
                
                # 转 int16 PCM 送 Vosk
                raw = (audio_np * 32768.0).astype(np.int16).tobytes()
                _local_asr.accept_waveform(raw)
                result = _local_asr.final_result()

                if result and result.get("text"):
                    user_text = result["text"].strip()
                    user_text = clean_asr_text(user_text)
                    logger.info(f"流式ASR结果: '{user_text}'")
                else:
                    logger.warning("流式ASR返回空结果")
                    
            except Exception as e:
                logger.error(f"流式ASR处理错误: {e}", exc_info=True)
                user_text = ""
        else:
            logger.warning("流式ASR引擎未初始化")
            user_text = ""

        if not user_text or len(user_text) < 1:
            logger.warning(f"流式ASR未能识别有效语音: '{user_text}'")
            def generate_error():
                yield json.dumps({"type": "error", "content": "未能识别有效语音内容"}).encode() + b'\n'
            return Response(generate_error(), mimetype='text/plain')

        # 声纹识别
        speaker_name = None
        if _speaker_manager and _speaker_engine:
            try:
                audio_np = convert_webm_to_wav_array(audio_bytes, sample_rate)
                if audio_np is not None and len(audio_np) > 0:
                    dur = len(audio_np) / sample_rate
                    min_dur = _config.voiceprint.min_audio_duration_seconds
                    if dur >= min_dur:
                        emb = _speaker_engine.extract_embedding(audio_np, sample_rate)
                        name, score = _speaker_manager.identify(emb)
                        if name and score > _config.voiceprint.identification_threshold:
                            speaker_name = name
                            logger.info("流式声纹识别: %s (置信度: %.3f)", name, score)
            except Exception as e:
                logger.debug("流式声纹识别跳过: %s", e)

        # 构建 LLM 输入
        llm_input = f"{speaker_name}说：{user_text}" if speaker_name else user_text
        
        # 发送用户输入确认
        def generate_user_confirmation():
            yield json.dumps({
                "type": "user_message",
                "content": user_text,
                "speaker": speaker_name
            }).encode() + b'\n'
            
            # LLM 流式响应
            try:
                response = _local_ollama.generate_full(llm_input)
                if not response:
                    response = "抱歉，我暂时无法回答。"
                    
                # 流式发送文本（逐字）
                for i, char in enumerate(response):
                    yield json.dumps({
                        "type": "text_chunk",
                        "content": char,
                        "is_final": i == len(response) - 1
                    }).encode() + b'\n'
                    time.sleep(0.03)  # 控制文本流速

                # TTS 完整音频（末尾发送）
                audio_b64 = _tts_to_base64(response)
                if audio_b64:
                    yield json.dumps({
                        "type": "audio",
                        "content": audio_b64,
                        "mime": "audio/mp3"
                    }).encode() + b'\n'
            except Exception as e:
                logger.error("LLM 错误: %s", e)
                yield json.dumps({
                    "type": "error",
                    "content": f"服务暂时不可用: {str(e)}"
                }).encode() + b'\n'
                
        return Response(generate_user_confirmation(), mimetype='text/plain')
    
    # 处理文本输入
    else:
        data = request.get_json(force=True)
        user_text = data.get("text", "").strip()
        if not user_text:
            def generate_error():
                yield json.dumps({"type": "error", "content": "text is required"}).encode() + b'\n'
            return Response(generate_error(), mimetype='text/plain')

        def generate_response():
            # LLM 流式响应
            try:
                response = _local_ollama.generate_full(user_text)
                if not response:
                    response = "抱歉，我暂时无法回答。"
                    
                # 发送文本流
                for i, char in enumerate(response):
                    yield json.dumps({
                        "type": "text_chunk",
                        "content": char,
                        "is_final": i == len(response) - 1
                    }).encode() + b'\n'
                    time.sleep(0.03)  # 控制文本流速
                # TTS 完整音频（末尾发送）
                audio_b64 = _tts_to_base64(response)
                if audio_b64:
                    yield json.dumps({
                        "type": "audio",
                        "content": audio_b64,
                        "mime": "audio/mp3"
                    }).encode() + b'\n'

            except Exception as e:
                logger.error("LLM 流式错误: %s", e)
                yield json.dumps({
                    "type": "error",
                    "content": f"服务暂时不可用: {str(e)}"
                }).encode() + b'\n'

                
        return Response(generate_response(), mimetype='text/plain')


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

def read_wav_to_array(wav_bytes, target_sr=16000):
    """直接读取 WAV 字节为 float32 numpy 数组，无需 ffmpeg。"""
    import io, wave
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if sw == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            return None

        if nch > 1:
            audio = audio.reshape(-1, nch).mean(axis=1)

        if sr != target_sr:
            logger.warning("采样率不匹配: %d vs %d", sr, target_sr)

        return audio
    except Exception as e:
        logger.error("WAV 读取失败: %s", e)
        return None

# 兼容旧函数名
convert_webm_to_wav_array = read_wav_to_array


def clean_asr_text(text):
    """清理ASR识别结果"""
    if not text:
        return text
    
    # 移除空格、制表符、换行符
    text = text.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
    
    # 移除常见无意义字符
    text = text.strip("。，、；：？！…—-()[]{}【】<>《》\"'`~@#$%^&*_+=\\|/0123456789")
    
    # 移除特殊Unicode字符
    text = ''.join(char for char in text if ord(char) < 65536 and char.isprintable())
    
    # 移除连续重复字符（可能是ASR错误）
    cleaned = ""
    prev_char = ""
    repeat_count = 0
    
    for char in text:
        if char == prev_char:
            repeat_count += 1
            if repeat_count <= 3:  # 允许最多3个连续重复字符
                cleaned += char
        else:
            cleaned += char
            repeat_count = 1
            prev_char = char
    
    return cleaned.strip()


def _tts_to_base64(text: str) -> str:
    """用 edge-tts 合成语音并返回 base64 MP3。网络不可用时返回空。"""
    try:
        import edge_tts
        chunks = []

        async def _run():
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])

        asyncio.run(_run())
        if chunks:
            return base64.b64encode(b"".join(chunks)).decode("utf-8")
    except Exception as e:
        logger.warning("TTS 合成失败 (可能网络不通): %s", e)
    return ""


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Web Chat Server")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--port", type=int, default=8080, help="端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    config_path = args.config or os.path.join(
        _PROJECT_ROOT, "dual_mode", "config.yaml")
    init_services(config_path)

    print(f"\n  Web 聊天服务已启动: http://localhost:{args.port}")
    print(f"  调试模式: {'启用' if args.debug else '禁用'}")
    print(f"  ASR引擎: {'已初始化' if _local_asr else '未初始化'}")
    print(f"\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)