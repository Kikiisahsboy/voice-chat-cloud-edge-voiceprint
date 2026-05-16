# -*- coding: utf-8 -*-
"""本地 Ollama 客户端，用于边端 LLM 推理。"""

import json
import logging
from typing import Generator, Optional

logger = logging.getLogger(__name__)

try:
    import requests
    _requests_ok = True
except ImportError:
    requests = None
    _requests_ok = False


class LocalOllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2:1.5b",
        system_prompt: str = "",
        temperature: float = 0.7,
        num_predict: int = 512,
        timeout: float = 60.0,
    ):
        if not _requests_ok:
            raise ImportError("需要 requests: pip install requests")

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def health_check(self) -> bool:
        """检查本地 Ollama 是否可用。"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                names = [m.get("name", "") for m in models]
                for name in names:
                    if self.model in name or name in self.model:
                        return True
                logger.warning("Ollama 在线但模型 '%s' 未找到", self.model)
                return False
            return False
        except Exception as e:
            logger.warning("Ollama 检查失败: %s", e)
            return False

    def generate_full(self, user_text: str) -> str:
        """非流式生成，返回完整回复。"""
        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            reply = resp.json().get("message", {}).get("content", "")
            self.messages.append({"role": "assistant", "content": reply})
            return reply.strip()
        except Exception as e:
            logger.error("Ollama 请求失败: %s", e)
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            raise

    def generate_stream(self, user_text: str) -> Generator[str, None, None]:
        """流式生成，逐句 yield 回复内容。"""
        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            "stream": True,
        }

        full = ""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, stream=True)
            resp.raise_for_status()
            buf = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        buf += token
                        full += token
                        # 遇到句子分隔符就 yield
                        for sep in ("。", "！", "？", "，", "；", "\n", ".", "!", "?"):
                            idx = buf.find(sep)
                            if idx >= 0:
                                sentence = buf[:idx + len(sep)]
                                buf = buf[idx + len(sep):]
                                yield sentence.strip()
                                break
                except (json.JSONDecodeError, KeyError):
                    continue
            if buf.strip():
                yield buf.strip()
        except Exception as e:
            logger.error("Ollama 流式请求失败: %s", e)
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            raise

        self.messages.append({"role": "assistant", "content": full})

    def reset_session(self):
        if self.messages and self.messages[0]['role'] == 'system':
            self.messages = [self.messages[0]]
        else:
            self.messages = []
