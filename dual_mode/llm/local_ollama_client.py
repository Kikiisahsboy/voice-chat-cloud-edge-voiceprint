# -*- coding: utf-8 -*-
"""
Local Ollama client for edge-side LLM inference.

Simplified from server/server/ollama_client.py:
- No model pre-warming (small models load fast)
- Supports both streaming and non-streaming generation
- Maintains conversation history for multi-turn dialogue
"""

import json
import logging
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)


class LocalOllamaClient:
    """Lightweight Ollama client for edge device LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2:1.5b",
        system_prompt: str = "",
        temperature: float = 0.7,
        num_predict: int = 512,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.chat_url = f"{self.base_url}/api/chat"
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

        self.messages = []
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt,
            })

    def health_check(self) -> bool:
        """Check if local Ollama is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model (or a prefix match) exists
                for name in model_names:
                    if self.model in name or name in self.model:
                        return True
                logger.warning(
                    "Ollama reachable but model '%s' not found in %s",
                    self.model, model_names)
                return False
            return False
        except Exception as e:
            logger.warning("Local Ollama health check failed: %s", e)
            return False

    def generate_stream(self, user_text: str) -> Generator[str, None, None]:
        """Streaming generation, yields sentence chunks."""
        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "stream": True,
        }

        full_reply = ""
        sentence_buffer = ""
        delimiters = "，。！？,.!?\n"

        try:
            with requests.post(
                self.chat_url, json=payload, stream=True, timeout=self.timeout
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                    except json.JSONDecodeError:
                        continue

                    if 'content' in chunk.get('message', {}):
                        delta = chunk['message']['content']
                        full_reply += delta
                        sentence_buffer += delta

                        while True:
                            split_pos = -1
                            for d in delimiters:
                                pos = sentence_buffer.find(d)
                                if pos != -1 and (split_pos == -1 or pos < split_pos):
                                    split_pos = pos
                            if split_pos != -1:
                                yield sentence_buffer[:split_pos + 1].strip()
                                sentence_buffer = sentence_buffer[split_pos + 1:]
                            else:
                                break

                    if chunk.get('done'):
                        if sentence_buffer.strip():
                            yield sentence_buffer.strip()
                        self.messages.append({
                            "role": "assistant",
                            "content": full_reply,
                        })
                        return

        except requests.exceptions.RequestException as e:
            logger.error("Local Ollama request error: %s", e)
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            raise

    def generate_full(self, user_text: str) -> str:
        """Non-streaming generation, returns complete response."""
        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "stream": False,
        }

        try:
            resp = requests.post(
                self.chat_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            reply = data.get("message", {}).get("content", "")
            self.messages.append({"role": "assistant", "content": reply})
            return reply.strip()
        except requests.exceptions.RequestException as e:
            logger.error("Local Ollama request error: %s", e)
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            raise

    def reset_session(self):
        """Clear conversation history, keeping system prompt if present."""
        if self.messages and self.messages[0]['role'] == 'system':
            self.messages = [self.messages[0]]
        else:
            self.messages = []
