# -*- coding: utf-8 -*-
"""YAML config loader with attribute-style access and validation."""

import os
from typing import Any


class ConfigLoader:
    def __init__(self, config_path: str):
        try:
            import yaml
        except ImportError:
            raise ImportError("需要 PyYAML: pip install pyyaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)
        self._validate()

    def _validate(self):
        for key in ['servers', 'llm', 'audio', 'conversation']:
            if key not in self._data:
                raise KeyError(f"缺少配置节: '{key}'")
        cloud = self._data.get('servers', {}).get('cloud', {})
        if not cloud.get('asr', {}).get('host'):
            raise ValueError("servers.cloud.asr.host 未配置")

    def __getattr__(self, name: str) -> Any:
        if name == '_data':
            raise AttributeError
        if name in self._data:
            v = self._data[name]
            return _wrap(v) if isinstance(v, dict) else v
        raise AttributeError(f"配置中没有 '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class _wrap:
    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, k: str) -> Any:
        if k == '_d':
            raise AttributeError
        if k in self._d:
            v = self._d[k]
            return _wrap(v) if isinstance(v, dict) else v
        raise AttributeError(f"'{k}' 不在配置中")

    def get(self, k: str, d: Any = None) -> Any:
        v = self._d.get(k, d)
        return _wrap(v) if isinstance(v, dict) else v
