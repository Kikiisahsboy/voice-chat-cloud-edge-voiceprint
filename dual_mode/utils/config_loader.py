# -*- coding: utf-8 -*-
import os
from typing import Any


class ConfigLoader:
    """Loads and provides attribute-style access to YAML configuration."""

    def __init__(self, config_path: str):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required. Install with: pip install pyyaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)

        self._validate()

    def _validate(self):
        required_top = ['servers', 'llm', 'voiceprint',
                        'wake_word', 'audio', 'conversation']
        for key in required_top:
            if key not in self._data:
                raise KeyError(f"Missing required config section: '{key}'")

        # Validate cloud server config
        cloud = self._data.get('servers', {}).get('cloud', {})
        if not cloud.get('asr', {}).get('host'):
            raise ValueError("servers.cloud.asr.host is required")
        if not cloud.get('tts', {}).get('host'):
            raise ValueError("servers.cloud.tts.host is required")

    def __getattr__(self, name: str) -> Any:
        if name == '_data':
            raise AttributeError("'_data' is not set")
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _DictWrapper(value)
            return value
        raise AttributeError(f"Config has no key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self):
        return f"ConfigLoader({list(self._data.keys())})"


class _DictWrapper:
    """Recursive attribute-style access for nested dicts."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name == '_data':
            raise AttributeError("'_data' is not set")
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _DictWrapper(value)
            return value
        raise AttributeError(f"'{name}' not found in config section")

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return _DictWrapper(value)
        return value

    def items(self):
        return self._data.items()

    def __repr__(self):
        return f"Config({list(self._data.keys())})"
