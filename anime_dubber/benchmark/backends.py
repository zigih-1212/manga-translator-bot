"""Backend registry for TTS benchmark.
Selects and instantiates TTS backends by name.
"""
from __future__ import annotations
from typing import Type

from src.tts.base import TTSBackend
from src.tts.cosyvoice3 import CosyVoice3Backend
from src.tts.omnivoice import OmnivoiceBackend
from src.tts.qwen3 import Qwen3TTSBackend
from src.tts.f5tts import F5TTSBackend

_BACKENDS: dict[str, Type[TTSBackend]] = {
    "cosyvoice3": CosyVoice3Backend,
    "omnivoice": OmnivoiceBackend,
    "qwen3": Qwen3TTSBackend,
    "f5tts": F5TTSBackend,
}

def available() -> list[str]:
    return list(_BACKENDS.keys())

def get_backend(name: str) -> TTSBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown TTS backend: {name}. Available: {available()}"
        )
    return _BACKENDS[name]()
