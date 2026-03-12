"""
Core module: model runners and interfaces.

Implement or load your model runners from a separate repository.
Each runner must implement BaseRunner.generate(image, prompt, **kwargs) -> str.
"""

from .base import BaseRunner
from .runners import (
    LLaVA4DRunner,
    Qwen3Runner,
    SpatialRGPTRunner,
    SpatialReasonerRunner,
    Sa2VARunner,
    ReasoningRunner,
    get_runner,
)

__all__ = [
    "BaseRunner",
    "LLaVA4DRunner",
    "Qwen3Runner",
    "SpatialRGPTRunner",
    "SpatialReasonerRunner",
    "Sa2VARunner",
    "ReasoningRunner",
    "get_runner",
]
