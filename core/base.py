"""
Abstract base class for model runners.

Implement generate(image, prompt, **kwargs) -> str.
Models are loaded from a separate core/repository to allow
running specialists on different GPU cores.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from PIL import Image


class BaseRunner(ABC):
    """Base interface for specialist and head/reasoning model runners."""

    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.device = device
        self._model = None

    @abstractmethod
    def load(self) -> None:
        """Load the model. Call once before generate()."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """
        Generate text given image (optional) and prompt.

        Args:
            image: PIL Image or None for text-only models.
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        pass

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
