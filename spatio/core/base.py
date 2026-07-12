from abc import ABC, abstractmethod
from typing import Any, Optional

from PIL import Image


class BaseRunner(ABC):
    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.device = device
        self._model = None

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        pass

    def unload(self) -> None:
        self._model = None
