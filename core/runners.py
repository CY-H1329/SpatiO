from typing import Dict, Optional

from PIL import Image

from .base import BaseRunner


class LLaVA4DRunner(BaseRunner):
    def __init__(self, model_id: str = "llava4d", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        try:
            from models.llava4d import LLaVA4DModel  # type: ignore
            self._model = LLaVA4DModel(device=self.device)
        except ImportError:
            self._model = None

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        if self._model is None:
            return ""
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


class Qwen3Runner(BaseRunner):
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        try:
            from models.qwen3 import Qwen3Runner as Q3R  # type: ignore
            self._model = Q3R(model_id=self.model_name, device=self.device)
        except ImportError:
            self._model = None

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        if self._model is None:
            return ""
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


class SpatialRGPTRunner(BaseRunner):
    def __init__(self, model_id: str = "spatial_rgpt", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        try:
            from models.spatial_rgpt import SpatialRGPTRunner as SRRunner  # type: ignore
            self._model = SRRunner(device=self.device)
        except ImportError:
            self._model = None

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        if self._model is None:
            return ""
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


class SpatialReasonerRunner(BaseRunner):
    def __init__(self, model_id: str = "spatial_reasoner", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        try:
            from models.spatial_reasoner import SpatialReasonerRunner as SRRunner  # type: ignore
            self._model = SRRunner(device=self.device)
        except ImportError:
            self._model = None

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        if self._model is None:
            return ""
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


class Sa2VARunner(BaseRunner):
    def __init__(self, model_id: str = "sa2va", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        try:
            from models.sa2va import Sa2VARunner as S2VRunner  # type: ignore
            self._model = S2VRunner(device=self.device)
        except ImportError:
            self._model = None

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        if self._model is None:
            return ""
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


class ReasoningRunner(BaseRunner):
    def __init__(self, model_id: str = "deepseek_r1", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        try:
            from models.reasoning import ReasoningModel  # type: ignore
            self._model = ReasoningModel(model_id=self.model_name, device=self.device)
        except ImportError:
            try:
                from models.qwen3 import Qwen3Runner  # type: ignore
                self._model = Qwen3Runner(
                    model_id="Qwen/Qwen3-VL-8B-Instruct" if "8b" in str(self.model_name).lower() else "Qwen/Qwen3-VL-4B-Instruct",
                    device=self.device,
                )
            except ImportError:
                self._model = None

    def generate(self, image: Optional[Image.Image], prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
        if self._model is None:
            return ""
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


RUNNER_MAP: Dict[str, type] = {
    "llava4d": LLaVA4DRunner,
    "qwen3_4b": Qwen3Runner,
    "spatial_rgpt": SpatialRGPTRunner,
    "spatial_reasoner": SpatialReasonerRunner,
    "sa2va": Sa2VARunner,
    "deepseek_r1": ReasoningRunner,
}


def get_runner(model_name: str, device: Optional[str] = None, **kwargs) -> BaseRunner:
    cls = RUNNER_MAP.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    return cls(model_id=model_name, device=device, **kwargs)
