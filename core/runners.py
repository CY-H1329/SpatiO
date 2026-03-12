"""
Specialist model runners.

Each runner loads its model from a separate core/repository.
Use device='cuda:0', 'cuda:1', etc. to run on different GPU cores.

Implement load() and generate() by importing from your model codebase.
"""

from typing import Dict, Optional

from PIL import Image

from .base import BaseRunner


# ---------------------------------------------------------------------------
# LLaVA-4D
# ---------------------------------------------------------------------------
class LLaVA4DRunner(BaseRunner):
    """LLaVA-4D specialist. Load from your model core (e.g. llava4d model)."""

    def __init__(self, model_id: str = "llava4d", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        """Load LLaVA-4D from your model repository."""
        try:
            from models.llava4d import LLaVA4DModel  # type: ignore
            self._model = LLaVA4DModel(device=self.device)
        except ImportError:
            # Fallback: placeholder for supplement — replace with your implementation
            self._model = None

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self._model is None:
            return "[LLaVA4D not loaded — implement load() with your model]"
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Qwen3-VL-4B
# ---------------------------------------------------------------------------
class Qwen3Runner(BaseRunner):
    """Qwen3-VL-4B specialist. Load from your model core."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        """Load Qwen3-VL from your model repository."""
        try:
            from models.qwen3 import Qwen3Runner as Q3R  # type: ignore
            self._model = Q3R(model_id=self.model_name, device=self.device)
        except ImportError:
            self._model = None

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self._model is None:
            return "[Qwen3-VL not loaded — implement load() with your model]"
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


# ---------------------------------------------------------------------------
# SpatialRGPT
# ---------------------------------------------------------------------------
class SpatialRGPTRunner(BaseRunner):
    """SpatialRGPT specialist. Load from your model core."""

    def __init__(self, model_id: str = "spatial_rgpt", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        """Load SpatialRGPT from your model repository."""
        try:
            from models.spatial_rgpt import SpatialRGPTRunner as SRRunner  # type: ignore
            self._model = SRRunner(device=self.device)
        except ImportError:
            self._model = None

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self._model is None:
            return "[SpatialRGPT not loaded — implement load() with your model]"
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


# ---------------------------------------------------------------------------
# SpatialReasoner
# ---------------------------------------------------------------------------
class SpatialReasonerRunner(BaseRunner):
    """SpatialReasoner specialist. Load from your model core."""

    def __init__(self, model_id: str = "spatial_reasoner", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        """Load SpatialReasoner from your model repository."""
        try:
            from models.spatial_reasoner import SpatialReasonerRunner as SRRunner  # type: ignore
            self._model = SRRunner(device=self.device)
        except ImportError:
            self._model = None

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self._model is None:
            return "[SpatialReasoner not loaded — implement load() with your model]"
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Sa2VA
# ---------------------------------------------------------------------------
class Sa2VARunner(BaseRunner):
    """Sa2VA specialist. Load from your model core."""

    def __init__(self, model_id: str = "sa2va", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        """Load Sa2VA from your model repository."""
        try:
            from models.sa2va import Sa2VARunner as S2VRunner  # type: ignore
            self._model = S2VRunner(device=self.device)
        except ImportError:
            self._model = None

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self._model is None:
            return "[Sa2VA not loaded — implement load() with your model]"
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Factory: get runner by model name
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Reasoning Agent (text-only or VLM)
# ---------------------------------------------------------------------------
class ReasoningRunner(BaseRunner):
    """Final reasoning agent. Load DeepSeek-R1, Qwen3-VL-8B, or your model."""

    def __init__(self, model_id: str = "deepseek_r1", device: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_id, device=device, **kwargs)

    def load(self) -> None:
        """Load reasoning model from your repository."""
        try:
            from models.reasoning import ReasoningModel  # type: ignore
            self._model = ReasoningModel(model_id=self.model_name, device=self.device)
        except ImportError:
            # Fallback: use Qwen3-VL for reasoning
            try:
                from models.qwen3 import Qwen3Runner  # type: ignore
                self._model = Qwen3Runner(
                    model_id="Qwen/Qwen3-VL-8B-Instruct" if "8b" in str(self.model_name).lower() else "Qwen/Qwen3-VL-4B-Instruct",
                    device=self.device,
                )
            except ImportError:
                self._model = None

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self._model is None:
            return "[Reasoning model not loaded — implement load() with your model]"
        return self._model.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)


RUNNER_MAP: Dict[str, type] = {
    "llava4d": LLaVA4DRunner,
    "qwen3_4b": Qwen3Runner,
    "spatial_rgpt": SpatialRGPTRunner,
    "spatial_reasoner": SpatialReasonerRunner,
    "sa2va": Sa2VARunner,
    "deepseek_r1": ReasoningRunner,
}


def get_runner(
    model_name: str,
    device: Optional[str] = None,
    **kwargs,
) -> BaseRunner:
    """Return a runner instance for the given model name."""
    cls = RUNNER_MAP.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(RUNNER_MAP.keys())}")
    return cls(model_id=model_name, device=device, **kwargs)
