"""Qwen3-VL model (Hugging Face)."""
from models._backend_qwen3 import Qwen3Runner as _Qwen3Runner

MODEL_ID_MAP = {
    "qwen3_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-VL-8B-Instruct",
}


class Qwen3Runner:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device=None, **kwargs):
        hf_model_id = MODEL_ID_MAP.get(model_id, model_id)
        self._r = _Qwen3Runner(
            model_id=hf_model_id,
            device=device or "cuda",
            use_flash_attn=kwargs.get("use_flash_attn", True),
        )

    def generate(self, image, prompt, max_new_tokens=1024, **kwargs):
        return self._r.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)
