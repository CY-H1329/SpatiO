"""LLaVA-1.5-7B proxy for the historical llava4d slot."""
from spatio.models._backend_llava import LLaVARunner


class LLaVA4DModel:
    def __init__(self, device=None):
        self._r = LLaVARunner(
            model_id="llava-hf/llava-1.5-7b-hf",
            device=device or "cuda",
        )

    def generate(self, image, prompt, max_new_tokens=1024, **kwargs):
        return self._r.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)
