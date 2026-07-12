"""Sa2VA model (ByteDance/Sa2VA-4B)."""
from spatio.models._backend_sa2va import Sa2VARunner as _Sa2VARunner


class Sa2VARunner:
    def __init__(self, device=None, **kwargs):
        self._r = _Sa2VARunner(device=device or "cuda", **kwargs)

    def generate(self, image, prompt, max_new_tokens=1024, **kwargs):
        return self._r.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)
