"""SpatialReasoner (ccvl/SpatialReasoner on Hugging Face)."""
from spatio.models._backend_spatial_reasoner import SpatialReasonerRunner as _SpatialReasonerRunner


class SpatialReasonerRunner:
    def __init__(self, device=None, **kwargs):
        kwargs.setdefault("use_flash_attn", True)
        self._r = _SpatialReasonerRunner(device=device or "cuda", **kwargs)

    def generate(self, image, prompt, max_new_tokens=1024, **kwargs):
        return self._r.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)
