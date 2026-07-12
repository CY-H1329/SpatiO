"""SpatialRGPT — requires SPATIALRGPT_PATH (official repo clone)."""
from models._backend_spatial_rgpt import SpatialRGPTRunner as _SpatialRGPTRunner


class SpatialRGPTRunner:
    def __init__(self, device=None, **kwargs):
        self._r = _SpatialRGPTRunner(device=device or "cuda", **kwargs)

    def generate(self, image, prompt, max_new_tokens=1024, **kwargs):
        return self._r.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)
