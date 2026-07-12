"""Final reasoning agent slot.

IMPORTANT — public release status
---------------------------------
The paper's official Reasoning Agent will be released in a **later update**.
What ships today is only a **temporary stand-in** (Qwen3-VL) so that the full
5-specialist pipeline can still run end-to-end.

Do not cite numbers from this stand-in as the paper's DeepSeek-R1 reasoner.
Override weights with SPATIO_REASONING_MODEL_ID if needed, or use
``--final_aggregator majority|weighted`` to skip the VLM reasoner.
"""
import logging
import os
import warnings

from models._backend_qwen3 import Qwen3Runner

logger = logging.getLogger(__name__)

_RELEASE_NOTICE = (
    "SpatiO: official Reasoning Agent is not in this release yet "
    "(coming later). Using temporary Qwen3-VL stand-in."
)


class ReasoningModel:
    """Interim reasoner stand-in until the official agent is published."""

    def __init__(self, model_id="deepseek_r1", device=None, **kwargs):
        warnings.warn(_RELEASE_NOTICE, UserWarning, stacklevel=2)
        logger.warning(_RELEASE_NOTICE)

        override = os.environ.get("SPATIO_REASONING_MODEL_ID", "").strip()
        if override:
            qwen_id = override
        elif "8b" in str(model_id).lower() or str(model_id).lower() in ("deepseek_r1", "reasoner"):
            qwen_id = "Qwen/Qwen3-VL-8B-Instruct"
        else:
            qwen_id = "Qwen/Qwen3-VL-4B-Instruct"
        self._r = Qwen3Runner(model_id=qwen_id, device=device or "cuda")

    def generate(self, image, prompt, max_new_tokens=1024, **kwargs):
        return self._r.generate(image, prompt, max_new_tokens=max_new_tokens, **kwargs)
