import os
from typing import Set

# Paper hyperparameters
KAPPA = 0.5
MU = 0.3
GAMMA = 0.3
LAMBDA_F = 0.3
LAMBDA_G = 0.1
RAMP_TEMP = 5.0
BETA = 5.0

# paper (default) = all 5 specialists; minimal = Qwen-only smoke
_PROFILE = (os.environ.get("SPATIO_PROFILE", "paper") or "paper").strip().lower()
_SPECIALIST_LLMS_DEFAULT = [
    "llava4d",
    "sa2va",
    "qwen3_4b",
    "spatial_rgpt",
    "spatial_reasoner",
]
_SPECIALIST_LLMS_ENV = os.environ.get("SPATIO_SPECIALIST_LLMS", "").strip()
if _SPECIALIST_LLMS_ENV:
    SPECIALIST_LLMS = [s.strip() for s in _SPECIALIST_LLMS_ENV.split(",") if s.strip()]
elif _PROFILE in ("minimal", "qwen", "qwen_only"):
    SPECIALIST_LLMS = ["qwen3_4b", "qwen3_4b", "qwen3_4b"]
else:
    SPECIALIST_LLMS = list(_SPECIALIST_LLMS_DEFAULT)

TOP_K_SPECIALISTS = int(os.environ.get("SPATIO_TOP_K", str(len(SPECIALIST_LLMS))))

_ROLE_SET_ID = os.environ.get("SPATIO_ROLE_SET", "").strip() or "human_v0"
try:
    from spatio.roles.registry import load_role_set  # type: ignore

    _rs = load_role_set(_ROLE_SET_ID)
    ROLES = _rs.role_ids
    ROLES_WITH_TOOLS: Set[str] = _rs.roles_with_tools
    ROLE_TO_TOOL = _rs.role_to_tool
except Exception:
    ROLES = [
        "direct_visual_heuristic",
        "explicit_3d_representation",
        "scene_graph_construction",
    ]
    ROLES_WITH_TOOLS = {"explicit_3d_representation", "scene_graph_construction"}
    ROLE_TO_TOOL = {
        "direct_visual_heuristic": "none",
        "explicit_3d_representation": "3d_representation",
        "scene_graph_construction": "scene_graph",
    }

HEAD_AGENT_MODEL = "qwen3_4b"
# Official Reasoning Agent later; models/reasoning.py is an interim stand-in.
REASONING_AGENT_MODEL = "deepseek_r1"

ALL_CATEGORIES = [
    "spatial_relation",
    "distance_depth",
    "size",
    "orientation",
    "counting",
]

CATEGORY_DESCRIPTIONS = {
    "spatial_relation": "Positional relationship between objects: above/below, next to, between.",
    "distance_depth": "How far apart objects are or how far from the camera/viewer.",
    "size": "Comparing the size, height, or scale of objects.",
    "orientation": "Which direction objects face, left/right/front/behind relative to viewpoint.",
    "counting": "Counting how many objects exist in the scene.",
}

INITIAL_SCORE = 0.5
