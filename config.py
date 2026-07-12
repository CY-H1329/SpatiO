import os
from typing import List, Set

# Paper hyperparameters (public)
KAPPA = 0.5
MU = 0.3
GAMMA = 0.3
LAMBDA_F = 0.3
LAMBDA_G = 0.1
RAMP_TEMP = 5.0
BETA = 5.0

# Full paper specialist pool (architecture-level; all 5)
SPECIALIST_LLMS: List[str] = [
    "llava4d",
    "sa2va",
    "qwen3_4b",
    "spatial_rgpt",
    "spatial_reasoner",
]
TOP_K_SPECIALISTS = int(os.environ.get("SPATIO_TOP_K", "5"))

# Specialist roles (ids only — prompt text withheld)
ROLES: List[str] = [
    "direct_visual_heuristic",
    "explicit_3d_representation",
    "scene_graph_construction",
]
ROLES_WITH_TOOLS: Set[str] = {"explicit_3d_representation", "scene_graph_construction"}
ROLE_TO_TOOL = {
    "direct_visual_heuristic": "none",
    "explicit_3d_representation": "3d_representation",
    "scene_graph_construction": "scene_graph",
}

HEAD_AGENT_MODEL = "qwen3_4b"
# Official Reasoning Agent code will be released later.
REASONING_AGENT_MODEL = "deepseek_r1"

ALL_CATEGORIES = [
    "spatial_relation",
    "distance_depth",
    "size",
    "orientation",
    "counting",
]

CATEGORY_DESCRIPTIONS = {
    "spatial_relation": "Positional relationship between objects.",
    "distance_depth": "Distance / depth relative to camera or reference.",
    "size": "Relative size / scale.",
    "orientation": "Facing / left-right / viewpoint orientation.",
    "counting": "Cardinality of objects in the scene.",
}

INITIAL_SCORE = 0.5

# Public HF ids (for documentation / future full release)
MODEL_CARD = {
    "llava4d": "llava-hf/llava-1.5-7b-hf",
    "sa2va": "ByteDance/Sa2VA-4B",
    "qwen3_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "spatial_rgpt": "a8cheng/SpatialRGPT-VILA1.5-8B",
    "spatial_reasoner": "ccvl/SpatialReasoner",
    "deepseek_r1": "(official Reasoning Agent — release later)",
}
