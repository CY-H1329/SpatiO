KAPPA = 0.5
MU = 0.3
GAMMA = 0.3
LAMBDA_F = 0.3
LAMBDA_G = 0.1
RAMP_TEMP = 5.0
BETA = 5.0

SPECIALIST_LLMS = [
    "llava4d",
    "qwen3_4b",
    "spatial_rgpt",
    "spatial_reasoner",
    "sa2va",
]

ROLES = [
    "direct_visual_heuristic",
    "explicit_3d_representation",
    "scene_graph_construction",
]

HEAD_AGENT_MODEL = "qwen3_4b"
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
