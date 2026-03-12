"""
SpatiO configuration.

Hyperparameters (paper defaults):
  k (kappa) = 0.5, mu = 0.3, gamma = 0.3,
  lambda_f = 0.3, lambda_g = 0.1, T = 5, beta = 5
"""

# ---------------------------------------------------------------------------
# Hyperparameters (paper Section 4)
# ---------------------------------------------------------------------------
KAPPA = 0.5          # k: penalty when final answer diverges from GT
MU = 0.3             # mu: balance short-term (f) vs long-term (g) EMA
GAMMA = 0.3          # gamma: direct reward injection into final score
LAMBDA_F = 0.3       # lambda_f: short-term EMA decay
LAMBDA_G = 0.1       # lambda_g: long-term EMA decay
RAMP_TEMP = 5.0      # T: phi(N_c) = 1 - exp(-N_c/T)
BETA = 5.0           # beta: weight sharpness w = exp(beta*s) / sum(exp(beta*s'))

# ---------------------------------------------------------------------------
# Specialist models (5 agents)
# ---------------------------------------------------------------------------
SPECIALIST_LLMS = [
    "llava4d",
    "qwen3_4b",
    "spatial_rgpt",
    "spatial_reasoner",
    "sa2va",
]

# ---------------------------------------------------------------------------
# Roles (3 strategies)
# ---------------------------------------------------------------------------
ROLES = [
    "direct_visual_heuristic",
    "explicit_3d_representation",
    "scene_graph_construction",
]

# ---------------------------------------------------------------------------
# Fixed models
# ---------------------------------------------------------------------------
HEAD_AGENT_MODEL = "qwen3_4b"
REASONING_AGENT_MODEL = "deepseek_r1"

# ---------------------------------------------------------------------------
# Categories (unified taxonomy)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Score map defaults
# ---------------------------------------------------------------------------
INITIAL_SCORE = 0.5
