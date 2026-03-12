"""
ScoreMap: per-category score maps (LLM x Role).

Each category has a |ROLES| x |LLMS| matrix of scores.
Step 0 -> random agent selection per role.
Step > 0 -> argmax selection per role (select_agents_by_score).
"""

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import SPECIALIST_LLMS, ROLES, INITIAL_SCORE


class ScoreMap:
    """Category-specific LLM x Role score maps."""

    def __init__(
        self,
        categories: List[str],
        llms: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        initial_score: float = INITIAL_SCORE,
        seed: int = 42,
    ):
        self.categories = list(categories)
        self.llms = list(llms or SPECIALIST_LLMS)
        self.roles = list(roles or ROLES)
        self.initial_score = initial_score
        self.rng = random.Random(seed)

        self._maps: Dict[str, Dict[str, Dict[str, float]]] = {}
        for cat in self.categories:
            self._maps[cat] = {
                role: {llm: initial_score for llm in self.llms}
                for role in self.roles
            }

    def select_agents(self, category: str, step: int) -> List[Tuple[str, str]]:
        """Return [(role, llm), ...] for each role. step==0 -> random; step>0 -> argmax."""
        if category not in self._maps:
            category = self.categories[0]

        assignments: List[Tuple[str, str]] = []
        for role in self.roles:
            if step == 0:
                llm = self.rng.choice(self.llms)
            else:
                scores = self._maps[category][role]
                llm = max(scores, key=scores.get)
            assignments.append((role, llm))
        return assignments

    def get_score(self, category: str, role: str, llm: str) -> float:
        return self._maps.get(category, {}).get(role, {}).get(llm, self.initial_score)

    def set_score(self, category: str, role: str, llm: str, value: float):
        if category in self._maps and role in self._maps[category]:
            self._maps[category][role][llm] = value

    def get_category_map(self, category: str) -> Optional[Dict[str, Dict[str, float]]]:
        return self._maps.get(category)

    def to_scores_dict(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Convert to scores[llm][category][role] for trust_score module."""
        out = {}
        for llm in self.llms:
            out[llm] = {}
            for cat in self.categories:
                out[llm][cat] = {}
                for role in self.roles:
                    out[llm][cat][role] = self.get_score(cat, role, llm)
        return out

    def from_scores_dict(self, scores: Dict[str, Dict[str, Dict[str, float]]]):
        """Update maps from scores[llm][category][role]."""
        for llm, cats in scores.items():
            for cat, roles in cats.items():
                for role, val in roles.items():
                    self.set_score(cat, role, llm, val)

    def save(self, path: str):
        data = {
            "categories": self.categories,
            "llms": self.llms,
            "roles": self.roles,
            "initial_score": self.initial_score,
            "maps": self._maps,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "ScoreMap":
        data = json.loads(Path(path).read_text())
        obj = cls(
            categories=data["categories"],
            llms=data["llms"],
            roles=data["roles"],
            initial_score=data["initial_score"],
        )
        obj._maps = data["maps"]
        return obj
