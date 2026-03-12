"""
SharedMemory: per-step cache for specialist agent outputs.

Cleared and rebuilt every step. The Final Reasoning Agent reads the full contents.
"""

from typing import Dict, List, Optional


class SharedMemory:
    """Per-step shared memory between specialist agents and the reasoning agent."""

    ROLE_STRATEGIES = {
        "direct_visual_heuristic": "Pictorial cues (occlusion, size, height in image). Strong for: count, general layout.",
        "explicit_3d_representation": "3D depth (z values, in front/behind). Strong for: closer/farther, depth order.",
        "scene_graph_construction": "2D spatial relations (above/below, left/right). Strong for: above/below, left/right, next to.",
    }

    def __init__(self):
        self._entries: List[Dict] = []

    def add(self, role: str, llm_name: str, answer: str, reason: str):
        self._entries.append({
            "role": role,
            "llm_name": llm_name,
            "answer": answer,
            "reason": reason,
        })

    def clear(self):
        self._entries = []

    def get_entries(self) -> List[Dict]:
        return list(self._entries)

    def to_prompt_text(self, role_weights: Optional[Dict] = None) -> str:
        """
        Format all entries as text for the Final Reasoning Agent prompt.
        role_weights: {(role, llm_name): w} — when provided, each agent line includes w.
        Format: Agent i  role={role}  w={w}  |  Answer: {answer}  |  Reason: {reason}
        """
        lines = []
        for i, e in enumerate(self._entries, 1):
            role, llm_name, answer, reason = e["role"], e["llm_name"], e["answer"], e["reason"]
            w = 0.5
            if role_weights is not None:
                w = role_weights.get((role, llm_name), 0.5)
            lines.append(f"  Agent {i}  role={role}  w={w:.3f}  |  Answer: {answer}  |  Reason: {reason}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)
