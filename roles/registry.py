import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    name: str
    tool: str  # "none" | "3d_representation" | "scene_graph"
    prompt_path: Path


@dataclass(frozen=True)
class RoleSet:
    set_id: str
    roles: List[RoleSpec]

    @property
    def role_ids(self) -> List[str]:
        return [r.role_id for r in self.roles]

    @property
    def roles_with_tools(self) -> Set[str]:
        out: Set[str] = set()
        for r in self.roles:
            if (r.tool or "none") != "none":
                out.add(r.role_id)
        return out

    @property
    def role_to_tool(self) -> Dict[str, str]:
        return {r.role_id: (r.tool or "none") for r in self.roles}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _role_sets_root() -> Path:
    return _repo_root() / "roles"


def load_role_set(set_id: Optional[str] = None) -> RoleSet:
    """
    Charge un role set depuis roles/<set_id>/roles.json.
    Défault: SPATIO_ROLE_SET ou "human_v0".
    """
    sid = (set_id or os.environ.get("SPATIO_ROLE_SET") or "human_v0").strip()
    root = _role_sets_root() / sid
    spec_path = root / "roles.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Role set introuvable: {sid} (attendu: {spec_path})")
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    roles: List[RoleSpec] = []
    for r in data.get("roles", []):
        rid = str(r["id"])
        tool = str(r.get("tool") or "none")
        prompt_file = root / "prompts" / f"{rid}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt introuvable pour rôle {rid}: {prompt_file}")
        roles.append(
            RoleSpec(
                role_id=rid,
                name=str(r.get("name") or rid),
                tool=tool,
                prompt_path=prompt_file,
            )
        )
    if not roles:
        raise ValueError(f"Role set vide: {sid}")
    return RoleSet(set_id=sid, roles=roles)


def read_role_prompt(role_id: str, set_id: Optional[str] = None) -> str:
    rs = load_role_set(set_id=set_id)
    for r in rs.roles:
        if r.role_id == role_id:
            return r.prompt_path.read_text(encoding="utf-8")
    raise KeyError(f"Rôle inconnu {role_id!r} dans role set {rs.set_id!r}")

