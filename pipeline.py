import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from config import (
    ALL_CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    ROLES,
    BETA,
    KAPPA,
    RAMP_TEMP,
    GAMMA,
    LAMBDA_F,
    LAMBDA_G,
    MU,
)
from prompts import build_head_agent_prompt, build_role_prompt, build_final_reasoning_prompt
from score_map import ScoreMap
from shared_memory import SharedMemory
from trust_score import (
    compute_weights_for_entries,
    run_step4,
    get_scores_from_state,
    select_agents_by_score,
    TrustState,
)

_ROLES_WITH_TOOLS = {"explicit_3d_representation", "scene_graph_construction"}
logger = logging.getLogger(__name__)


def _infer_answer_type(query: str) -> str:
    if not query or not query.strip():
        return "free_form"
    q = query.strip().upper()
    if "OPTIONS:" in q and ("(A)" in q or "(B)" in q):
        return "multiple_choice"
    return "free_form"


def parse_category(raw: str, valid_categories: List[str]) -> str:
    raw_clean = (raw or "").strip().lower()
    for cat in valid_categories:
        if cat.lower() == raw_clean:
            return cat
    for cat in valid_categories:
        if cat.lower() in raw_clean:
            return cat
    return valid_categories[0]


def parse_specialist_output(raw: str, answer_type: str = "multiple_choice") -> Tuple[str, str]:
    raw = (raw or "").strip()
    answer = ""
    reason = ""

    if answer_type == "free_form":
        m = re.search(r"Answer\s*:\s*(.+?)(?=\n|Reason\s*:|\Z)", raw, re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip()[:100]
        if not answer:
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            answer = lines[0][:100] if lines else ""
    else:
        ans_m = re.search(r"Answer\s*:\s*\(?([A-F])\)?", raw, re.IGNORECASE)
        if ans_m:
            answer = f"({ans_m.group(1).upper()})"
        if not answer:
            fallback = re.search(r"\(([A-F])\)", raw)
            if fallback:
                answer = f"({fallback.group(1).upper()})"

    reason_m = re.search(r"Reason\s*:\s*(.+?)(?=\nAnswer\s*:|\Z)", raw, re.IGNORECASE | re.DOTALL)
    if reason_m:
        reason = reason_m.group(1).strip()[:2000]

    return answer, reason


def parse_final_answer(raw: str, answer_type: str = "multiple_choice") -> str:
    raw = (raw or "").strip()
    if answer_type == "free_form":
        m = re.search(r"Answer\s*:\s*(.+?)(?=\n|Reason\s*:|\Z)", raw, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()[:100]
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[0][:100] if lines else ""
    m = re.search(r"Answer\s*:\s*\(?([A-F])\)?", raw, re.IGNORECASE)
    if m:
        return f"({m.group(1).upper()})"
    m = re.search(r"\(([A-F])\)", raw)
    if m:
        return f"({m.group(1)})"
    return raw[:20]


def _is_correct(pred: str, gt: str, answer_type: str) -> bool:
    from trust_score import similarity_answer
    return similarity_answer(pred, gt) >= 0.99


def run_step(
    image: Image.Image,
    query: str,
    gt: Optional[str],
    step: int,
    score_map: ScoreMap,
    trust_state: Optional[Dict[str, Dict[str, Dict[str, TrustState]]]],
    head_generate: Callable[[Image.Image, str], str],
    specialist_generate: Callable[[str, Image.Image, str], str],
    reasoning_generate: Callable,
    N_c_per_category: Optional[Dict[str, int]] = None,
    update_trust: bool = True,
    force_category: Optional[str] = None,
    use_vlm_reasoning: bool = False,
    answer_type: Optional[str] = None,
    use_beta_weights: bool = True,
) -> Dict:
    t0 = time.time()
    timing = {"head_agent_sec": 0.0, "specialists_sec": [], "reasoning_agent_sec": 0.0}
    if answer_type is None:
        answer_type = _infer_answer_type(query)

    # 1. Head Agent -> category
    if force_category and force_category in ALL_CATEGORIES:
        category = force_category
    else:
        _t = time.time()
        head_prompt = build_head_agent_prompt(query, ALL_CATEGORIES, CATEGORY_DESCRIPTIONS)
        head_raw = head_generate(image, head_prompt)
        category = parse_category(head_raw, ALL_CATEGORIES)
        timing["head_agent_sec"] = round(time.time() - _t, 3)

    # 2. Agent selection
    if trust_state is not None and step > 0:
        scores = get_scores_from_state(trust_state)
        from config import SPECIALIST_LLMS
        role_to_agent = select_agents_by_score(scores, category, SPECIALIST_LLMS)
        assignments = [(r, a) for r, a in role_to_agent.items()]
    else:
        assignments = score_map.select_agents(category, step)

    # 3. Run specialists -> SharedMemory
    shared_memory = SharedMemory()
    agent_details = []
    tool_output_cache = {}

    for role, llm_name in assignments:
        _t_spec = time.time()
        tool_output = None
        if role in _ROLES_WITH_TOOLS and role not in tool_output_cache:
            try:
                from tools import get_3d_representation, get_scene_graph  # type: ignore
                if role == "explicit_3d_representation":
                    tool_output_cache[role] = get_3d_representation(image)
                elif role == "scene_graph_construction":
                    tool_output_cache[role] = get_scene_graph(image)
            except ImportError:
                tool_output_cache[role] = ""
            tool_output = tool_output_cache.get(role, "")

        role_prompt = build_role_prompt(role, query, tool_output=tool_output, answer_type=answer_type)
        raw_output = specialist_generate(llm_name, image, role_prompt)
        timing["specialists_sec"].append({"role": role, "llm": llm_name, "sec": round(time.time() - _t_spec, 3)})
        answer, reason = parse_specialist_output(raw_output, answer_type=answer_type)
        shared_memory.add(role, llm_name, answer, reason)
        agent_details.append({"role": role, "llm_name": llm_name, "answer": answer, "reason": reason})

    # 4. Compute role weights (beta) for final reasoning
    role_weights = None
    if use_beta_weights:
        if trust_state is not None:
            scores = get_scores_from_state(trust_state)
        else:
            scores = score_map.to_scores_dict()
        role_weights = compute_weights_for_entries(
            shared_memory.get_entries(), scores, category, beta=BETA
        )

    # 5. Final Reasoning Agent
    _t = time.time()
    shared_text = shared_memory.to_prompt_text(role_weights=role_weights)
    reasoning_prompt = build_final_reasoning_prompt(
        query=query,
        shared_memory_text=shared_text,
        category=category,
        answer_type=answer_type,
        with_image=use_vlm_reasoning,
    )
    reasoning_raw = reasoning_generate(reasoning_prompt, image=image)
    timing["reasoning_agent_sec"] = round(time.time() - _t, 3)
    final_answer = parse_final_answer(reasoning_raw, answer_type=answer_type)

    # 6. Update trust state (train phase)
    if update_trust and trust_state is not None and gt:
        agent_answers = {e["llm_name"]: e["answer"] for e in shared_memory.get_entries()}
        agent_roles = {e["llm_name"]: e["role"] for e in shared_memory.get_entries()}
        N_c = (N_c_per_category or {}).get(category, step + 1)
        scores = {
            llm: {cat: {r: 0.5 for r in ROLES} for cat in score_map.categories}
            for llm in score_map.llms
        }
        run_step4(
            trust_state,
            scores,
            agent_answers,
            final_answer,
            gt,
            category,
            agent_roles,
            N_c,
            kappa=KAPPA,
            T=RAMP_TEMP,
            gamma=GAMMA,
            lambda_f=LAMBDA_F,
            lambda_g=LAMBDA_G,
            mu=MU,
        )
        score_map.from_scores_dict(get_scores_from_state(trust_state))

    elapsed = time.time() - t0
    correct = _is_correct(final_answer, gt, answer_type) if gt else None
    timing["total_sec"] = round(elapsed, 3)

    return {
        "step": step,
        "category": category,
        "assignments": assignments,
        "agent_details": agent_details,
        "final_answer": final_answer,
        "gt": gt,
        "correct": correct,
        "reasoning_raw": reasoning_raw[:3000],
        "elapsed_sec": round(elapsed, 2),
        "timing": timing,
    }
