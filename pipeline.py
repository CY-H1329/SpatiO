import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random

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

from config import ROLES_WITH_TOOLS, ROLE_TO_TOOL
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
    parallel_specialists: bool = False,
    final_aggregator: str = "reasoner",
    use_delta_penalty: bool = True,
    no_short_term_ema: bool = False,
    no_long_term_ema: bool = False,
    top_k: int = 5,
    beta: float = BETA,
    role_assignment: str = "default",
    fixed_role_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Execute one step of the SpatiO pipeline.

    specialist_generate(llm_name, image, prompt) -> str
    reasoning_generate(prompt, image=None) -> str

    When use_beta_weights=True, role weights w = exp(beta*s)/sum(exp(beta*s')) are
    injected into the final reasoning prompt.

    parallel_specialists: lance les 3 rôles spécialistes en parallèle (ThreadPoolExecutor).
    Les appels partageant le même ``llm_name`` doivent être sérialisés côté appelant (locks).
    """
    t0 = time.time()
    timing: Dict[str, Any] = {
        "head_agent_sec": 0.0,
        "specialists_sec": [],
        "reasoning_agent_sec": 0.0,
        "parallel_specialists": bool(parallel_specialists),
    }
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
    role_assignment = (role_assignment or "default").strip().lower()
    from config import SPECIALIST_LLMS as _active_specialists

    # Candidate pool for this step (Top-k).
    pool_all = list(_active_specialists)
    k = int(top_k or 0)
    if k <= 0:
        k = 1
    if k > len(pool_all):
        k = len(pool_all)

    if trust_state is not None and step > 0:
        scores_dict = get_scores_from_state(trust_state)
        # score per agent = max role score in this category (simple, stable)
        per_agent: List[Tuple[str, float]] = []
        for a in pool_all:
            role_scores = [scores_dict.get(a, {}).get(category, {}).get(r, 0.5) for r in ROLES]
            per_agent.append((a, max(role_scores) if role_scores else 0.5))
        per_agent.sort(key=lambda x: x[1], reverse=True)
        candidate_agents = [a for a, _ in per_agent[:k]]
    else:
        # Step 0 (pas de trust_state fiable): sample déterministe via RNG du score_map
        rng = getattr(score_map, "rng", random.Random(42))
        candidate_agents = list(pool_all)
        rng.shuffle(candidate_agents)
        candidate_agents = candidate_agents[:k]

    if not candidate_agents:
        candidate_agents = list(pool_all)

    if role_assignment in ("fixed", "fixed_role", "always_same"):
        # Always same mapping role -> agent (ignore trust scores).
        role_map = dict(fixed_role_map or {})
        if not role_map:
            # default fixed: first roles map to first agents in candidate pool
            for idx, r in enumerate(ROLES):
                if idx < len(candidate_agents):
                    role_map[r] = candidate_agents[idx]
        assignments = [(r, role_map[r]) for r in ROLES if r in role_map]
    elif role_assignment in ("random", "random_role"):
        rng = random.Random(step + 1337)
        agents = list(candidate_agents)
        rng.shuffle(agents)
        roles = list(ROLES)
        if len(agents) < len(roles):
            roles = roles[: len(agents)]
        assignments = [(roles[i], agents[i]) for i in range(len(roles))]
    else:
        # Default = trust score based (or score_map for step 0), but restricted to candidate_agents.
        if trust_state is not None and step > 0:
            scores = get_scores_from_state(trust_state)
            role_to_agent = select_agents_by_score(scores, category, candidate_agents)
            assignments = [(r, a) for r, a in role_to_agent.items()]
        else:
            # score_map selection restricted to candidate pool
            assignments = score_map.select_agents(category, step)
            assignments = [(r, a) for r, a in assignments if a in candidate_agents] or assignments[: len(candidate_agents)]

    # 3. Run specialists -> SharedMemory
    shared_memory = SharedMemory()
    agent_details = []
    tool_output_cache: Dict[str, str] = {}

    for role, _llm in assignments:
        if role in ROLES_WITH_TOOLS and role not in tool_output_cache:
            try:
                from tools import get_3d_representation, get_scene_graph  # type: ignore
                tool_kind = (ROLE_TO_TOOL or {}).get(role, "none")
                if tool_kind == "3d_representation":
                    tool_output_cache[role] = get_3d_representation(image)
                elif tool_kind == "scene_graph":
                    tool_output_cache[role] = get_scene_graph(image)
            except ImportError:
                tool_output_cache[role] = ""

    def _run_one_specialist(pair: Tuple[str, str]) -> Tuple[str, str, str, float]:
        role, llm_name = pair
        _t_spec = time.time()
        tool_output: Optional[str] = None
        if role in ROLES_WITH_TOOLS:
            tool_output = tool_output_cache.get(role) or None
        role_prompt = build_role_prompt(role, query, tool_output=tool_output, answer_type=answer_type)
        raw_output = specialist_generate(llm_name, image, role_prompt)
        dt = round(time.time() - _t_spec, 3)
        return role, llm_name, raw_output, dt

    use_parallel = bool(parallel_specialists) and len(assignments) >= 2
    if use_parallel:
        _t_wall = time.time()
        n_workers = min(3, len(assignments))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            spec_results = list(pool.map(_run_one_specialist, list(assignments)))
        timing["specialists_parallel_wall_sec"] = round(time.time() - _t_wall, 3)
        for role, llm_name, raw_output, dt in spec_results:
            timing["specialists_sec"].append({"role": role, "llm": llm_name, "sec": dt})
            answer, reason = parse_specialist_output(raw_output, answer_type=answer_type)
            shared_memory.add(role, llm_name, answer, reason)
            agent_details.append({"role": role, "llm_name": llm_name, "answer": answer, "reason": reason})
    else:
        for role, llm_name in assignments:
            _t_spec = time.time()
            tool_output: Optional[str] = None
            if role in ROLES_WITH_TOOLS:
                tool_output = tool_output_cache.get(role) or None

            role_prompt = build_role_prompt(role, query, tool_output=tool_output, answer_type=answer_type)
            raw_output = specialist_generate(llm_name, image, role_prompt)
            timing["specialists_sec"].append({"role": role, "llm": llm_name, "sec": round(time.time() - _t_spec, 3)})
            answer, reason = parse_specialist_output(raw_output, answer_type=answer_type)
            shared_memory.add(role, llm_name, answer, reason)
            agent_details.append({"role": role, "llm_name": llm_name, "answer": answer, "reason": reason})

    if not use_parallel:
        timing["specialists_parallel_wall_sec"] = round(
            sum(float(x["sec"]) for x in timing["specialists_sec"]), 3
        )

    # 4. Compute role weights (beta) for final reasoning
    role_weights = None
    if use_beta_weights:
        if trust_state is not None:
            scores = get_scores_from_state(trust_state)
        else:
            scores = score_map.to_scores_dict()
        role_weights = compute_weights_for_entries(
            shared_memory.get_entries(), scores, category, beta=float(beta)
        )

    # 5. Final aggregation: either LLM reasoner or voting baselines
    final_aggregator = (final_aggregator or "reasoner").strip().lower()
    reasoning_raw = ""

    def _majority_vote(answers: List[str]) -> str:
        answers = [a for a in answers if a]
        if not answers:
            return ""
        counts: Dict[str, int] = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        return max(counts, key=counts.get)

    def _weighted_vote(entries: List[Dict[str, Any]]) -> str:
        # weight = TTO-derived role weight if present, else 1.0
        tally: Dict[str, float] = {}
        for e in entries:
            ans = (e.get("answer") or "").strip()
            if not ans:
                continue
            w = 1.0
            if role_weights is not None:
                w = float(role_weights.get((e.get("role"), e.get("llm_name")), 1.0))
            tally[ans] = tally.get(ans, 0.0) + w
        return max(tally, key=tally.get) if tally else ""

    if final_aggregator in ("majority", "majority_vote"):
        final_answer = _majority_vote([e["answer"] for e in shared_memory.get_entries()])
        reasoning_raw = f"[VOTE majority] {final_answer}"
        timing["reasoning_agent_sec"] = 0.0
    elif final_aggregator in ("weighted", "confidence_weighted", "weighted_vote", "confidence"):
        final_answer = _weighted_vote(shared_memory.get_entries())
        reasoning_raw = f"[VOTE weighted] {final_answer}"
        timing["reasoning_agent_sec"] = 0.0
    else:
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
            lambda_f=0.0 if no_short_term_ema else LAMBDA_F,
            lambda_g=0.0 if no_long_term_ema else LAMBDA_G,
            mu=MU,
            use_delta_penalty=use_delta_penalty,
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
        "final_aggregator": final_aggregator,
        "ablation": {
            "use_delta_penalty": bool(use_delta_penalty),
            "no_short_term_ema": bool(no_short_term_ema),
            "no_long_term_ema": bool(no_long_term_ema),
            "top_k": int(top_k),
            "beta": float(beta),
            "role_assignment": role_assignment,
        },
        "elapsed_sec": round(elapsed, 2),
        "timing": timing,
    }
