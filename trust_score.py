import copy
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from config import ROLES, KAPPA, MU, GAMMA, LAMBDA_F, LAMBDA_G, RAMP_TEMP, BETA


def select_agents_by_score(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    category: str,
    candidate_agents: List[str],
) -> Dict[str, str]:
    roles = list(ROLES)
    agents = list(candidate_agents)
    if len(agents) < len(roles):
        roles = roles[: len(agents)]
    if not roles:
        return {}

    assignment: Dict[str, str] = {}
    used_agents: set = set()

    for role in roles:
        best_agent = None
        best_score = -1e9
        for agent in agents:
            if agent in used_agents:
                continue
            s = scores.get(agent, {}).get(category, {}).get(role, 0.5)
            if s > best_score:
                best_score = s
                best_agent = agent
        if best_agent is not None:
            assignment[role] = best_agent
            used_agents.add(best_agent)
        elif agents:
            assignment[role] = agents[0]
    return assignment


def _normalize_for_comparison(s: str) -> str:
    if not s:
        return ""
    return " ".join((s or "").strip().lower().split())


def _extract_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    tail = text[-500:] if len(text) > 500 else text
    m = re.search(r"\(([^)]+)\)", tail)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final\s*answer|answer)[:\s]+([^\n,]+)", tail, re.I)
    if m:
        return m.group(1).strip()
    tokens = re.findall(r"[A-Za-z0-9]+", tail)
    return tokens[-1] if tokens else ""


def similarity_answer(
    pred: str, gt: str, normalize_fn: Optional[Callable[[str], str]] = None
) -> float:
    if normalize_fn is not None:
        p, g = normalize_fn(pred), normalize_fn(gt)
    else:
        gt_part = _extract_answer(gt) or gt
        g = _normalize_for_comparison(gt_part)
        if not g:
            return 0.5
        pred_part = _extract_answer(pred) or pred
        p = _normalize_for_comparison(pred_part)
    if not g:
        return 0.5
    return 1.0 if p == g else 0.0


def step1_compute_rewards(
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    kappa: float = KAPPA,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, float]:
    sim_fn = similarity_fn or similarity_answer
    final_correct = sim_fn(final_answer, gt_answer) >= 0.99
    sim_final = sim_fn(final_answer, gt_answer)

    rewards = {}
    for agent_id, answer in agent_answers.items():
        sim_i = sim_fn(answer, gt_answer)
        R_i = 2.0 * sim_i - 1.0
        if not final_correct:
            delta = max(0.0, sim_final - sim_i)
            R_i = R_i - kappa * delta
        rewards[agent_id] = R_i
    return rewards


def step2_phi_scale(N_c: int, T: float = RAMP_TEMP) -> float:
    return 1.0 if T <= 0 else 1.0 - math.exp(-N_c / T)


def step2_scale_rewards(
    rewards: Dict[str, float],
    N_c: int,
    T: float = RAMP_TEMP,
) -> Dict[str, float]:
    phi = step2_phi_scale(N_c, T)
    return {agent_id: phi * R_i for agent_id, R_i in rewards.items()}


def step3_update_scores_simple(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    scaled_rewards: Dict[str, float],
    category: str,
    agent_roles: Dict[str, str],
    gamma: float = GAMMA,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out = copy.deepcopy(scores)
    for agent_id, R_tilde in scaled_rewards.items():
        role = agent_roles.get(agent_id, ROLES[0])
        if agent_id not in out:
            out[agent_id] = {}
        if category not in out[agent_id]:
            out[agent_id][category] = {}
        if role not in out[agent_id][category]:
            out[agent_id][category][role] = 0.5
        s_old = out[agent_id][category][role]
        out[agent_id][category][role] = s_old + gamma * R_tilde
    return out


@dataclass
class TrustState:
    n_plus: float = 0.5
    n_minus: float = 0.5
    f: float = 0.5
    g: float = 0.5
    s: float = 0.5


def _reward_to_01(R_tilde: float) -> float:
    return max(0.0, min(1.0, (R_tilde + 1.0) / 2.0))


def step4_update_credibility_full(
    state: Dict[str, Dict[str, Dict[str, TrustState]]],
    scaled_rewards: Dict[str, float],
    category: str,
    agent_roles: Dict[str, str],
    lambda_f: float = LAMBDA_F,
    lambda_g: float = LAMBDA_G,
    mu: float = MU,
    gamma: float = GAMMA,
) -> Dict[str, Dict[str, Dict[str, TrustState]]]:
    out = copy.deepcopy(state)
    for agent_id, R_tilde in scaled_rewards.items():
        role = agent_roles.get(agent_id, ROLES[0])
        if agent_id not in out:
            out[agent_id] = {}
        if category not in out[agent_id]:
            out[agent_id][category] = {}
        if role not in out[agent_id][category]:
            out[agent_id][category][role] = TrustState()

        t = out[agent_id][category][role]
        r_tilde = _reward_to_01(R_tilde)

        t.n_plus += r_tilde
        t.n_minus += (1.0 - r_tilde)
        q = t.n_plus / (t.n_plus + t.n_minus) if (t.n_plus + t.n_minus) > 0 else 0.5

        t.f = (1.0 - lambda_f) * t.f + lambda_f * R_tilde
        t.g = (1.0 - lambda_g) * t.g + lambda_g * q
        s_tilde = mu * t.f + (1.0 - mu) * t.g
        t.s = s_tilde + gamma * R_tilde
    return out


def get_scores_from_state(
    state: Dict[str, Dict[str, Dict[str, TrustState]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        agent: {cat: {role: t.s for role, t in roles.items()} for cat, roles in cats.items()}
        for agent, cats in state.items()
    }


def compute_role_weights(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    category: str,
    role: str,
    agents: List[str],
    beta: float = BETA,
) -> Dict[str, float]:
    if not agents:
        return {}
    s_list = [
        scores.get(a, {}).get(category, {}).get(role, 0.5)
        for a in agents
    ]
    max_s = max(s_list)
    exp_s = [math.exp(beta * (s - max_s)) for s in s_list]
    total = sum(exp_s)
    if total <= 0:
        return {a: 1.0 / len(agents) for a in agents}
    return {a: exp_s[i] / total for i, a in enumerate(agents)}


def compute_weights_for_entries(
    entries: List[Dict],
    scores: Dict[str, Dict[str, Dict[str, float]]],
    category: str,
    beta: float = BETA,
) -> Dict[str, float]:
    role_to_agents: Dict[str, List[str]] = {}
    for e in entries:
        r, a = e["role"], e["llm_name"]
        if r not in role_to_agents:
            role_to_agents[r] = []
        if a not in role_to_agents[r]:
            role_to_agents[r].append(a)

    weights: Dict[str, float] = {}
    for role, agents in role_to_agents.items():
        w = compute_role_weights(scores, category, role, agents, beta=beta)
        for agent, val in w.items():
            weights[(role, agent)] = val
    return weights


def run_step4(
    state: Dict[str, Dict[str, Dict[str, TrustState]]],
    scores: Dict[str, Dict[str, Dict[str, float]]],
    agent_answers: Dict[str, str],
    final_answer: str,
    gt_answer: str,
    category: str,
    agent_roles: Dict[str, str],
    N_c: int,
    kappa: float = KAPPA,
    T: float = RAMP_TEMP,
    gamma: float = GAMMA,
    lambda_f: float = LAMBDA_F,
    lambda_g: float = LAMBDA_G,
    mu: float = MU,
) -> Dict[str, Dict[str, Dict[str, TrustState]]]:
    rewards = step1_compute_rewards(agent_answers, final_answer, gt_answer, kappa=kappa)
    scaled = step2_scale_rewards(rewards, N_c, T=T)
    updated_state = step4_update_credibility_full(
        state, scaled, category, agent_roles,
        lambda_f=lambda_f, lambda_g=lambda_g, mu=mu, gamma=gamma,
    )
    for agent, cats in updated_state.items():
        if agent not in scores:
            scores[agent] = {}
        for cat, roles in cats.items():
            if cat not in scores[agent]:
                scores[agent][cat] = {}
            for role, t in roles.items():
                scores[agent][cat][role] = t.s
    return updated_state
