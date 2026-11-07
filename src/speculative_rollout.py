"""Test-Time Compute: Speculative Rollout & Best-of-N Verification for ZAC 2026.
Executes candidate trajectory rollouts on independent in-memory `zaic_tools.ToolEnv` clones.
Scores trajectories by zero-cost schema compliance, absence of runtime execution errors,
and penalization of premature `sys.ask_missing_info`.
"""
from __future__ import annotations

import copy
import json
from typing import Any


def evaluate_trajectory_quality(
    env: Any,
    tool_calls: list[dict[str, Any]],
    catalog_tools_map: dict[str, dict[str, Any]],
) -> float:
    """Computes a heuristic quality score for a generated trajectory without ground truth."""
    if not tool_calls:
        # Valid empty trajectory (e.g. Out-of-scope or complete cancellation)
        return 100.0

    score = 100.0

    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("arguments") or {}

        # 1. Penalize premature or ungrounded ask_missing_info
        if name == "sys.ask_missing_info":
            score -= 60.0

        # 2. Strict schema validation
        val = env.validate(name, args)
        if not val.get("ok"):
            score -= 40.0

        # 3. Parameter nullness check on required fields
        spec = catalog_tools_map.get(name) or {}
        reqs = (spec.get("parameters") or {}).get("required") or []
        for r in reqs:
            if r not in args or args[r] in (None, "", [], {}):
                score -= 25.0

    return score


def select_best_rollout(
    candidates: list[dict[str, Any]],
    catalog_tools_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Selects the highest scoring candidate trajectory.
    Each candidate is a dict with keys: 'tool_calls', 'state_hash', 'env'.
    """
    if not candidates:
        raise ValueError("Candidates list cannot be empty")
    if len(candidates) == 1:
        return candidates[0]

    best_candidate = candidates[0]
    best_score = -9999.0

    for cand in candidates:
        score = evaluate_trajectory_quality(
            cand["env"],
            cand["tool_calls"],
            catalog_tools_map,
        )
        if score > best_score:
            best_score = score
            best_candidate = cand

    return best_candidate
