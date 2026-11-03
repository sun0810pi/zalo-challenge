"""Strict trajectory and dataset validator for simulator-grounded SFT generation (ZAC 2026 LifeMate).

Enforces:
1. Schema validation:
   - Tool exists in the actual catalog.
   - Tool is installed in the scenario.
   - Arguments match the actual JSON schema (required parameters, types, enum values, formats, x-one-of).
2. Execution validation:
   - No unexpected rejected calls (ok=False).
   - No soft-failure noop calls (done=False) unless explicitly intended and verified.
   - No fabricated IDs: IDs used in write operations must be grounded in initial context/history or actual previous tool results.
   - Read-before-write requirements (x-ref) must be respected.
   - Multi-step execution must use actual intermediate results.
   - Intended actions must achieve their expected outcome (custom post-condition assertions).
3. State validation:
   - Capture the actual initial state.
   - Execute the full trajectory on the official simulator.
   - Capture the resulting state and compute changed application states.
   - Replay the same trajectory from a fresh equivalent Session and verify identical tool results and state hash.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from src.guardrail import parse_tool_calls


def _extract_tokens_and_ids_from_result(result: dict[str, Any]) -> tuple[set[str], dict[str, set[str]]]:
    """Extract all entity IDs and tokens from a single simulator tool result."""
    known_ids: set[str] = set()
    by_key: dict[str, set[str]] = {}
    if not isinstance(result, dict):
        return known_ids, by_key

    for k, v in result.items():
        if k in ("ok", "done", "message", "_hint"):
            continue
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for id_field in ("id", "merchant_id", "symbol", "route_id", "channel", "contact_id", "account_id", "beneficiary_id", "place_id", "recipe_id"):
                        val = item.get(id_field)
                        if isinstance(val, str) and val:
                            known_ids.add(val)
                            by_key.setdefault(k, set()).add(val)
        elif isinstance(v, dict):
            for id_field in ("id", "merchant_id", "symbol", "route_id", "channel", "contact_id", "account_id", "beneficiary_id", "place_id", "recipe_id"):
                val = v.get(id_field)
                if isinstance(val, str) and val:
                    known_ids.add(val)
                    by_key.setdefault(k, set()).add(val)
        elif isinstance(v, str) and k == "id" and v:
            known_ids.add(v)
            by_key.setdefault(k, set()).add(v)

    return known_ids, by_key


def compute_changed_state(initial_state: dict[str, Any], final_state: dict[str, Any]) -> dict[str, Any]:
    """Return dictionary of apps whose state changed between initial_state and final_state."""
    changed: dict[str, Any] = {}
    all_apps = set(initial_state.keys()) | set(final_state.keys())
    for app in sorted(all_apps):
        if initial_state.get(app) != final_state.get(app):
            changed[app] = copy.deepcopy(final_state.get(app))
    return changed


def validate_executed_trajectory(
    *,
    scenario_id: str,
    scenario_type: str,
    entry: dict[str, Any],
    sample: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    state_hash_val: str,
    catalog: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    session_cls: Any,
    expected_outcome_fn: Any | None = None,
    expect_mutation: bool = True,
) -> dict[str, Any]:
    """Run comprehensive Schema + Execution + State Replay validation on a generated trajectory.
    Returns a detailed validation result dict with `passed: bool` and all check breakdowns.
    """
    errors: list[str] = []
    schema_errors: list[str] = []
    unexpected_error_count = 0
    invalid_id_count = 0
    read_before_write_violations = 0

    # 1. Fresh Session for Replay Verification
    replay_session = session_cls(sample_id=scenario_id, entry=copy.deepcopy(entry), catalog=catalog)
    replay_initial_state = replay_session.get_state()

    if replay_initial_state != initial_state:
        errors.append("Initial state mismatch between execution session and fresh replay session.")

    if len(tool_calls) != len(tool_results):
        errors.append(f"Mismatch between tool_calls count ({len(tool_calls)}) and tool_results count ({len(tool_results)}).")

    # Seed grounded IDs from user audio_text and history tool_results
    grounded_ids: set[str] = set()
    grounded_collections: dict[str, set[str]] = {}
    user_text = sample.get("audio_text", "")
    grounded_tokens = set(re.findall(r"[A-Za-z0-9_]+", user_text))

    for turn in sample.get("history") or []:
        if turn.get("audio_text"):
            grounded_tokens.update(re.findall(r"[A-Za-z0-9_]+", turn["audio_text"]))
        for tc, tr in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []):
            t_app = tc.get("name", "").split(".", 1)[0]
            ids, by_k = _extract_tokens_and_ids_from_result(tr)
            grounded_ids.update(ids)
            for k, id_set in by_k.items():
                grounded_collections.setdefault(f"{t_app}.{k}", set()).update(id_set)

    replayed_results: list[dict[str, Any]] = []

    for step_idx, (tc, recorded_res) in enumerate(zip(tool_calls, tool_results)):
        t_name = tc.get("name", "")
        t_args = tc.get("arguments") or {}

        # A. Schema validation via official replay_session.validate()
        val_check = replay_session.validate(t_name, t_args)
        if not val_check.get("ok"):
            msg = f"Step {step_idx} ({t_name}): Schema error: {val_check.get('error')}"
            schema_errors.append(msg)
            errors.append(msg)

        # B. Read-before-write & Grounded ID check BEFORE executing step
        spec = catalog_tools_map.get(t_name) or {}
        kind = spec.get("kind", "write")
        props = (spec.get("parameters") or {}).get("properties") or {}

        for param_k, param_v in t_args.items():
            if not isinstance(param_v, str) or not param_v:
                continue
            p_spec = props.get(param_k) or {}
            xref = p_spec.get("x-ref")
            is_id_param = bool(xref) or (kind == "write" and param_k.endswith("_id"))

            if is_id_param:
                # Check if grounded in previous tool results, history, or user text
                if param_v not in grounded_ids and param_v not in grounded_tokens:
                    invalid_id_count += 1
                    read_before_write_violations += 1
                    errors.append(
                        f"Step {step_idx} ({t_name}): Ungrounded ID '{param_k}={param_v}' (x-ref={xref}). "
                        f"Must come from prior read tool result or history."
                    )
                elif xref:
                    # Also verify collection match if xref specifies a collection (e.g. grocery.products vs pantry.items)
                    # Note: some tools like smart_home.get_status return 'devices' or 'scenes', etc.
                    if xref in grounded_collections and param_v not in grounded_collections[xref] and param_v not in grounded_tokens:
                        invalid_id_count += 1
                        read_before_write_violations += 1
                        errors.append(
                            f"Step {step_idx} ({t_name}): Cross-collection ID violation for '{param_k}={param_v}'. "
                            f"Expected in '{xref}' ({sorted(grounded_collections[xref])}), not from another collection."
                        )

        # Also check nested item_id in list arguments (e.g. food_delivery.place_order items=[{item_id: ...}])
        for param_k, param_v in t_args.items():
            if isinstance(param_v, list):
                for elem in param_v:
                    if isinstance(elem, dict) and isinstance(elem.get("item_id"), str):
                        iid = elem["item_id"]
                        if iid not in grounded_ids and iid not in grounded_tokens:
                            invalid_id_count += 1
                            read_before_write_violations += 1
                            errors.append(
                                f"Step {step_idx} ({t_name}): Ungrounded nested item_id='{iid}'."
                            )

        # C. Execute on official replay session
        actual_res = replay_session.call(t_name, copy.deepcopy(t_args))
        replayed_results.append(actual_res)

        # Check that recorded result is identical to actual simulator execution result
        if actual_res != recorded_res:
            errors.append(
                f"Step {step_idx} ({t_name}): Recorded tool_result does not match fresh simulator replay result.\n"
                f"  Recorded: {json.dumps(recorded_res, ensure_ascii=False)}\n"
                f"  Replayed: {json.dumps(actual_res, ensure_ascii=False)}"
            )

        # Check for rejected call (ok=False) or unexpected noop (done=False)
        if not actual_res.get("ok"):
            unexpected_error_count += 1
            errors.append(f"Step {step_idx} ({t_name}): Rejected tool call: {actual_res}")
        elif actual_res.get("done") is False:
            unexpected_error_count += 1
            errors.append(f"Step {step_idx} ({t_name}): Soft-failure noop (done=False): {actual_res.get('message')}")

        # D. Harvest IDs from actual_res for subsequent steps
        t_app = t_name.split(".", 1)[0]
        new_ids, by_k = _extract_tokens_and_ids_from_result(actual_res)
        grounded_ids.update(new_ids)
        for k, id_set in by_k.items():
            grounded_collections.setdefault(f"{t_app}.{k}", set()).update(id_set)

    # 3. State Replay Consistency Check
    replay_final_state = replay_session.get_state()
    replay_state_hash = replay_session.get_state_hash()

    state_replay_consistent = (replay_final_state == final_state) and (replay_state_hash == state_hash_val)
    if not state_replay_consistent:
        errors.append(
            f"State replay mismatch: recorded_hash={state_hash_val} vs replay_hash={replay_state_hash}"
        )

    changed_apps = compute_changed_state(initial_state, final_state)

    # 4. Mutation vs Clarification/Negative-Condition/Cancel Policy Check
    if expect_mutation and not changed_apps:
        errors.append(
            f"Scenario '{scenario_id}' (type={scenario_type}) expected state mutation, but no app state changed."
        )
    elif not expect_mutation and changed_apps:
        errors.append(
            f"Scenario '{scenario_id}' (type={scenario_type}) expected NO state mutation, but apps {list(changed_apps.keys())} changed."
        )

    # 5. Scenario-specific expected outcome verification
    if expected_outcome_fn is not None:
        try:
            outcome_ok, outcome_err = expected_outcome_fn(
                initial_state=initial_state,
                final_state=final_state,
                changed_state=changed_apps,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            if not outcome_ok:
                errors.append(f"Expected outcome check failed: {outcome_err}")
        except Exception as exc:
            errors.append(f"Expected outcome check raised exception: {type(exc).__name__}: {exc}")

    return {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "passed": len(errors) == 0,
        "schema_valid": len(schema_errors) == 0,
        "execution_valid": unexpected_error_count == 0 and invalid_id_count == 0 and read_before_write_violations == 0,
        "state_replay_consistent": state_replay_consistent,
        "unexpected_error_count": unexpected_error_count,
        "invalid_id_count": invalid_id_count,
        "read_before_write_violations": read_before_write_violations,
        "changed_apps": list(changed_apps.keys()),
        "state_hash": state_hash_val,
        "errors": errors,
    }


def validate_sft_messages_record(
    record: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate serialized ChatML messages structure, `<think>` + `<tool_call>` formatting, and prompt compatibility."""
    errs: list[str] = []
    sid = record.get("id", "<missing_id>")
    msgs = record.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return [f"{sid}: messages must be a list of at least 3 turns (system, user, assistant)."]

    if msgs[0].get("role") != "system":
        errs.append(f"{sid}: Turn 0 must be role='system'.")
    sys_content = msgs[0].get("content", "")
    if "### BỐI CẢNH NGƯỜI DÙNG (CONTEXT)" not in sys_content or "### DANH SÁCH ỨNG DỤNG ĐÃ CÀI ĐẶT" not in sys_content:
        errs.append(f"{sid}: System prompt missing standard build_initial_messages() blocks.")
    if "[Ứng dụng:" not in sys_content:
        errs.append(f"{sid}: System prompt missing application header blocks.")

    if msgs[1].get("role") != "user":
        errs.append(f"{sid}: Turn 1 must be role='user'.")
    user_content = msgs[1].get("content", "")
    if "### LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ" not in user_content or "### YÊU CẦU HIỆN TẠI CẦN XỬ LÝ" not in user_content:
        errs.append(f"{sid}: User prompt missing standard history/request headers.")

    assistant_count = 0
    for idx, m in enumerate(msgs[2:], start=2):
        role = m.get("role")
        content = m.get("content", "")
        if idx % 2 == 0:
            if role != "assistant":
                errs.append(f"{sid}: Turn {idx} expected role='assistant', got {role!r}.")
            assistant_count += 1
            if "<think>" not in content or "</think>" not in content:
                errs.append(f"{sid}: Assistant turn {idx} missing <think>...</think> block.")
            if idx < len(msgs) - 1:
                # Intermediate assistant turn must have exactly 1 valid <tool_call>
                calls = parse_tool_calls(content)
                if len(calls) != 1:
                    errs.append(f"{sid}: Intermediate assistant turn {idx} must emit exactly 1 <tool_call>, got {len(calls)}.")
                else:
                    tname = calls[0]["name"]
                    if tname not in catalog_tools_map:
                        errs.append(f"{sid}: Assistant turn {idx} called unknown tool {tname!r}.")
            else:
                # Final assistant turn must NOT have <tool_call>
                if "<tool_call>" in content:
                    errs.append(f"{sid}: Final assistant turn {idx} must not contain <tool_call>.")
        else:
            if role != "user":
                errs.append(f"{sid}: Turn {idx} expected role='user' (<tool_response>), got {role!r}.")
            if not content.startswith("<tool_response>") or not content.endswith("</tool_response>"):
                errs.append(f"{sid}: User turn {idx} must wrap tool result in <tool_response>...</tool_response>.")

    if assistant_count == 0:
        errs.append(f"{sid}: No assistant turns found.")
    return errs


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    if len(norm) < n:
        return {norm}
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def compute_near_duplicate_stats(
    scenarios: list[dict[str, Any]],
    jaccard_threshold: float = 0.72,
) -> dict[str, Any]:
    """Compute exact duplicate and near-duplicate user request statistics within and across train/val splits."""
    seen_texts: dict[str, str] = {}
    exact_duplicates: list[tuple[str, str]] = []
    near_duplicates_within_split: list[dict[str, Any]] = []
    near_duplicates_cross_split: list[dict[str, Any]] = []

    ngrams_list = []
    for s in scenarios:
        sid = s["scenario_id"]
        txt = s["user_request"].strip()
        norm_txt = re.sub(r"\s+", " ", txt.lower())
        if norm_txt in seen_texts:
            exact_duplicates.append((seen_texts[norm_txt], sid))
        else:
            seen_texts[norm_txt] = sid
        ngrams_list.append((sid, s["split"], _char_ngrams(norm_txt, 3)))

    max_cross_split_jaccard = 0.0
    for i in range(len(ngrams_list)):
        id_a, sp_a, ng_a = ngrams_list[i]
        for j in range(i + 1, len(ngrams_list)):
            id_b, sp_b, ng_b = ngrams_list[j]
            union = len(ng_a | ng_b)
            sim = (len(ng_a & ng_b) / union) if union else 0.0
            if sp_a != sp_b and sim > max_cross_split_jaccard:
                max_cross_split_jaccard = sim
            if sim >= jaccard_threshold:
                entry = {"id_a": id_a, "id_b": id_b, "jaccard_3gram": round(sim, 4)}
                if sp_a == sp_b:
                    near_duplicates_within_split.append(entry)
                else:
                    near_duplicates_cross_split.append(entry)

    return {
        "exact_duplicate_count": len(exact_duplicates),
        "exact_duplicate_pairs": exact_duplicates,
        "near_duplicate_threshold_3gram_jaccard": jaccard_threshold,
        "near_duplicate_within_split_count": len(near_duplicates_within_split),
        "near_duplicate_cross_split_count": len(near_duplicates_cross_split),
        "max_cross_split_3gram_jaccard": round(max_cross_split_jaccard, 4),
    }

