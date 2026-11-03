"""Build Simulator-Grounded SFT Dataset (`simulator_grounded_v1`) for ZAC 2026 LifeMate.

Executes the mandatory 4-phase workflow:
  Phase 1 — Preflight: Verify official `zaic_tools.executor.Session` API & multi-step execution.
  Phase 2 — Pilot Generation: Execute & validate the 24 high-value pilot scenarios.
  Phase 3 — Dataset Generation: Execute & validate all 140 scenarios (112 train / 28 val),
            serialize ChatML trajectories using `build_initial_messages()` with `SemanticToolRouter`
            and `catalog_apps_map` (matching `run_infer.py`), and write all deliverables to
            `data/sft/candidates/simulator_grounded_v1/`.
  Phase 4 — Final Verification: Reload JSONL files, validate every record's schema/ChatML/labels,
            verify prompt compatibility with `run_infer.py` and `train_sft.py`, and compare stats
            against the previous `data/sft/` dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "public_test") not in sys.path:
    sys.path.insert(0, str(ROOT / "public_test"))

from zaic_tools.executor import Session  # noqa: E402
from zaic_tools.validate import catalog_tools  # noqa: E402

from scripts.data.scenario_library import (  # noqa: E402
    build_expanded_scenarios,
    build_pilot_24_scenarios,
)
from scripts.data.simulator_scenario_engine import (  # noqa: E402
    ScenarioSpec,
    execute_scenario_on_simulator,
)
from scripts.data.trajectory_validator import (  # noqa: E402
    compute_near_duplicate_stats,
    validate_executed_trajectory,
    validate_sft_messages_record,
)
from src.agent import LifeMateAgent  # noqa: E402
from src.preprocessor_a import ContextLanguagePreprocessorA  # noqa: E402
from src.prompt_builder import build_initial_messages  # noqa: E402
from src.router import SemanticToolRouter  # noqa: E402

GENERATOR_VERSION = "simulator_grounded_v1.0.0"
DEFAULT_SEED = 42


def _compact_json(obj: Any, max_len: int = 120) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _summarize_step_observation(prev_res: dict[str, Any], next_args: dict[str, Any]) -> str:
    """Summarize what concrete ID/value from `prev_res` grounds `next_args`."""
    matched_pairs: list[str] = []
    prev_dump = json.dumps(prev_res, ensure_ascii=False)
    for k, v in (next_args or {}).items():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            v_str = str(v)
            if v_str and v_str in prev_dump:
                matched_pairs.append(f"{k}={v!r}")
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for ik, iv in item.items():
                        if isinstance(iv, str) and iv in prev_dump:
                            matched_pairs.append(f"{ik}={iv!r}")
    if matched_pairs:
        return "trích xuất " + ", ".join(matched_pairs[:3])
    # Fallback summary of top-level keys in prev_res
    keys = [k for k in prev_res.keys() if k != "ok"]
    return f"dữ liệu ({', '.join(keys[:3])})" if keys else "kết quả trả về"


def build_concise_step_thought(
    step_idx: int,
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    spec: ScenarioSpec,
) -> str:
    """Build concise, decision-focused Vietnamese reasoning grounded in actual simulator results."""
    tc = tool_calls[step_idx]
    t_name = tc["name"]
    t_args = tc.get("arguments") or {}
    total_steps = len(tool_calls)

    if t_name == "sys.ask_missing_info":
        target_tool = t_args.get("tool_name", "")
        params = ", ".join(t_args.get("params") or [])
        if step_idx == 0:
            return (
                f"Người dùng chưa cung cấp tham số bắt buộc ({params}) cho `{target_tool}`. "
                f"Gọi `sys.ask_missing_info` để yêu cầu bổ sung."
            )
        prev_tool = tool_calls[step_idx - 1]["name"]
        return (
            f"Từ kết quả `{prev_tool}`, vẫn thiếu hoặc chưa xác định duy nhất tham số ({params}) cho `{target_tool}`. "
            f"Gọi `sys.ask_missing_info` để hỏi rõ người dùng."
        )

    if step_idx == 0:
        if total_steps > 1:
            next_tool = tool_calls[1]["name"]
            return (
                f"Bước 1/{total_steps}: Gọi `{t_name}` ({_compact_json(t_args, 90)}) "
                f"để lấy dữ liệu và ID thực tế trước khi thực hiện `{next_tool}`."
            )
        return (
            f"Gọi `{t_name}` ({_compact_json(t_args, 90)}) "
            f"để tra cứu thông tin và kiểm tra điều kiện yêu cầu của người dùng."
        )

    prev_tool = tool_calls[step_idx - 1]["name"]
    prev_res = tool_results[step_idx - 1]
    obs_summary = _summarize_step_observation(prev_res, t_args)
    return (
        f"Bước {step_idx + 1}/{total_steps}: Dựa trên kết quả `{prev_tool}` ({obs_summary}), "
        f"gọi `{t_name}` với tham số {_compact_json(t_args, 100)}."
    )


def build_closing_thought_and_reply(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    spec: ScenarioSpec,
) -> tuple[str, str]:
    """Construct the final assistant `<think>` and user-facing response."""
    if not tool_calls:
        thought = (
            "Người dùng đã đổi ý và hủy toàn bộ yêu cầu ở cuối câu nói. "
            "Không thực hiện bất kỳ lệnh gọi công cụ nào."
        )
        reply = "Mình đã ghi nhận bạn hủy yêu cầu và không thực hiện thao tác nào."
        return thought, reply

    last_call = tool_calls[-1]
    last_name = last_call["name"]
    last_res = tool_results[-1]

    if last_name == "sys.ask_missing_info":
        t_args = last_call.get("arguments") or {}
        target_tool = t_args.get("tool_name", "")
        params = ", ".join(t_args.get("params") or [])
        thought = f"Đã gọi `sys.ask_missing_info` cho `{target_tool}` (cần bổ sung: {params})."
        reply = f"Vui lòng cho mình biết thêm thông tin ({params}) để thực hiện yêu cầu nhé."
        return thought, reply

    if not spec.expect_mutation:
        thought = (
            f"Đã kiểm tra kết quả từ `{last_name}` ({_compact_json(last_res, 110)}). "
            f"Điều kiện thực hiện thao tác ghi/đặt tiếp theo không thỏa mãn nên dừng lại."
        )
        reply = "Mình đã kiểm tra thông tin theo yêu cầu. Do điều kiện hiện tại không phù hợp nên mình không thực hiện thao tác tiếp theo."
        return thought, reply

    thought = (
        f"Lệnh `{last_name}` đã thực thi thành công trên hệ thống ({_compact_json(last_res, 110)}). "
        f"Hoàn tất toàn bộ yêu cầu của người dùng."
    )
    reply = "Đã hoàn thành yêu cầu của bạn."
    return thought, reply


def serialize_trace_to_sft_record(
    spec: ScenarioSpec,
    trace: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    catalog_apps_map: dict[str, dict[str, Any]],
    router: SemanticToolRouter,
    preprocessor: ContextLanguagePreprocessorA,
) -> dict[str, Any]:
    """Serialize an executed simulator trace into the exact ChatML SFT format compatible with `run_infer.py`."""
    sample = trace["sample"]
    raw_user_text = sample.get("audio_text") or sample.get("user") or ""
    cleaned_user_text = preprocessor.preprocess(raw_user_text, sample)

    messages = build_initial_messages(
        sample=sample,
        final_user_text=cleaned_user_text,
        catalog_tools_map=catalog_tools_map,
        history_transcripts={},
        router=router,
        catalog_apps_map=catalog_apps_map,
    )

    tool_calls = trace["tool_calls"]
    tool_results = trace["tool_results"]

    for step_idx, (tc, tr) in enumerate(zip(tool_calls, tool_results)):
        thought = build_concise_step_thought(step_idx, tool_calls, tool_results, spec)
        call_payload = {"name": tc["name"], "arguments": tc.get("arguments") or {}}
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"<think>\n{thought}\n</think>\n"
                    f"<tool_call>\n{json.dumps(call_payload, ensure_ascii=False)}\n</tool_call>"
                ),
            }
        )
        prompt_res = LifeMateAgent._augment_empty_read_hint(
            tc["name"], tc.get("arguments") or {}, tr
        )
        messages.append(
            {
                "role": "user",
                "content": f"<tool_response>\n{json.dumps(prompt_res, ensure_ascii=False)}\n</tool_response>",
            }
        )

    closing_thought, closing_reply = build_closing_thought_and_reply(tool_calls, tool_results, spec)
    messages.append(
        {
            "role": "assistant",
            "content": f"<think>\n{closing_thought}\n</think>\n{closing_reply}",
        }
    )

    return {
        "id": spec.scenario_id,
        "split": spec.split,
        "scenario_type": spec.scenario_type,
        "template_id": spec.template_id,
        "primary_apps": spec.primary_apps,
        "messages": messages,
    }


def run_phase1_preflight(pub_cat: dict[str, Any], pub_tools_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Phase 1 — Preflight: Execute a synthetic multi-step scenario on the official `Session` and verify results."""
    pilot_specs = build_pilot_24_scenarios()
    spec0 = pilot_specs[0]  # pilot_001_parking_reserve_select (parking.find -> parking.reserve)
    trace = execute_scenario_on_simulator(spec0, pub_cat, pub_tools_map, Session)
    val = validate_executed_trajectory(
        scenario_id=spec0.scenario_id,
        scenario_type=spec0.scenario_type,
        entry=trace["entry"],
        sample=trace["sample"],
        tool_calls=trace["tool_calls"],
        tool_results=trace["tool_results"],
        initial_state=trace["initial_state"],
        final_state=trace["final_state"],
        state_hash_val=trace["state_hash"],
        catalog=pub_cat,
        catalog_tools_map=pub_tools_map,
        session_cls=Session,
        expected_outcome_fn=spec0.verify_fn,
        expect_mutation=spec0.expect_mutation,
    )
    if not val["passed"] or len(trace["tool_calls"]) < 2:
        raise RuntimeError(f"Phase 1 preflight failed: {val['errors']}")
    return {
        "scenario_id": spec0.scenario_id,
        "tool_calls": trace["tool_calls"],
        "tool_results": trace["tool_results"],
        "state_hash": trace["state_hash"],
        "changed_apps": val["changed_apps"],
        "passed": val["passed"],
    }


def run_validator_negative_self_tests(
    pub_cat: dict[str, Any],
    pub_tools_map: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    """Verify that `validate_executed_trajectory` catches fabricated IDs, RBW violations, and bad state hashes."""
    spec0 = build_pilot_24_scenarios()[0]
    trace = execute_scenario_on_simulator(spec0, pub_cat, pub_tools_map, Session)

    # 1. Fabricated ID test
    bad_calls_id = [
        trace["tool_calls"][0],
        {"name": "parking.reserve", "arguments": {"lot_id": "lot_fabricated_999", "from_time": "2026-11-20T14:00:00+07:00", "hours": 3}},
    ]
    res_fab = validate_executed_trajectory(
        scenario_id="neg_test_fabricated_id",
        scenario_type="multi_result_selection",
        entry=trace["entry"],
        sample=trace["sample"],
        tool_calls=bad_calls_id,
        tool_results=trace["tool_results"],
        initial_state=trace["initial_state"],
        final_state=trace["final_state"],
        state_hash_val=trace["state_hash"],
        catalog=pub_cat,
        catalog_tools_map=pub_tools_map,
        session_cls=Session,
    )

    # 2. Read-before-write violation test (calling parking.reserve without parking.find)
    bad_calls_rbw = [trace["tool_calls"][1]]
    res_rbw = validate_executed_trajectory(
        scenario_id="neg_test_rbw_violation",
        scenario_type="read_before_write",
        entry=trace["entry"],
        sample=trace["sample"],
        tool_calls=bad_calls_rbw,
        tool_results=[trace["tool_results"][1]],
        initial_state=trace["initial_state"],
        final_state=trace["final_state"],
        state_hash_val=trace["state_hash"],
        catalog=pub_cat,
        catalog_tools_map=pub_tools_map,
        session_cls=Session,
    )

    # 3. Corrupted state hash test
    res_hash = validate_executed_trajectory(
        scenario_id="neg_test_state_hash",
        scenario_type="multi_result_selection",
        entry=trace["entry"],
        sample=trace["sample"],
        tool_calls=trace["tool_calls"],
        tool_results=trace["tool_results"],
        initial_state=trace["initial_state"],
        final_state=trace["final_state"],
        state_hash_val="0" * 64,
        catalog=pub_cat,
        catalog_tools_map=pub_tools_map,
        session_cls=Session,
    )

    return {
        "catches_fabricated_id": (not res_fab["passed"]) and res_fab["invalid_id_count"] > 0,
        "catches_rbw_violation": (not res_rbw["passed"]) and res_rbw["read_before_write_violations"] > 0,
        "catches_corrupted_state_hash": (not res_hash["passed"]) and (not res_hash["state_replay_consistent"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Simulator-Grounded SFT Dataset (simulator_grounded_v1)")
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "data/sft/candidates/simulator_grounded_v1"),
        help="Target directory for simulator_grounded_v1 deliverables",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    traces_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    pub_cat_path = ROOT / "public_test/public_test_tools_catalog.json"
    pub_cat = json.loads(pub_cat_path.read_text(encoding="utf-8"))
    pub_tools_map = catalog_tools(pub_cat)
    pub_apps_map = {app["id"]: app for app in pub_cat.get("apps", [])}

    router = SemanticToolRouter(top_k_recommended=4)
    preprocessor = ContextLanguagePreprocessorA()

    print("=" * 80)
    print("PHASE 1 — PREFLIGHT CHECK ON OFFICIAL SIMULATOR")
    print("=" * 80)
    preflight_info = run_phase1_preflight(pub_cat, pub_tools_map)
    neg_self_tests = run_validator_negative_self_tests(pub_cat, pub_tools_map)
    print(f"[Preflight] Scenario: {preflight_info['scenario_id']}")
    for idx, (tc, tr) in enumerate(zip(preflight_info["tool_calls"], preflight_info["tool_results"])):
        print(f"  Call {idx+1}: {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
        print(f"  Real Simulator Result {idx+1}: {json.dumps(tr, ensure_ascii=False)}")
    print(f"[Preflight] Final state_hash: {preflight_info['state_hash']} | Changed apps: {preflight_info['changed_apps']}")
    print(f"[Preflight] Validator negative self-tests: {neg_self_tests}")
    if not all(neg_self_tests.values()):
        raise RuntimeError(f"Validator negative self-test failed: {neg_self_tests}")

    print("\n" + "=" * 80)
    print("PHASE 2 — PILOT GENERATION (24 SCENARIOS)")
    print("=" * 80)
    pilot_specs = build_pilot_24_scenarios()
    pilot_passed = 0
    for spec in pilot_specs:
        trace = execute_scenario_on_simulator(spec, pub_cat, pub_tools_map, Session)
        val = validate_executed_trajectory(
            scenario_id=spec.scenario_id,
            scenario_type=spec.scenario_type,
            entry=trace["entry"],
            sample=trace["sample"],
            tool_calls=trace["tool_calls"],
            tool_results=trace["tool_results"],
            initial_state=trace["initial_state"],
            final_state=trace["final_state"],
            state_hash_val=trace["state_hash"],
            catalog=pub_cat,
            catalog_tools_map=pub_tools_map,
            session_cls=Session,
            expected_outcome_fn=spec.verify_fn,
            expect_mutation=spec.expect_mutation,
        )
        if val["passed"] and trace["execution_exception"] is None:
            pilot_passed += 1
        else:
            raise RuntimeError(f"Pilot scenario {spec.scenario_id} failed: {val['errors']}")
    print(f"[Phase 2 Pilot] {pilot_passed}/{len(pilot_specs)} scenarios passed 100% simulator & state replay checks.")

    print("\n" + "=" * 80)
    print("PHASE 3 — EXPANDED DATASET GENERATION & VALIDATION")
    print("=" * 80)
    all_specs = build_expanded_scenarios()

    # Clear old trace files if any
    for old_f in traces_dir.glob("*.json"):
        old_f.unlink()

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    validation_details: list[dict[str, Any]] = []

    schema_valid_count = 0
    exec_valid_count = 0
    replay_consistent_count = 0
    total_unexpected_errors = 0
    total_invalid_ids = 0
    total_rbw_violations = 0
    total_tool_calls_executed = 0

    app_counts_all: Counter[str] = Counter()
    app_counts_train: Counter[str] = Counter()
    app_counts_val: Counter[str] = Counter()
    type_counts_all: Counter[str] = Counter()
    type_counts_train: Counter[str] = Counter()
    type_counts_val: Counter[str] = Counter()
    tool_call_counts: Counter[str] = Counter()

    for spec in all_specs:
        trace = execute_scenario_on_simulator(spec, pub_cat, pub_tools_map, Session)
        val_res = validate_executed_trajectory(
            scenario_id=spec.scenario_id,
            scenario_type=spec.scenario_type,
            entry=trace["entry"],
            sample=trace["sample"],
            tool_calls=trace["tool_calls"],
            tool_results=trace["tool_results"],
            initial_state=trace["initial_state"],
            final_state=trace["final_state"],
            state_hash_val=trace["state_hash"],
            catalog=pub_cat,
            catalog_tools_map=pub_tools_map,
            session_cls=Session,
            expected_outcome_fn=spec.verify_fn,
            expect_mutation=spec.expect_mutation,
        )

        sft_record = serialize_trace_to_sft_record(
            spec=spec,
            trace=trace,
            catalog_tools_map=pub_tools_map,
            catalog_apps_map=pub_apps_map,
            router=router,
            preprocessor=preprocessor,
        )
        msg_errors = validate_sft_messages_record(sft_record, pub_tools_map)
        if msg_errors:
            val_res["passed"] = False
            val_res["errors"].extend(msg_errors)

        if trace["execution_exception"] is not None:
            val_res["passed"] = False
            val_res["errors"].append(f"Execution exception: {trace['execution_exception']}")

        # Update counters
        if val_res["schema_valid"]:
            schema_valid_count += 1
        if val_res["execution_valid"]:
            exec_valid_count += 1
        if val_res["state_replay_consistent"]:
            replay_consistent_count += 1
        total_unexpected_errors += val_res["unexpected_error_count"]
        total_invalid_ids += val_res["invalid_id_count"]
        total_rbw_violations += val_res["read_before_write_violations"]
        total_tool_calls_executed += len(trace["tool_calls"])

        for tc in trace["tool_calls"]:
            tool_call_counts[tc["name"]] += 1

        # Compute state diff for trace serialization
        state_diff = {}
        for app_k, after_v in trace["final_state"].items():
            before_v = trace["initial_state"].get(app_k)
            if before_v != after_v:
                state_diff[app_k] = {"before": before_v, "after": after_v}

        trace_payload = {
            "scenario_id": spec.scenario_id,
            "split": spec.split,
            "template_id": spec.template_id,
            "scenario_type": spec.scenario_type,
            "primary_apps": spec.primary_apps,
            "installed_apps": spec.installed_apps,
            "objective": spec.objective,
            "expect_mutation": spec.expect_mutation,
            "user_request": spec.user_request,
            "current_time": spec.current_time,
            "city": spec.city,
            "initial_state": trace["initial_state"],
            "tool_calls": trace["tool_calls"],
            "tool_results": trace["tool_results"],
            "final_state": trace["final_state"],
            "state_diff": state_diff,
            "state_hash": trace["state_hash"],
            "validation": val_res,
        }
        (traces_dir / f"{spec.scenario_id}.json").write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        validation_details.append(val_res)

        if val_res["passed"]:
            jsonl_row = {
                "id": sft_record["id"],
                "messages": sft_record["messages"],
            }
            if spec.split == "train":
                train_records.append(jsonl_row)
            else:
                val_records.append(jsonl_row)

            type_counts_all[spec.scenario_type] += 1
            if spec.split == "train":
                type_counts_train[spec.scenario_type] += 1
            else:
                type_counts_val[spec.scenario_type] += 1

            for app_name in spec.primary_apps:
                app_counts_all[app_name] += 1
                if spec.split == "train":
                    app_counts_train[app_name] += 1
                else:
                    app_counts_val[app_name] += 1
        else:
            failure_records.append(trace_payload)

    # Check template separation and near-duplicate statistics
    train_templates = {s.template_id for s in all_specs if s.split == "train"}
    val_templates = {s.template_id for s in all_specs if s.split == "val"}
    overlapping_templates = sorted(train_templates & val_templates)

    dup_stats = compute_near_duplicate_stats(
        [
            {
                "scenario_id": s.scenario_id,
                "split": s.split,
                "user_request": s.user_request,
                "primary_apps": s.primary_apps,
            }
            for s in all_specs
        ]
    )

    # Save sft_train.jsonl, sft_val.jsonl, failures.jsonl
    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"
    failures_path = out_dir / "failures.jsonl"

    train_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_records) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_records) + "\n",
        encoding="utf-8",
    )
    failures_path.write_text(
        ("\n".join(json.dumps(r, ensure_ascii=False) for r in failure_records) + "\n")
        if failure_records
        else "",
        encoding="utf-8",
    )

    total_attempted = len(all_specs)
    total_accepted = len(train_records) + len(val_records)
    total_rejected = len(failure_records)

    unseen_11_apps = [
        "ecommerce",
        "event_tickets",
        "grocery",
        "habit_tracker",
        "pantry",
        "parking",
        "public_transit",
        "recipes",
        "stocks",
        "train_bus_tickets",
        "translator",
    ]
    unseen_coverage = {
        app: {
            "total": app_counts_all.get(app, 0),
            "train": app_counts_train.get(app, 0),
            "val": app_counts_val.get(app, 0),
        }
        for app in unseen_11_apps
    }

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_seed": args.seed,
        "simulator": {
            "package": "zaic_tools",
            "package_version": "0.1.0",
            "executor_class": "zaic_tools.executor.Session",
            "catalog_path": "public_test/public_test_tools_catalog.json",
            "catalog_schema_version": pub_cat.get("schema_version", "2.7.0"),
            "catalog_apps_count": len(pub_cat.get("apps", [])),
            "catalog_tools_count": len(pub_tools_map),
        },
        "prompt_configuration": {
            "builder": "src.prompt_builder.build_initial_messages",
            "router": "src.router.SemanticToolRouter(top_k_recommended=4)",
            "preprocessor": "src.preprocessor_a.ContextLanguagePreprocessorA",
            "uses_catalog_apps_map": True,
        },
        "counts": {
            "pilot_scenarios": len(pilot_specs),
            "total_scenarios_attempted": total_attempted,
            "accepted_count": total_accepted,
            "rejected_count": total_rejected,
            "train_count": len(train_records),
            "val_count": len(val_records),
            "train_ratio": round(len(train_records) / max(1, total_accepted), 4),
            "val_ratio": round(len(val_records) / max(1, total_accepted), 4),
            "total_simulator_tool_calls_executed": total_tool_calls_executed,
        },
        "app_coverage": {
            "primary_apps_total": dict(sorted(app_counts_all.items())),
            "primary_apps_train": dict(sorted(app_counts_train.items())),
            "primary_apps_val": dict(sorted(app_counts_val.items())),
            "unseen_11_apps_coverage": unseen_coverage,
        },
        "split_scenario_ids": {
            "train": [s.scenario_id for s in all_specs if s.split == "train"],
            "val": [s.scenario_id for s in all_specs if s.split == "val"],
        },
    }

    validation_report = {
        "generator_version": GENERATOR_VERSION,
        "total_attempted": total_attempted,
        "accepted_count": total_accepted,
        "rejected_count": total_rejected,
        "validation_pass_rate": round(total_accepted / max(1, total_attempted), 4),
        "schema_validation_pass_rate": round(schema_valid_count / max(1, total_attempted), 4),
        "execution_success_rate": round(exec_valid_count / max(1, total_attempted), 4),
        "state_replay_consistency_rate": round(replay_consistent_count / max(1, total_attempted), 4),
        "unexpected_error_count": total_unexpected_errors,
        "invalid_id_count": total_invalid_ids,
        "read_before_write_violations": total_rbw_violations,
        "template_overlap_between_train_and_val": overlapping_templates,
        "duplicate_and_near_duplicate_stats": dup_stats,
        "validator_negative_self_tests": neg_self_tests,
        "sample_counts_per_app": {
            "all": dict(sorted(app_counts_all.items())),
            "train": dict(sorted(app_counts_train.items())),
            "val": dict(sorted(app_counts_val.items())),
        },
        "sample_counts_per_scenario_type": {
            "all": dict(sorted(type_counts_all.items())),
            "train": dict(sorted(type_counts_train.items())),
            "val": dict(sorted(type_counts_val.items())),
        },
    }

    generation_stats = {
        "generator_version": GENERATOR_VERSION,
        "total_scenarios": total_accepted,
        "train_scenarios": len(train_records),
        "val_scenarios": len(val_records),
        "total_tool_calls_executed": total_tool_calls_executed,
        "avg_tool_calls_per_scenario": round(total_tool_calls_executed / max(1, total_accepted), 2),
        "tool_call_frequency": dict(sorted(tool_call_counts.items())),
        "scenario_type_distribution": dict(sorted(type_counts_all.items())),
        "unseen_11_apps_summary": unseen_coverage,
        "preflight_verification": preflight_info,
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "generation_stats.json").write_text(
        json.dumps(generation_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[Phase 3] Attempted: {total_attempted} | Accepted: {total_accepted} | Rejected: {total_rejected}")
    print(f"[Phase 3] Split: train={len(train_records)} ({manifest['counts']['train_ratio']*100:.1f}%), val={len(val_records)} ({manifest['counts']['val_ratio']*100:.1f}%)")
    print(f"[Phase 3] Total real simulator tool calls executed: {total_tool_calls_executed}")
    print(f"[Phase 3] Template overlap between train & val: {len(overlapping_templates)}")
    print(f"[Phase 3] Duplicate / Near-duplicate stats: {dup_stats}")


if __name__ == "__main__":
    main()
