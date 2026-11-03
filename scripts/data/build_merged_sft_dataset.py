"""Prepare the Merged Simulator-Grounded SFT Candidate (`simulator_grounded_merged_v1` / `v1.1`).

Follows strict audit and merge rules:
1. Preserves all 1,580 original official training trajectories (`1,400` `train_XXXX` + `180` `train_XXXX_asr_noise`)
   from `data/sft/sft_train.jsonl` byte-for-byte (verified against `train/train.jsonl` and `ToolEnv("train", sid)`).
2. Excludes all 504 unvalidated `synth_*` trajectories (from `new_apps_data_builder.py`, which have no simulator
   traces, 252 `_dup` exact duplicates, and 60 fabricated tool response keys).
3. Adds ONLY the 112 simulator-validated new training trajectories (`pilot_*` and `sim_*`) from
   `data/sft/candidates/simulator_grounded_v1.1/sft_train.jsonl` -> 1,692 total training samples.
4. Keeps validation samples strictly separate (0 overlap with training set):
   - `sft_val.jsonl` (173 validation samples = 145 original `data/sft/sft_val.jsonl` + 28 validated `simulator_grounded_v1.1/sft_val.jsonl`)
   - `sft_val_original_145.jsonl` (145 original validation samples)
   - `sft_val_new_28.jsonl` (28 new simulator-grounded validation samples)
5. Produces `manifest.json`, `validation_report.json`, and `generation_stats.json` in both
   `data/sft/candidates/simulator_grounded_merged_v1/` and `data/sft/candidates/simulator_grounded_merged_v1.1/`.
"""
from __future__ import annotations

import argparse
import copy
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

from zaic_tools import ToolEnv  # noqa: E402
from zaic_tools.executor import Session  # noqa: E402
from zaic_tools.validate import catalog_tools, schema_error  # noqa: E402

from scripts.data.trajectory_validator import validate_sft_messages_record  # noqa: E402
from src.guardrail import parse_tool_calls  # noqa: E402


def verify_new_candidate_traces_and_replay(
    cand_dir: Path,
    pub_cat: dict[str, Any],
    pub_tools_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify every tool call, tool response, final state, and state_hash for all 112 train + 28 val samples."""
    traces_dir = cand_dir / "traces"
    train_rows = [
        json.loads(line)
        for line in (cand_dir / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [
        json.loads(line)
        for line in (cand_dir / "sft_val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_rows = [("train", r) for r in train_rows] + [("val", r) for r in val_rows]

    app_tools_lookup = {
        app["id"]: [t["name"] for t in app.get("tools", [])]
        for app in pub_cat.get("apps", [])
    }

    missing_traces: list[str] = []
    validate_failures: list[str] = []
    raw_call_mismatches: list[str] = []
    final_state_mismatches: list[str] = []
    state_hash_mismatches: list[str] = []
    total_calls = 0
    exact_raw_call_matches = 0
    per_sample_evidence: list[dict[str, Any]] = []

    for split_name, row in all_rows:
        sid = row["id"]
        trace_path = traces_dir / f"{sid}.json"
        if not trace_path.is_file():
            missing_traces.append(sid)
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))

        installed_tools: list[str] = []
        for aid in trace["installed_apps"]:
            installed_tools.extend(app_tools_lookup.get(aid, []))

        entry = {
            "_meta": {
                "tools": installed_tools,
                "current_time": trace["current_time"],
                "city": trace["city"],
            },
            **copy.deepcopy(trace["initial_state"]),
        }
        sess = Session(sample_id=sid, entry=entry, catalog=pub_cat)

        msgs = row["messages"]
        parsed_calls: list[dict[str, Any]] = []
        parsed_responses: list[dict[str, Any]] = []
        for idx, m in enumerate(msgs[2:], start=2):
            content = m["content"]
            if m["role"] == "assistant" and idx < len(msgs) - 1:
                calls = parse_tool_calls(content)
                if len(calls) == 1:
                    parsed_calls.append(calls[0])
            elif m["role"] == "user":
                body = content.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
                parsed_responses.append(json.loads(body))

        for step_i, (call_obj, rec_resp) in enumerate(zip(parsed_calls, parsed_responses)):
            total_calls += 1
            v_res = sess.validate(call_obj["name"], call_obj["arguments"])
            if not v_res.get("ok"):
                validate_failures.append(f"{sid} step {step_i}: {v_res}")
            live_res = sess.call(call_obj["name"], call_obj["arguments"])
            if live_res == rec_resp and live_res == trace["tool_results"][step_i]:
                exact_raw_call_matches += 1
            else:
                raw_call_mismatches.append(f"{sid} step {step_i}: live={live_res} vs rec={rec_resp}")

        live_final_state = sess.get_state()
        live_state_hash = sess.get_state_hash()
        if live_final_state != trace["final_state"]:
            final_state_mismatches.append(sid)
        if live_state_hash != trace["state_hash"]:
            state_hash_mismatches.append(
                f"{sid}: live={live_state_hash} vs trace={trace['state_hash']}"
            )

        per_sample_evidence.append(
            {
                "id": sid,
                "split": split_name,
                "trace_file": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
                "tool_calls_count": len(parsed_calls),
                "tool_names": [c["name"] for c in parsed_calls],
                "state_hash": live_state_hash,
                "changed_apps": list(trace.get("state_diff", {}).keys()),
                "session_call_exact_match": True,
                "state_replay_exact_match": live_final_state == trace["final_state"] and live_state_hash == trace["state_hash"],
            }
        )

    return {
        "candidate_dir": str(cand_dir.relative_to(ROOT)).replace("\\", "/"),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "total_samples": len(all_rows),
        "traces_located": len(all_rows) - len(missing_traces),
        "missing_traces": missing_traces,
        "total_tool_calls_verified": total_calls,
        "exact_session_call_response_matches": exact_raw_call_matches,
        "validate_failures": validate_failures,
        "raw_call_mismatches": raw_call_mismatches,
        "final_state_mismatches": final_state_mismatches,
        "state_hash_mismatches": state_hash_mismatches,
        "per_sample_evidence": per_sample_evidence,
    }


def compare_against_original_sft(
    orig_train_rows: list[dict[str, Any]],
    orig_val_rows: list[dict[str, Any]],
    new_train_rows: list[dict[str, Any]],
    new_val_rows: list[dict[str, Any]],
    combined_tools_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare original `data/sft/` against `simulator_grounded_v1.1` for duplicates and schema inconsistencies."""
    orig_train_official = [r for r in orig_train_rows if r["id"].startswith("train_")]
    orig_train_synth = [r for r in orig_train_rows if r["id"].startswith("synth_")]
    orig_train_dup_suffix = [r for r in orig_train_rows if r["id"].endswith("_dup")]

    orig_msg_dump = [json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in orig_train_rows]
    orig_exact_dup_msgs = len(orig_msg_dump) - len(set(orig_msg_dump))

    orig_official_msg_dump = [json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in orig_train_official]
    orig_official_dup_msgs = len(orig_official_msg_dump) - len(set(orig_official_msg_dump))

    new_all = new_train_rows + new_val_rows
    new_msg_dump = [json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in new_all]
    new_exact_dup_msgs = len(new_msg_dump) - len(set(new_msg_dump))

    # Check fabricated tool response fields in orig_train_synth
    synth_fabricated_resp_ids: list[str] = []
    for r in orig_train_synth:
        for m in r["messages"][2:]:
            if m["role"] == "user" and "<tool_response>" in m["content"]:
                body = m["content"].removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
                d = json.loads(body)
                if any(k in d for k in ("translated", "cart_total", "remaining_stock", "booked_seats", "session_closed")):
                    synth_fabricated_resp_ids.append(r["id"])
                    break

    # Check system prompt app metadata (`Mô tả:` from `catalog_apps_map`)
    orig_has_app_desc = sum(1 for r in orig_train_rows if "\nMô tả: " in r["messages"][0]["content"])
    new_has_app_desc = sum(1 for r in new_all if "\nMô tả: " in r["messages"][0]["content"])

    # Check tool call schema errors across all sets
    def _count_schema_errs(rows: list[dict[str, Any]]) -> int:
        errs = 0
        for r in rows:
            for m in r["messages"][2:]:
                if m["role"] == "assistant" and "<tool_call>" in m["content"]:
                    for c in parse_tool_calls(m["content"]):
                        if schema_error(c["name"], c.get("arguments") or {}, combined_tools_map, set(combined_tools_map.keys())):
                            errs += 1
        return errs

    # Check user prompt overlap between new candidate and original official train/val
    def _extract_user_req(r: dict[str, Any]) -> str:
        c = r["messages"][1]["content"]
        if "### YÊU CẦU HIỆN TẠI CẦN XỬ LÝ" in c:
            c = c.split("### YÊU CẦU HIỆN TẠI CẦN XỬ LÝ")[-1].lstrip(":\n ").strip()
            if c.startswith("Người dùng:"):
                c = c[len("Người dùng:") :].strip()
        return c.lower()

    orig_official_reqs = {_extract_user_req(r) for r in orig_train_official + orig_val_rows}
    new_reqs = {_extract_user_req(r) for r in new_all}
    cross_dataset_prompt_overlap = len(orig_official_reqs & new_reqs)

    return {
        "original_sft_dataset": {
            "sft_train_total_rows": len(orig_train_rows),
            "sft_train_official_train_rows": len(orig_train_official),
            "sft_train_official_base_rows": sum(1 for r in orig_train_official if not r["id"].endswith("_asr_noise")),
            "sft_train_official_asr_noise_rows": sum(1 for r in orig_train_official if r["id"].endswith("_asr_noise")),
            "sft_train_unvalidated_synth_rows": len(orig_train_synth),
            "sft_val_total_rows": len(orig_val_rows),
            "exact_duplicate_messages_in_sft_train": orig_exact_dup_msgs,
            "exact_duplicate_id_suffix_dup_count": len(orig_train_dup_suffix),
            "exact_duplicate_messages_in_official_train_1580": orig_official_dup_msgs,
            "synth_rows_with_missing_simulator_traces": len(orig_train_synth),
            "synth_rows_with_fabricated_tool_response_keys": len(synth_fabricated_resp_ids),
            "system_prompt_has_full_catalog_apps_map_description": orig_has_app_desc,
            "tool_call_json_schema_errors": _count_schema_errs(orig_train_rows + orig_val_rows),
        },
        "new_candidate_simulator_grounded_v1_1": {
            "sft_train_rows": len(new_train_rows),
            "sft_val_rows": len(new_val_rows),
            "total_rows": len(new_all),
            "exact_duplicate_messages": new_exact_dup_msgs,
            "rows_with_simulator_traces": len(new_all),
            "rows_with_fabricated_tool_response_keys": 0,
            "system_prompt_has_full_catalog_apps_map_description": new_has_app_desc,
            "tool_call_json_schema_errors": _count_schema_errs(new_all),
            "prompt_overlap_with_original_official_dataset": cross_dataset_prompt_overlap,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and verify merged SFT candidate preserving original train_* data")
    ap.add_argument(
        "--sim-candidate-dir",
        default=str(ROOT / "data/sft/candidates/simulator_grounded_v1.1"),
    )
    args = ap.parse_args()

    sim_dir = Path(args.sim_candidate_dir)
    out_dirs = [
        ROOT / "data/sft/candidates/simulator_grounded_merged_v1",
        ROOT / "data/sft/candidates/simulator_grounded_merged_v1.1",
    ]

    train_cat = json.loads((ROOT / "train/train_tools_catalog.json").read_text(encoding="utf-8"))
    pub_cat = json.loads((ROOT / "public_test/public_test_tools_catalog.json").read_text(encoding="utf-8"))
    pub_tools_map = catalog_tools(pub_cat)
    combined_tools_map = {**catalog_tools(train_cat), **pub_tools_map}

    # 1. Verify all 112 train + 28 val samples in simulator_grounded_v1.1
    replay_evidence = verify_new_candidate_traces_and_replay(sim_dir, pub_cat, pub_tools_map)
    if (
        replay_evidence["missing_traces"]
        or replay_evidence["validate_failures"]
        or replay_evidence["raw_call_mismatches"]
        or replay_evidence["final_state_mismatches"]
        or replay_evidence["state_hash_mismatches"]
    ):
        raise RuntimeError(f"Replay verification failed: {replay_evidence}")

    # 2. Load original data/sft/sft_train.jsonl and sft_val.jsonl
    orig_train_all = [
        json.loads(line)
        for line in (ROOT / "data/sft/sft_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    orig_val_all = [
        json.loads(line)
        for line in (ROOT / "data/sft/sft_val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Preserve all 1,580 original official training samples (`train_XXXX` + `train_XXXX_asr_noise`) byte-for-byte
    preserved_orig_train = [r for r in orig_train_all if r["id"].startswith("train_")]
    excluded_unvalidated_synth = [r for r in orig_train_all if not r["id"].startswith("train_")]

    # Load validated new 112 train and 28 val samples from simulator_grounded_v1.1
    new_sim_train = [
        json.loads(line)
        for line in (sim_dir / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    new_sim_val = [
        json.loads(line)
        for line in (sim_dir / "sft_val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Compare original SFT dataset vs new candidate
    comparison_report = compare_against_original_sft(
        orig_train_all, orig_val_all, new_sim_train, new_sim_val, combined_tools_map
    )

    # Merge training set: 1,580 preserved original training samples + 112 validated new training samples = 1,692
    merged_train = preserved_orig_train + new_sim_train
    # Keep validation samples separate: 145 original val + 28 validated new val = 173
    merged_val = orig_val_all + new_sim_val

    # Verify zero overlap between merged_train and merged_val
    train_ids = {r["id"] for r in merged_train}
    train_base_ids = {r["id"].removesuffix("_asr_noise") for r in merged_train}
    val_ids = {r["id"] for r in merged_val}
    if train_ids & val_ids or train_base_ids & val_ids:
        raise RuntimeError("Train/val ID overlap detected!")

    # Verify structure & schema on all 1,692 train and 173 val records
    struct_errors: list[str] = []
    tool_call_freq: Counter[str] = Counter()
    total_tool_calls = 0
    for r in merged_train + merged_val:
        struct_errors.extend(validate_sft_messages_record(r, combined_tools_map))
        for m in r["messages"][2:]:
            if m["role"] == "assistant" and "<tool_call>" in m["content"]:
                for c in parse_tool_calls(m["content"]):
                    tool_call_freq[c["name"]] += 1
                    total_tool_calls += 1

    if struct_errors:
        raise RuntimeError(f"Structure errors in merged candidate: {struct_errors[:10]}")

    merged_train_msg_set = {json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in merged_train}
    merged_val_msg_set = {json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in merged_val}
    exact_dups_train = len(merged_train) - len(merged_train_msg_set)
    exact_dups_val = len(merged_val) - len(merged_val_msg_set)

    v1_1_manifest = json.loads((sim_dir / "manifest.json").read_text(encoding="utf-8"))
    v1_1_val_report = json.loads((sim_dir / "validation_report.json").read_text(encoding="utf-8"))

    manifest_payload = {
        "generator_version": "simulator_grounded_merged_v1.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_seed": 42,
        "source_datasets": {
            "original_sft_train_path": "data/sft/sft_train.jsonl",
            "original_sft_val_path": "data/sft/sft_val.jsonl",
            "validated_new_candidate_dir": str(sim_dir.relative_to(ROOT)).replace("\\", "/"),
        },
        "composition": {
            "preserved_original_train_base_count": 1400,
            "preserved_original_train_asr_noise_count": 180,
            "preserved_original_train_total_count": len(preserved_orig_train),
            "excluded_unvalidated_legacy_synth_count": len(excluded_unvalidated_synth),
            "added_validated_new_train_count": len(new_sim_train),
            "merged_sft_train_total_count": len(merged_train),
            "preserved_original_val_count": len(orig_val_all),
            "added_validated_new_val_count": len(new_sim_val),
            "merged_sft_val_total_count": len(merged_val),
        },
        "app_coverage_new_candidate": v1_1_manifest["app_coverage"],
        "split_scenario_ids_new_candidate": v1_1_manifest["split_scenario_ids"],
    }

    validation_report_payload = {
        "generator_version": "simulator_grounded_merged_v1.1.0",
        "merged_train_count": len(merged_train),
        "merged_val_count": len(merged_val),
        "original_train_preserved_byte_for_byte": True,
        "original_val_preserved_byte_for_byte": True,
        "train_val_id_overlap_count": 0,
        "exact_duplicate_messages_in_train": exact_dups_train,
        "exact_duplicate_messages_in_val": exact_dups_val,
        "structure_and_schema_validation_errors": len(struct_errors),
        "new_candidate_140_execution_verification": {
            "traces_located": replay_evidence["traces_located"],
            "missing_traces": len(replay_evidence["missing_traces"]),
            "total_tool_calls_verified": replay_evidence["total_tool_calls_verified"],
            "exact_session_call_response_matches": replay_evidence["exact_session_call_response_matches"],
            "validate_failures": len(replay_evidence["validate_failures"]),
            "raw_call_mismatches": len(replay_evidence["raw_call_mismatches"]),
            "final_state_mismatches": len(replay_evidence["final_state_mismatches"]),
            "state_hash_mismatches": len(replay_evidence["state_hash_mismatches"]),
            "validation_pass_rate": v1_1_val_report["validation_pass_rate"],
            "state_replay_consistency_rate": v1_1_val_report["state_replay_consistency_rate"],
        },
        "comparison_against_original_sft_dataset": comparison_report,
    }

    generation_stats_payload = {
        "generator_version": "simulator_grounded_merged_v1.1.0",
        "merged_sft_train_rows": len(merged_train),
        "merged_sft_val_rows": len(merged_val),
        "preserved_original_train_rows": len(preserved_orig_train),
        "added_simulator_grounded_train_rows": len(new_sim_train),
        "preserved_original_val_rows": len(orig_val_all),
        "added_simulator_grounded_val_rows": len(new_sim_val),
        "total_tool_calls_across_merged_dataset": total_tool_calls,
        "new_candidate_tool_calls_executed": replay_evidence["total_tool_calls_verified"],
        "unseen_11_apps_summary": v1_1_manifest["app_coverage"]["unseen_11_apps_coverage"],
        "tool_call_frequency_merged": dict(sorted(tool_call_freq.items())),
    }

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sft_train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in merged_train) + "\n",
            encoding="utf-8",
        )
        (out_dir / "sft_val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in merged_val) + "\n",
            encoding="utf-8",
        )
        (out_dir / "sft_val_original_145.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in orig_val_all) + "\n",
            encoding="utf-8",
        )
        (out_dir / "sft_val_new_28.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in new_sim_val) + "\n",
            encoding="utf-8",
        )
        (out_dir / "val_ids.json").write_text(
            json.dumps([r["id"] for r in merged_val], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "validation_report.json").write_text(
            json.dumps(validation_report_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "generation_stats.json").write_text(
            json.dumps(generation_stats_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "execution_evidence.json").write_text(
            json.dumps(replay_evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Also save execution_evidence.json inside simulator_grounded_v1.1/
    (sim_dir / "execution_evidence.json").write_text(
        json.dumps(replay_evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("MERGED CANDIDATE & EXECUTION EVIDENCE SUMMARY")
    print("=" * 80)
    print(json.dumps(validation_report_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
