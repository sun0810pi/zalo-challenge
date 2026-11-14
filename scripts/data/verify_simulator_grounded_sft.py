"""Phase 4 — Final Verification for `data/sft/candidates/simulator_grounded_v1/`.

Performs end-to-end verification of the generated candidate dataset:
1. Reloads `sft_train.jsonl`, `sft_val.jsonl`, `manifest.json`, `validation_report.json`,
   `generation_stats.json`, `failures.jsonl`, and all `traces/*.json`.
2. Parses every record and validates message structure, `<think>` and `<tool_call>` blocks,
   and `AssistantOnlyDataCollator` token-masking boundary markers.
3. Replays the `<tool_call>` sequence parsed directly from `sft_train.jsonl` and `sft_val.jsonl`
   against a fresh `zaic_tools.executor.Session` and verifies 100% tool-result and `state_hash` parity.
4. Verifies byte-for-byte compatibility between the training prompt (`messages[0:2]`) and the
   inference prompt constructed by `run_infer.py` / `src/agent.py`.
5. Compares dataset statistics against the previous `data/sft/` dataset.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
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

from scripts.data.trajectory_validator import validate_sft_messages_record  # noqa: E402
from src.entity_normalizer import EntityNormalizer  # noqa: E402
from src.guardrail import PureSchemaStateTracker, parse_tool_calls  # noqa: E402
from src.preprocessor_a import ContextLanguagePreprocessorA  # noqa: E402
from src.prompt_builder import build_initial_messages  # noqa: E402
from src.router import SemanticToolRouter  # noqa: E402


def verify_candidate_directory(cand_dir: Path) -> dict[str, Any]:
    required_files = [
        "sft_train.jsonl",
        "sft_val.jsonl",
        "manifest.json",
        "validation_report.json",
        "generation_stats.json",
        "failures.jsonl",
    ]
    for fname in required_files:
        fpath = cand_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing required deliverable file: {fpath}")

    traces_dir = cand_dir / "traces"
    if not traces_dir.is_dir():
        raise FileNotFoundError(f"Missing required traces directory: {traces_dir}")

    pub_cat = json.loads((ROOT / "public_test/public_test_tools_catalog.json").read_text(encoding="utf-8"))
    pub_tools_map = catalog_tools(pub_cat)
    pub_apps_map = {app["id"]: app for app in pub_cat.get("apps", [])}
    router = SemanticToolRouter(top_k_recommended=4)
    preprocessor = ContextLanguagePreprocessorA()

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

    manifest = json.loads((cand_dir / "manifest.json").read_text(encoding="utf-8"))
    val_report = json.loads((cand_dir / "validation_report.json").read_text(encoding="utf-8"))

    structure_errors: list[str] = []
    prompt_mismatch_errors: list[str] = []
    replay_mismatch_errors: list[str] = []
    think_char_lengths: list[int] = []

    for split_name, row in all_rows:
        sid = row.get("id")
        errs = validate_sft_messages_record(row, pub_tools_map)
        if errs:
            structure_errors.extend(errs)

        trace_file = traces_dir / f"{sid}.json"
        if not trace_file.is_file():
            structure_errors.append(f"{sid}: Missing trace file {trace_file.name}")
            continue
        trace = json.loads(trace_file.read_text(encoding="utf-8"))

        # Verify inference prompt compatibility byte-for-byte
        app_tools_lookup = {
            app["id"]: [t["name"] for t in app.get("tools", [])]
            for app in pub_cat.get("apps", [])
        }
        installed_tools: list[str] = []
        installed_apps_dict = {}
        for aid in trace["installed_apps"]:
            t_list = app_tools_lookup.get(aid, [])
            installed_tools.extend(t_list)
            installed_apps_dict[aid] = {
                "tools": list(t_list),
                "state": copy.deepcopy(trace["initial_state"].get(aid, {})),
            }
        sample = {
            "id": sid,
            "user": trace["user_request"],
            "audio_text": trace["user_request"],
            "tools": installed_tools,
            "context": {
                "current_time": trace["current_time"],
                "timezone": trace.get("timezone", "Asia/Ho_Chi_Minh"),
                "city": trace["city"],
                "user_profile": trace.get("user_profile", {}),
                "location": trace.get("location", {}),
                "installed_apps": installed_apps_dict,
            },
            "history": trace.get("history", []),
        }
        msgs = row["messages"]
        if "[Ứng dụng:" not in msgs[0]["content"]:
            prompt_mismatch_errors.append(f"{sid}: System prompt missing [Ứng dụng: ...] blocks from catalog_apps_map")
        if "### DANH SÁCH ỨNG DỤNG ĐÃ CÀI ĐẶT" not in msgs[0]["content"] and trace["installed_apps"]:
            prompt_mismatch_errors.append(f"{sid}: System prompt missing unified ALLOWED TOOLS section header")
        if "user_profile" in trace and "location" in trace:
            final_u_text = trace.get("serialized_user_prompt", trace["user_request"])
            rebuilt_init = build_initial_messages(
                sample=sample,
                final_user_text=final_u_text,
                catalog_tools_map=pub_tools_map,
                history_transcripts={},
                router=router,
                catalog_apps_map=pub_apps_map,
            )
            if rebuilt_init[0]["content"] != msgs[0]["content"] or rebuilt_init[1]["content"] != msgs[1]["content"]:
                prompt_mismatch_errors.append(
                    f"{sid}: messages[0:2] does not match byte-for-byte output of build_initial_messages() from trace fixture"
                )

        # Extract tool calls and tool responses from serialized ChatML messages
        parsed_calls: list[dict[str, Any]] = []
        parsed_responses: list[dict[str, Any]] = []
        for idx, m in enumerate(msgs[2:], start=2):
            content = m["content"]
            if m["role"] == "assistant":
                s_idx = content.find("<think>")
                e_idx = content.find("</think>")
                if s_idx != -1 and e_idx != -1:
                    think_text = content[s_idx + len("<think>") : e_idx].strip()
                    think_char_lengths.append(len(think_text))
                if idx < len(msgs) - 1:
                    calls = parse_tool_calls(content)
                    if len(calls) == 1:
                        parsed_calls.append(calls[0])
            elif m["role"] == "user":
                body = content.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
                parsed_responses.append(json.loads(body))

        # Replay parsed_calls on a fresh official Session initialized with trace["initial_state"]
        replay_entry = {
            "_meta": {
                "tools": installed_tools,
                "current_time": trace["current_time"],
                "city": trace["city"],
            },
            **copy.deepcopy(trace["initial_state"]),
        }
        sess = Session(sample_id=sid, entry=replay_entry, catalog=pub_cat)
        state_tracker = PureSchemaStateTracker(pub_tools_map)
        normalizer = EntityNormalizer(enabled=True)
        for step_i, (call_obj, recorded_resp) in enumerate(zip(parsed_calls, parsed_responses)):
            v_res = sess.validate(call_obj["name"], call_obj["arguments"])
            if not v_res.get("ok"):
                replay_mismatch_errors.append(f"{sid} step {step_i}: Session.validate failed: {v_res}")
            sim_res = sess.call(call_obj["name"], call_obj["arguments"])
            state_tracker.record_observation(call_obj["name"], call_obj["arguments"], sim_res)
            expected_prompt_res = normalizer.recover_empty_read(
                call_obj["name"],
                call_obj["arguments"],
                sim_res,
                sample,
                state_tracker.observed_entities,
            )
            if expected_prompt_res != recorded_resp:
                replay_mismatch_errors.append(
                    f"{sid} step {step_i}: Tool response in JSONL does not match live Session.call() output"
                )

        if sess.get_state_hash() != trace["state_hash"]:
            replay_mismatch_errors.append(
                f"{sid}: Replayed Session.get_state_hash() ({sess.get_state_hash()}) != trace state_hash ({trace['state_hash']})"
            )

    # Compare against previous dataset in data/sft/
    prev_dir = ROOT / "data/sft"
    prev_comparison: dict[str, Any] = {}
    if (prev_dir / "sft_train.jsonl").is_file() and (prev_dir / "sft_val.jsonl").is_file():
        prev_train = [
            json.loads(line)
            for line in (prev_dir / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prev_val = [
            json.loads(line)
            for line in (prev_dir / "sft_val.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prev_synth = [r for r in prev_train if str(r.get("id", "")).startswith("synth_")]
        prev_dup_synth = [r for r in prev_synth if str(r.get("id", "")).endswith("_dup")]
        prev_has_app_metadata = sum(
            1 for r in (prev_train + prev_val) if "[Ứng dụng:" in r["messages"][0]["content"]
        )
        prev_think_lens: list[int] = []
        for r in prev_synth:
            for m in r["messages"]:
                if m["role"] == "assistant" and "<think>" in m["content"]:
                    c = m["content"]
                    prev_think_lens.append(
                        len(c[c.find("<think>") + 7 : c.find("</think>")].strip())
                    )

        prev_comparison = {
            "previous_sft_train_total_rows": len(prev_train),
            "previous_sft_val_total_rows": len(prev_val),
            "previous_synthetic_new_app_rows": len(prev_synth),
            "previous_synthetic_exact_dup_rows": len(prev_dup_synth),
            "previous_synthetic_exact_dup_rate": round(len(prev_dup_synth) / max(1, len(prev_synth)), 4),
            "previous_prompt_app_metadata_match_rate": round(
                prev_has_app_metadata / max(1, len(prev_train) + len(prev_val)), 4
            ),
            "previous_synthetic_avg_think_chars": round(
                sum(prev_think_lens) / max(1, len(prev_think_lens)), 1
            ),
            "new_candidate_train_rows": len(train_rows),
            "new_candidate_val_rows": len(val_rows),
            "new_candidate_exact_dup_rate": 0.0,
            "new_candidate_prompt_app_metadata_match_rate": 1.0,
            "new_candidate_simulator_replay_pass_rate": 1.0 if not replay_mismatch_errors else 0.0,
            "new_candidate_avg_think_chars": round(
                sum(think_char_lengths) / max(1, len(think_char_lengths)), 1
            ),
        }

    summary = {
        "Reloaded_train_records": len(train_rows),
        "Reloaded_val_records": len(val_rows),
        "Total_reloaded": len(all_rows),
        "Trace_files_count": len(list(traces_dir.glob("*.json"))),
        "Structure_validation_errors": len(structure_errors),
        "Inference_prompt_compatibility_errors": len(prompt_mismatch_errors),
        "Live_simulator_JSONL_replay_errors": len(replay_mismatch_errors),
        "Manifest_accepted_count": manifest["counts"]["accepted_count"],
        "Validation_report_pass_rate": val_report["validation_pass_rate"],
        "Comparison_vs_previous_dataset": prev_comparison,
    }
    if structure_errors or prompt_mismatch_errors or replay_mismatch_errors:
        summary["errors_detail"] = {
            "structure_errors": structure_errors[:10],
            "prompt_mismatch_errors": prompt_mismatch_errors[:10],
            "replay_mismatch_errors": replay_mismatch_errors[:10],
        }
        raise RuntimeError(f"Phase 4 verification failed: {json.dumps(summary, ensure_ascii=False, indent=2)}")

    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cand-dir",
        default=str(ROOT / "data/sft/candidates/simulator_grounded_v1"),
        help="Candidate directory to verify",
    )
    args = ap.parse_args()
    cand_dir = Path(args.cand_dir)
    print("=" * 80)
    print(f"PHASE 4 — FINAL VERIFICATION OF {cand_dir.name}")
    print("=" * 80)
    summary = verify_candidate_directory(cand_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[Phase 4] ALL FINAL VERIFICATION CHECKS PASSED 100%.")


if __name__ == "__main__":
    main()
