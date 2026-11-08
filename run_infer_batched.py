"""High-Throughput Batched ReAct Inference Pipeline for ZAC 2026 LifeMate.
Runs concurrent multi-turn ReAct loops across active sample batches via vLLM continuous batching.
Features switchable Solutions 2, 3, and 4 (Fuzzy Normalizer, Hybrid Engine, Action Triage).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from evaluate import evaluate_predictions
from src.action_triage import ActionTriageBarrier
from src.agent import LifeMateAgent, LLMBackend, fix_all_seeds
from src.asr import Transcriber
from src.entity_normalizer import EntityNormalizer
from src.guardrail import PureSchemaStateTracker, normalize_arguments, parse_tool_calls
from src.hybrid_engine import HybridProgramAidedEngine
from src.preprocessor_a import ContextLanguagePreprocessorA
from src.prompt_builder import build_initial_messages
from src.router import SemanticToolRouter


class SampleReActState:
    """Encapsulates the live state of a single sample in the batched ReAct loop."""

    def __init__(
        self,
        index: int,
        sample: dict[str, Any],
        env: Any,
        cleaned_user_text: str,
        cleaned_hist_transcripts: dict[str, str],
        catalog_tools_map: dict[str, dict[str, Any]],
        catalog_apps_map: dict[str, dict[str, Any]],
        router: SemanticToolRouter,
        max_steps: int = 8,
        enable_fuzzy_normalizer: bool = True,
        enable_hybrid_engine: bool = True,
        enable_action_triage: bool = True,
    ):
        self.index = index
        self.sample = sample
        self.sample_id = sample["id"]
        self.env = env
        self.cleaned_user_text = cleaned_user_text
        self.max_steps = max_steps
        self.step = 0
        self.finished = False
        self.guardrail_retries = 0
        self.seen_calls: set[str] = set()
        self.suppress_sms = False
        self.tool_calls: list[dict[str, Any]] = []
        self.state_hash: str = ""
        self.start_time = time.perf_counter()
        self.elapsed = 0.0

        self.enable_fuzzy_normalizer = enable_fuzzy_normalizer
        self.enable_hybrid_engine = enable_hybrid_engine
        self.enable_action_triage = enable_action_triage

        self.normalizer = EntityNormalizer(enabled=enable_fuzzy_normalizer)
        self.hybrid_engine = HybridProgramAidedEngine(enabled=enable_hybrid_engine)
        self.action_triage = ActionTriageBarrier(enabled=enable_action_triage)

        cur_time = sample.get("context", {}).get("current_time", "")
        self.tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"

        # Check early exit: empty query, out-of-scope, or complete cancellation
        if not cleaned_user_text:
            self._finish()
            return

        if self.enable_action_triage:
            is_oos, _ = self.action_triage.check_out_of_scope(sample, cleaned_user_text)
            if is_oos:
                self._finish()
                return

        if self.enable_hybrid_engine and self.hybrid_engine.is_complete_cancellation(cleaned_user_text):
            self._finish()
            return

        self.state_tracker = PureSchemaStateTracker(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=catalog_tools_map,
        )

        self.messages = build_initial_messages(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=catalog_tools_map,
            history_transcripts=cleaned_hist_transcripts,
            router=router,
            catalog_apps_map=catalog_apps_map,
        )

    def _finish(self) -> None:
        self.finished = True
        self.tool_calls = self.env.get_tool_calls()
        self.state_hash = self.env.get_state_hash()
        self.elapsed = time.perf_counter() - self.start_time
        self.messages = []


def main():
    ap = argparse.ArgumentParser(description="Chạy pipeline suy luận đa luồng theo lô (Batched ReAct) cho ZAC 2026")
    ap.add_argument("--split", default="public_test", help="Tên split dữ liệu (public_test, train, private_test...)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", default="submissions/submission_batched.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers", "auto"])
    ap.add_argument("--batch-size", type=int, default=16, help="Kích thước lô mẫu chạy đồng thời (mặc định 16)")
    ap.add_argument("--asr-model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--asr-lora-path", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument(
        "--preprocessor",
        choices=["a", "b", "safe"],
        default="safe",
        help="Tiền xử lý ngôn ngữ: safe (SafeContextPreprocessor mặc định), a (Fuzzy Context Aligner), hoặc b (Neural Restorer)",
    )
    ap.add_argument("--ignore-asr-cache", action="store_true", help="Bỏ qua cache ASR, chạy nhận dạng lại")
    ap.add_argument("--use-gt-asr", action="store_true")

    # Switches for Solutions 2, 3, 4
    ap.add_argument("--no-fuzzy-normalizer", action="store_true", help="[Solution 2] Tắt EntityNormalizer & Fuzzy Read Recovery")
    ap.add_argument("--no-hybrid-engine", action="store_true", help="[Solution 3] Tắt HybridProgramAidedEngine (Cancellation, Weather, Math)")
    ap.add_argument("--no-action-triage", action="store_true", help="[Solution 4] Tắt ActionTriageBarrier (Out-of-Scope & ask_missing_info guard)")

    args = ap.parse_args()

    fix_all_seeds(42)
    root = Path(args.root)
    split_dir = root / args.split

    if str(split_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(split_dir.resolve()))
    from zaic_tools import ToolEnv
    from zaic_tools.validate import catalog_tools

    catalog_path = split_dir / f"{args.split}_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_tools_map = catalog_tools(catalog)
    catalog_apps_map = {app["id"]: app for app in catalog.get("apps", [])}

    samples = [
        json.loads(line)
        for line in (split_dir / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.ids_file:
        target_ids = set(json.loads(Path(args.ids_file).read_text(encoding="utf-8")))
        samples = [s for s in samples if s["id"] in target_ids]
    if args.limit > 0:
        samples = samples[: args.limit]

    print(
        f"[Batched Pipeline] Split={args.split} | Số mẫu={len(samples)} | Batch Size={args.batch_size} | "
        f"LLM={args.model} | LoRA={args.lora_path} | "
        f"FuzzyNormalizer={not args.no_fuzzy_normalizer} | "
        f"HybridEngine={not args.no_hybrid_engine} | "
        f"ActionTriage={not args.no_action_triage}"
    )

    if args.ignore_asr_cache:
        asr_cache_p = None
    elif args.asr_lora_path:
        lora_tag = Path(args.asr_lora_path).name or "lora"
        asr_cache_p = root / "cache" / f"asr_{args.split}_{lora_tag}.json"
    elif "turbo" in args.asr_model or "ct2" in args.asr_model:
        asr_cache_p = root / "cache" / f"asr_{args.split}.json"
    else:
        model_tag = Path(args.asr_model).name or "custom"
        asr_cache_p = root / "cache" / f"asr_{args.split}_{model_tag}.json"

    transcriber = Transcriber(
        model_name=args.asr_model,
        lora_path=args.asr_lora_path,
        cache_path=asr_cache_p,
        use_gt_audio_text=(args.use_gt_asr and args.split == "train"),
    )
    print(f"\n[GIAI ĐOẠN 1/2] Kiểm tra ASR Cache cho {len(samples)} mẫu...")
    transcriber.pretranscribe_samples(samples, split_dir)
    transcriber.unload_model()

    print(f"\n[GIAI ĐOẠN 2/2] Khởi động LLM Engine: {args.model}...")
    llm = LLMBackend(
        model_name_or_path=args.model,
        lora_path=args.lora_path,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.preprocessor == "b":
        from src.preprocessor_b import ContextLanguagePreprocessorB

        preprocessor = ContextLanguagePreprocessorB(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    elif args.preprocessor == "safe":
        from src.asr_fallback import SafeContextPreprocessor

        _safe_prep = SafeContextPreprocessor(sim_threshold=0.80)

        class _SafePreprocessorAdapter:
            def preprocess(self, raw_asr_text: str, sample: dict) -> str:
                cleaned, _ = _safe_prep.preprocess_with_evidence(raw_asr_text, sample)
                return cleaned

        preprocessor = _SafePreprocessorAdapter()
    else:
        preprocessor = ContextLanguagePreprocessorA()
    router = SemanticToolRouter(top_k_recommended=4)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    completed_results: dict[int, dict[str, Any]] = {}
    completed_count = 0

    next_sample_idx = 0
    active_states: list[SampleReActState] = []

    def _fill_active_pool():
        nonlocal next_sample_idx, completed_count
        while len(active_states) < args.batch_size and next_sample_idx < len(samples):
            idx = next_sample_idx + 1
            item = samples[next_sample_idx]
            next_sample_idx += 1

            env = ToolEnv(args.split, item["id"], root=root)
            raw_text, raw_hist = transcriber.transcribe_sample(item, split_dir)
            cl_text = preprocessor.preprocess(raw_text, item)
            cl_hist = {k: preprocessor.preprocess(v, item) for k, v in raw_hist.items()}

            if item.get("history") and cl_hist:
                for turn in item["history"]:
                    if not turn.get("audio_text"):
                        aud = turn.get("audio", "")
                        if aud in cl_hist:
                            turn["audio_text"] = cl_hist[aud]

            state = SampleReActState(
                index=idx,
                sample=item,
                env=env,
                cleaned_user_text=cl_text,
                cleaned_hist_transcripts=cl_hist,
                catalog_tools_map=catalog_tools_map,
                catalog_apps_map=catalog_apps_map,
                router=router,
                max_steps=args.max_steps,
                enable_fuzzy_normalizer=not args.no_fuzzy_normalizer,
                enable_hybrid_engine=not args.no_hybrid_engine,
                enable_action_triage=not args.no_action_triage,
            )

            if state.finished:
                completed_results[idx] = {
                    "id": state.sample_id,
                    "tool_calls": state.tool_calls,
                    "state_hash": state.state_hash,
                }
                completed_count += 1
            else:
                active_states.append(state)

    _fill_active_pool()

    while active_states:
        pending_states = [s for s in active_states if not s.finished]
        if not pending_states:
            break

        prompts_to_generate = [s.messages for s in pending_states]
        replies = llm.generate_batch(prompts_to_generate)

        for state, reply in zip(pending_states, replies):
            state.step += 1
            raw_calls = parse_tool_calls(reply)

            if not raw_calls:
                state.finished = True
                state.tool_calls = state.env.get_tool_calls()
                state.state_hash = state.env.get_state_hash()
                state.elapsed = time.perf_counter() - state.start_time
                completed_results[state.index] = {
                    "id": state.sample_id,
                    "tool_calls": state.tool_calls,
                    "state_hash": state.state_hash,
                }
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(samples):
                    print(
                        f"[{completed_count}/{len(samples)}] {state.sample_id} ({state.elapsed:.2f}s, {state.step} bước) -> {len(state.tool_calls)} calls"
                    )
                continue

            call = raw_calls[0]
            t_name = call["name"]
            t_args = normalize_arguments(
                t_name,
                call.get("arguments") or {},
                catalog_tools_map,
                tz_offset=state.tz_offset,
            )

            # Solution 3: DateTime Auto-Repair
            if state.enable_hybrid_engine:
                for k in ("time", "start", "end", "schedule_time"):
                    if k in t_args and isinstance(t_args[k], str):
                        t_args[k] = state.hybrid_engine.normalize_iso_datetime(t_args[k], state.tz_offset)

            # Solution 4: Guard sys.ask_missing_info
            if t_name == "sys.ask_missing_info" and state.enable_action_triage:
                allowed, triage_reason = state.action_triage.guard_ask_missing_info(
                    t_args, state.sample, state.seen_calls, catalog_tools_map, state.state_tracker.observed_entities
                )
                if not allowed:
                    state.guardrail_retries += 1
                    state.messages.append({"role": "assistant", "content": reply})
                    state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"<tool_response>\n"
                                f"{json.dumps({'ok': False, 'validation_error': triage_reason, 'hint': 'Hãy chọn đối tượng phù hợp nhất từ kết quả tra cứu để thực thi công cụ tiếp theo.'}, ensure_ascii=False)}\n"
                                f"</tool_response>"
                            ),
                        }
                    )
                    if state.guardrail_retries >= 3 or state.step >= state.max_steps:
                        state._finish()
                        completed_results[state.index] = {
                            "id": state.sample_id,
                            "tool_calls": state.tool_calls,
                            "state_hash": state.state_hash,
                        }
                        completed_count += 1
                    continue

            # Solution 3: Stock Math & Bill Payment adjustments
            if state.enable_hybrid_engine:
                if t_name == "stocks.place_order":
                    t_args = state.hybrid_engine.adjust_stock_order(
                        t_args, state.cleaned_user_text, state.state_tracker.observed_entities
                    )
                elif t_name == "bill_payment.pay":
                    t_args, supp_sms = state.hybrid_engine.adjust_bill_payment(
                        t_args, state.cleaned_user_text, state.state_tracker.observed_entities, state.state_tracker.balances
                    )
                    if t_args is None:
                        state.suppress_sms = True
                        continue
                    if supp_sms:
                        state.suppress_sms = True

                if t_name == "sms.send" and state.suppress_sms:
                    continue

                if t_name == "parking.reserve" and any("ride_hailing.book" in s and "bike" in s for s in state.seen_calls):
                    continue

            # Solution 2: Entity normalization
            if state.enable_fuzzy_normalizer:
                t_args = state.normalizer.resolve_fuzzy_arguments(
                    t_name, t_args, state.state_tracker.observed_entities, state.sample
                )

            state.messages.append({"role": "assistant", "content": reply})

            call_sig = f"{t_name}:{json.dumps(t_args, sort_keys=True, ensure_ascii=False)}"
            if call_sig in state.seen_calls:
                state.guardrail_retries += 1
                state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response>\n"
                            f"{json.dumps({'ok': False, 'error': 'Bạn vừa gọi đúng lệnh này rồi, không gọi lặp lại.'}, ensure_ascii=False)}\n"
                            f"</tool_response>"
                        ),
                    }
                )
                if state.guardrail_retries >= 3 or state.step >= state.max_steps:
                    state._finish()
                    completed_results[state.index] = {
                        "id": state.sample_id,
                        "tool_calls": state.tool_calls,
                        "state_hash": state.state_hash,
                    }
                    completed_count += 1
                continue

            # Zero-cost schema validation
            val_res = state.env.validate(t_name, t_args)
            if not val_res.get("ok"):
                state.guardrail_retries += 1
                err_msg = val_res.get("error", "Sai schema")
                state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response>\n"
                            f"{json.dumps({'ok': False, 'validation_error': err_msg}, ensure_ascii=False)}\n"
                            f"</tool_response>"
                        ),
                    }
                )
                if state.guardrail_retries >= 3 or state.step >= state.max_steps:
                    state._finish()
                    completed_results[state.index] = {
                        "id": state.sample_id,
                        "tool_calls": state.tool_calls,
                        "state_hash": state.state_hash,
                    }
                    completed_count += 1
                continue

            # Pure schema state precheck
            pre_err = state.state_tracker.check_pre_execution(t_name, t_args)
            if pre_err is not None:
                state.guardrail_retries += 1
                state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response>\n"
                            f"{json.dumps({'ok': False, 'precheck_error': pre_err}, ensure_ascii=False)}\n"
                            f"</tool_response>"
                        ),
                    }
                )
                if state.guardrail_retries >= 3 or state.step >= state.max_steps:
                    state._finish()
                    completed_results[state.index] = {
                        "id": state.sample_id,
                        "tool_calls": state.tool_calls,
                        "state_hash": state.state_hash,
                    }
                    completed_count += 1
                continue

            # Execute real simulator call
            res = state.env.call(t_name, t_args)
            state.seen_calls.add(call_sig)
            state.state_tracker.record_observation(t_name, t_args, res)

            # Solution 3: Weather condition check
            if state.enable_hybrid_engine:
                if t_name in ("weather.get_forecast", "weather.get_current"):
                    if not state.hybrid_engine.check_weather_condition(state.cleaned_user_text, res):
                        state._finish()
                        completed_results[state.index] = {
                            "id": state.sample_id,
                            "tool_calls": state.tool_calls,
                            "state_hash": state.state_hash,
                        }
                        completed_count += 1
                        continue

                if t_name == "smart_home.fan_set":
                    extra_fans = state.hybrid_engine.expand_smart_home_fans(
                        state.env, state.cleaned_user_text, state.state_tracker.observed_entities, t_args
                    )
                    for ef in extra_fans:
                        state.seen_calls.add(f"{ef['name']}:{json.dumps(ef['arguments'], sort_keys=True, ensure_ascii=False)}")
                        ef_res = state.env.call(ef["name"], ef["arguments"])
                        state.state_tracker.record_observation(ef["name"], ef["arguments"], ef_res)

            # Solution 2: Fuzzy Read Recovery
            if state.enable_fuzzy_normalizer:
                prompt_res = state.normalizer.recover_empty_read(
                    t_name, t_args, res, state.sample, state.state_tracker.observed_entities
                )
            else:
                prompt_res = res

            state.messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{json.dumps(prompt_res, ensure_ascii=False)}\n</tool_response>",
                }
            )

            if t_name == "sys.ask_missing_info" or state.step >= state.max_steps:
                state._finish()
                completed_results[state.index] = {
                    "id": state.sample_id,
                    "tool_calls": state.tool_calls,
                    "state_hash": state.state_hash,
                }
                completed_count += 1

        active_states = [s for s in active_states if not s.finished]
        _fill_active_pool()

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx in range(1, len(samples) + 1):
            if idx in completed_results:
                fout.write(json.dumps(completed_results[idx], ensure_ascii=False) + "\n")

    total_time = time.perf_counter() - t0
    print(
        f"\n[Batched ReAct] Hoàn tất {len(samples)} mẫu trong {total_time:.1f}s "
        f"(TB: {total_time / max(1, len(samples)):.2f}s/mẫu) -> {out_path}"
    )

    evaluate_predictions(str(out_path), root=root, split=args.split)


if __name__ == "__main__":
    main()
