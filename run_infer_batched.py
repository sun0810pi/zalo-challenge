"""High-Throughput Batched ReAct Inference Pipeline for ZAC 2026 LifeMate.
Runs concurrent multi-turn ReAct loops across active sample batches via vLLM continuous batching.
Features switchable Solutions 2, 3, and 4 (Fuzzy Normalizer, Hybrid Engine, Action Triage).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from evaluate import evaluate_predictions
from src.agent import (
    LLMBackend,
    ReActSessionState,
    build_preprocessor,
    fix_all_seeds,
    step_react_state,
)
from src.asr import Transcriber
from src.router import SemanticToolRouter

# Backwards-compatible alias for external imports
SampleReActState = ReActSessionState


def _build_runtime_preprocessor(mode: str, use_asr_fallback: bool) -> tuple[Any, Any | None]:
    """Build text preprocessor and optional UncertaintyAwareASRFallback adapter."""
    preprocessor = build_preprocessor(mode=mode, sim_threshold=0.80)
    fallback_engine = None
    if use_asr_fallback:
        try:
            from src.asr_fallback import UncertaintyAwareASRFallback

            fallback_engine = UncertaintyAwareASRFallback(
                sim_threshold=0.80,
                enable_qwen3_fallback=True,
            )
        except Exception as exc:
            print(f"[CẢNH BÁO] Không khởi tạo được UncertaintyAwareASRFallback ({exc}); tiếp tục với {mode}.")
    return preprocessor, fallback_engine


def _preprocess_sample_audio_texts(
    raw_text: str,
    raw_hist: dict[str, str],
    item: dict[str, Any],
    split_dir: Path,
    preprocessor: Any,
    fallback_engine: Any | None,
) -> tuple[str, dict[str, str]]:
    if fallback_engine is not None:
        audio_rel = item.get("audio")
        audio_abs = (split_dir / audio_rel) if audio_rel else None
        cl_text, _ = fallback_engine.process_utterance(
            raw_text,
            item,
            audio_path=audio_abs if (audio_abs and audio_abs.is_file()) else None,
        )
    else:
        cl_text = preprocessor.preprocess(raw_text, item)

    cl_hist: dict[str, str] = {}
    for k, v in raw_hist.items():
        if fallback_engine is not None:
            hist_abs = split_dir / k
            h_clean, _ = fallback_engine.process_utterance(
                v,
                item,
                audio_path=hist_abs if hist_abs.is_file() else None,
            )
            cl_hist[k] = h_clean
        else:
            cl_hist[k] = preprocessor.preprocess(v, item)
    return cl_text, cl_hist


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
        "--max-apps",
        type=int,
        default=None,
        help="Giới hạn số app tối đa trong ALLOWED TOOLS (mặc định None = giữ toàn bộ app khớp lúc train; đặt 4 hoặc 6 nếu GPU VRAM thấp)",
    )
    ap.add_argument(
        "--preprocessor",
        choices=["a", "b", "safe"],
        default="safe",
        help="Tiền xử lý ngôn ngữ: safe (SafeContextPreprocessor mặc định), a (Fuzzy Context Aligner), hoặc b (Neural Restorer)",
    )
    ap.add_argument(
        "--use-asr-fallback",
        action="store_true",
        help="Bật UncertaintyAwareASRFallback (Qwen3-ASR-0.6B) cho các slot nghi ngờ",
    )
    ap.add_argument("--ignore-asr-cache", action="store_true", help="Bỏ qua cache ASR, ép nhận dạng lại")
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
        f"LLM={args.model} | LoRA={args.lora_path} | Preprocessor={args.preprocessor.upper()} | "
        f"ASRFallback={args.use_asr_fallback} | "
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

    preprocessor, fallback_engine = _build_runtime_preprocessor(args.preprocessor, args.use_asr_fallback)
    router = SemanticToolRouter(top_k_recommended=4)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    completed_results: dict[int, dict[str, Any]] = {}
    completed_count = 0

    next_sample_idx = 0
    active_states: list[ReActSessionState] = []

    def _record_completed(st: ReActSessionState) -> None:
        nonlocal completed_count
        completed_results[st.index] = {
            "id": st.sample_id,
            "tool_calls": st.tool_calls,
            "state_hash": st.state_hash,
        }
        completed_count += 1
        if completed_count % 10 == 0 or completed_count == len(samples):
            print(
                f"[{completed_count}/{len(samples)}] {st.sample_id} ({st.elapsed:.2f}s, {st.step} bước) -> {len(st.tool_calls)} calls"
            )

    def _fill_active_pool():
        nonlocal next_sample_idx
        while len(active_states) < args.batch_size and next_sample_idx < len(samples):
            idx = next_sample_idx + 1
            item = samples[next_sample_idx]
            next_sample_idx += 1

            env = ToolEnv(args.split, item["id"], root=root)
            raw_text, raw_hist = transcriber.transcribe_sample(item, split_dir)
            cl_text, cl_hist = _preprocess_sample_audio_texts(
                raw_text, raw_hist, item, split_dir, preprocessor, fallback_engine
            )

            if item.get("history") and cl_hist:
                for turn in item["history"]:
                    if not turn.get("audio_text"):
                        aud = turn.get("audio", "")
                        if aud in cl_hist:
                            turn["audio_text"] = cl_hist[aud]

            state = ReActSessionState(
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
                max_apps=args.max_apps,
            )

            if state.finished:
                _record_completed(state)
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
            step_react_state(state, reply)
            if state.finished:
                _record_completed(state)

        active_states = [s for s in active_states if not s.finished]
        _fill_active_pool()

    transcriber.save_cache()

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
