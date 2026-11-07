"""Test-Time Compute: In-Memory Best-of-N Simulator Rollout Inference for ZAC 2026 LifeMate.
Generates N candidate trajectories per sample (e.g. N = 3), executes each candidate
on an independent in-memory clone of `zaic_tools.ToolEnv`, evaluates execution health
(0-cost schema validation, absence of runtime errors, penalty on premature ask_missing_info),
and commits the cleanest trajectory to `submission.jsonl`.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate_predictions
from src.agent import LLMBackend, LifeMateAgent, fix_all_seeds
from src.asr import Transcriber
from src.preprocessor_a import ContextLanguagePreprocessorA
from src.router import SemanticToolRouter
from src.speculative_rollout import evaluate_trajectory_quality


def main():
    ap = argparse.ArgumentParser(description="Test-Time Compute: In-Memory Best-of-N Simulator Rollout for ZAC 2026")
    ap.add_argument("--split", default="public_test", choices=["public_test", "train"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", default="submissions/submission_best_of_3.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    ap.add_argument("--num-candidates", type=int, default=3, help="Số lượng nhánh rollout mô phỏng song song (N trong Best-of-N)")
    ap.add_argument("--temperatures", default="0.0,0.6,0.8", help="Danh sách nhiệt độ sinh cho từng ứng viên (cách nhau bởi dấu phẩy)")
    ap.add_argument("--asr-model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--asr-lora-path", default=None)
    ap.add_argument("--preprocessor", default="a", choices=["a", "b", "safe"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--ignore-asr-cache", action="store_true")
    ap.add_argument("--use-gt-asr", action="store_true")
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

    temps = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    if len(temps) < args.num_candidates:
        temps.extend([0.7] * (args.num_candidates - len(temps)))
    temps = temps[: args.num_candidates]

    print(
        f"[Best-of-{args.num_candidates} Simulator Rollout] Split={args.split} | Số mẫu={len(samples)} | "
        f"Candidates={args.num_candidates} (Temps: {temps}) | Model={args.model} | LoRA={args.lora_path}"
    )

    # 1. ASR Phase
    if args.ignore_asr_cache:
        asr_cache_p = None
    elif args.asr_lora_path:
        lora_tag = Path(args.asr_lora_path).name or "lora"
        asr_cache_p = root / "cache" / f"asr_{args.split}_{lora_tag}.json"
    else:
        asr_cache_p = root / "cache" / f"asr_{args.split}.json"

    transcriber = Transcriber(
        model_name=args.asr_model,
        lora_path=args.asr_lora_path,
        cache_path=asr_cache_p,
        use_gt_audio_text=(args.use_gt_asr and args.split == "train"),
    )
    print(f"\n[GIAI ĐOẠN 1/2] Chuẩn bị nhận dạng ASR cho {len(samples)} mẫu...")
    transcriber.pretranscribe_samples(samples, split_dir)
    transcriber.unload_model()

    # 2. LLM Engine Phase
    print(f"\n[GIAI ĐOẠN 2/2] Khởi động LLM Engine: {args.model}...")
    llm = LLMBackend(
        model_name_or_path=args.model,
        lora_path=args.lora_path,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    preprocessor = ContextLanguagePreprocessorA()
    router = SemanticToolRouter(top_k_recommended=4)

    # Create agent instance
    agent = LifeMateAgent(
        llm=llm,
        catalog_tools_map=catalog_tools_map,
        max_steps=args.max_steps,
        preprocessor=preprocessor,
        router=router,
        catalog_apps_map=catalog_apps_map,
        enable_fuzzy_normalizer=True,
        enable_hybrid_engine=True,
        enable_action_triage=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = []

    print(f"\n[Best-of-N Search] Bắt đầu suy luận song song {args.num_candidates} đường mô phỏng mỗi mẫu...\n")

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(samples, 1):
            s_t0 = time.perf_counter()
            final_text, hist_transcripts = transcriber.transcribe_sample(item, split_dir)

            candidate_records = []

            for c_idx in range(args.num_candidates):
                temp = temps[c_idx]
                # Clone fresh ToolEnv in memory for this candidate
                c_env = ToolEnv(args.split, item["id"], root=root)

                # Run sample rollout
                calls, s_hash = agent.run_sample(
                    env=c_env,
                    sample=item,
                    final_user_text=final_text,
                    history_transcripts=hist_transcripts,
                )

                score = evaluate_trajectory_quality(c_env, calls, catalog_tools_map)
                candidate_records.append({
                    "candidate_idx": c_idx + 1,
                    "temp": temp,
                    "calls": calls,
                    "state_hash": s_hash,
                    "score": score,
                })

            # Select candidate with highest quality score
            best_cand = max(candidate_records, key=lambda c: c["score"])
            elapsed = time.perf_counter() - s_t0

            record = {
                "id": item["id"],
                "tool_calls": best_cand["calls"],
                "state_hash": best_cand["state_hash"],
            }
            results.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            scores_str = ", ".join(f"C{c['candidate_idx']}:{c['score']:.0f}" for c in candidate_records)
            print(
                f"[{idx}/{len(samples)}] {item['id']} ({elapsed:.2f}s) -> "
                f"Chọn C{best_cand['candidate_idx']} ({scores_str}) | Calls: {len(best_cand['calls'])}"
            )

    total_time = time.perf_counter() - t0
    print(
        f"\n[Best-of-{args.num_candidates}] Hoàn tất {len(samples)} mẫu trong {total_time:.1f}s "
        f"(TB: {total_time / max(1, len(samples)):.2f}s/mẫu) -> {out_path}"
    )

    if args.split == "train":
        evaluate_predictions(str(out_path), root=root, split=args.split)


if __name__ == "__main__":
    main()
