"""End-to-end deterministic inference pipeline for ZAC 2026 LifeMate (ASR + Preprocessor + Router + Qwen3-4B Agent + Guardrail)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from evaluate import evaluate_predictions
from src.agent import LifeMateAgent, LLMBackend, fix_all_seeds
from src.asr import Transcriber
from src.preprocessor_a import ContextLanguagePreprocessorA
from src.preprocessor_b import ContextLanguagePreprocessorB
from src.router import SemanticToolRouter


def main():
    ap = argparse.ArgumentParser(description="Chạy pipeline ZAC 2026 LifeMate tạo file submission.jsonl")
    ap.add_argument("--split", default="public_test", help="Tên split dữ liệu (public_test, train, private_test...)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", default="submission.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="Tên hoặc đường dẫn model LLM (< 5B)")
    ap.add_argument("--lora-path", default=None, help="Đường dẫn tới LoRA adapter (nếu đã fine-tune)")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    ap.add_argument("--asr-model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--asr-lora-path", default=None, help="Đường dẫn tới Whisper LoRA adapter")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.68, help="Tỷ lệ VRAM cấp cho vLLM (mặc định 0.68)")
    ap.add_argument(
        "--preprocessor",
        default="safe",
        choices=["a", "b", "safe"],
        help="Tiền xử lý ngôn ngữ: safe (SafeContextPreprocessor an toàn chống nuốt từ, mặc định), a (Fuzzy Aligner), b (Neural Restorer)",
    )
    ap.add_argument(
        "--use-asr-fallback",
        action="store_true",
        help="Tùy chọn: Bật cơ chế dự phòng Qwen3-ASR-0.6B khi Whisper có độ không chắc chắn cao ở các slot quan trọng",
    )
    ap.add_argument("--asr-cache-path", default=None, help="Đường dẫn trực tiếp tới file ASR cache JSON")
    ap.add_argument("--ignore-asr-cache", action="store_true", help="Bỏ qua cache ASR, ép transcriber chạy lại toàn bộ audio với model/LoRA mới")
    ap.add_argument("--use-gt-asr", action="store_true", help="Dùng audio_text có sẵn (chỉ dùng khi test nhanh LLM trên tập train)")
    ap.add_argument("--ids-file", default=None, help="File JSON chứa danh sách ID cần chạy (VD: data/sft/val_ids.json)")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu chạy thử (0 = chạy toàn bộ)")
    ap.add_argument("--max-steps", type=int, default=8, help="Số bước ReAct tối đa mỗi mẫu")

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
    from zaic_tools import ToolEnv  # noqa: E402
    from zaic_tools.validate import catalog_tools  # noqa: E402

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
        f"[Pipeline] Split={args.split} | Số mẫu={len(samples)} | "
        f"LLM={args.model} (LoRA: {args.lora_path}) | "
        f"ASR={args.asr_model} (LoRA: {args.asr_lora_path}) | "
        f"Preprocessor={args.preprocessor.upper()} | "
        f"FuzzyNormalizer={not args.no_fuzzy_normalizer} | "
        f"HybridEngine={not args.no_hybrid_engine} | "
        f"ActionTriage={not args.no_action_triage}"
    )

    if args.ignore_asr_cache:
        asr_cache_p = None
    elif args.asr_cache_path:
        asr_cache_p = Path(args.asr_cache_path)
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

    # 1. GIAI DOAN 1: Nhan dang truoc toan bo audio va luu cache an toan
    print(f"\n[GIAI ĐOẠN 1/2] Chuẩn bị nhận dạng ASR cho {len(samples)} mẫu...")
    transcriber.pretranscribe_samples(samples, split_dir)

    # 2. GIAI DOAN 2: Giai phong hoan toan Whisper khoi GPU VRAM truoc khi khoi dong vLLM
    transcriber.unload_model()

    # 3. GIAI DOAN 3: Khoi dong LLM engine (vLLM so huu 100% GPU, khong con xung dot cuDNN)
    print(f"\n[GIAI ĐOẠN 2/2] Khởi động LLM Engine: {args.model}...")
    llm = LLMBackend(
        model_name_or_path=args.model,
        lora_path=args.lora_path,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.preprocessor == "b":
        preprocessor = ContextLanguagePreprocessorB(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    elif args.preprocessor == "safe" or args.use_asr_fallback:
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

    agent = LifeMateAgent(
        llm=llm,
        catalog_tools_map=catalog_tools_map,
        max_steps=args.max_steps,
        preprocessor=preprocessor,
        router=router,
        catalog_apps_map=catalog_apps_map,
        enable_fuzzy_normalizer=not args.no_fuzzy_normalizer,
        enable_hybrid_engine=not args.no_hybrid_engine,
        enable_action_triage=not args.no_action_triage,
    )

    t0 = time.perf_counter()
    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(samples, 1):
            s_t0 = time.perf_counter()
            env = ToolEnv(args.split, item["id"], root=root)
            final_text, hist_transcripts = transcriber.transcribe_sample(item, split_dir)
            if item.get("history") and hist_transcripts:
                for turn in item["history"]:
                    if not turn.get("audio_text"):
                        aud = turn.get("audio", "")
                        if aud in hist_transcripts:
                            turn["audio_text"] = hist_transcripts[aud]
            tool_calls, st_hash = agent.run_sample(
                env=env,
                sample=item,
                final_user_text=final_text,
                history_transcripts=hist_transcripts,
            )
            row_out = {
                "id": item["id"],
                "tool_calls": tool_calls,
                "state_hash": st_hash,
            }
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            fout.flush()
            elapsed = time.perf_counter() - s_t0
            if idx % 10 == 0 or idx == len(samples) or idx <= 3:
                cl_text = agent.preprocessor.preprocess(final_text, item)
                txt_info = f"ASR: {final_text[:45]!r}" if cl_text == final_text else f"ASR: {final_text[:30]!r} -> Prep: {cl_text[:35]!r}"
                print(
                    f"[{idx}/{len(samples)}] {item['id']} ({elapsed:.2f}s) | {txt_info} -> {len(tool_calls)} calls"
                )

    transcriber.save_cache()
    total_time = time.perf_counter() - t0
    print(f"\nHoàn tất {len(samples)} mẫu trong {total_time:.1f}s (TB: {total_time / max(1, len(samples)):.2f}s/mẫu) -> {args.output}")

    evaluate_predictions(args.output, root=root, split=args.split)


if __name__ == "__main__":
    main()
