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
    ap.add_argument("--split", default="public_test", choices=["public_test", "train"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", default="submission.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="Tên hoặc đường dẫn model LLM (< 5B)")
    ap.add_argument("--lora-path", default=None, help="Đường dẫn tới LoRA adapter (nếu đã fine-tune)")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    ap.add_argument("--asr-model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    ap.add_argument("--preprocessor", default="a", choices=["a", "b"], help="Tiền xử lý ngôn ngữ: a (Fuzzy Context Aligner, 0-param) hoặc b (Neural Restorer < 0.5B)")
    ap.add_argument("--use-gt-asr", action="store_true", help="Dùng audio_text có sẵn (chỉ dùng khi test nhanh LLM trên tập train)")
    ap.add_argument("--ids-file", default=None, help="File JSON chứa danh sách ID cần chạy (VD: data/sft/val_ids.json)")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu chạy thử (0 = chạy toàn bộ)")
    ap.add_argument("--max-steps", type=int, default=8, help="Số bước ReAct tối đa mỗi mẫu")
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

    print(f"[Pipeline] Split={args.split} | Số mẫu={len(samples)} | LLM={args.model} | LoRA={args.lora_path} | Preprocessor={args.preprocessor.upper()}")

    transcriber = Transcriber(
        model_name=args.asr_model,
        cache_path=root / "cache" / f"asr_{args.split}.json",
        use_gt_audio_text=(args.use_gt_asr and args.split == "train"),
    )
    llm = LLMBackend(
        model_name_or_path=args.model,
        lora_path=args.lora_path,
        backend=args.backend,
    )

    preprocessor = (
        ContextLanguagePreprocessorB(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        if args.preprocessor == "b"
        else ContextLanguagePreprocessorA()
    )
    router = SemanticToolRouter(top_k_recommended=4)

    agent = LifeMateAgent(
        llm=llm,
        catalog_tools_map=catalog_tools_map,
        max_steps=args.max_steps,
        preprocessor=preprocessor,
        router=router,
    )

    t0 = time.perf_counter()
    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(samples, 1):
            s_t0 = time.perf_counter()
            env = ToolEnv(args.split, item["id"], root=root)
            final_text, hist_transcripts = transcriber.transcribe_sample(item, split_dir)
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
                print(
                    f"[{idx}/{len(samples)}] {item['id']} ({elapsed:.2f}s) | ASR: {final_text[:50]!r} -> {len(tool_calls)} calls"
                )

    transcriber.save_cache()
    total_time = time.perf_counter() - t0
    print(f"\nHoàn tất {len(samples)} mẫu trong {total_time:.1f}s (TB: {total_time / max(1, len(samples)):.2f}s/mẫu) -> {args.output}")

    evaluate_predictions(args.output, root=root, split=args.split)


if __name__ == "__main__":
    main()
