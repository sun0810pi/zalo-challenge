"""Benchmark & side-by-side comparison of Preprocessor A (Fuzzy Context Aligner)
vs Preprocessor B (Neural Model) on real public_test Whisper outputs.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.preprocessor_a import ContextLanguagePreprocessorA
from src.preprocessor_b import ContextLanguagePreprocessorB


def main():
    parser = argparse.ArgumentParser(description="So sánh Preprocessor A vs Preprocessor B")
    parser.add_argument("--limit", type=int, default=15, help="Số mẫu thử nghiệm")
    parser.add_argument("--model-b", default="Qwen/Qwen2.5-0.5B-Instruct", help="Tên model neural cho Option B")
    args = parser.parse_args()

    whisper_cache_path = ROOT / "cache/asr_public_test.json"
    if not whisper_cache_path.is_file():
        print(f"Không tìm thấy {whisper_cache_path}!")
        return

    whisper_cache = json.loads(whisper_cache_path.read_text(encoding="utf-8"))

    # Load Gemini ground truth ASR
    gemini_map: dict[str, str] = {}
    for gf in glob.glob(str(ROOT / "cache/gemini_asr/*.json")):
        gemini_map.update(json.loads(Path(gf).read_text(encoding="utf-8")))

    # Load public_test samples
    samples_map = {}
    pub_test_path = ROOT / "public_test/public_test.jsonl"
    for line in pub_test_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            samples_map[row["id"]] = row

    prep_a = ContextLanguagePreprocessorA()
    prep_b = ContextLanguagePreprocessorB(model_name=args.model_b)

    print("=" * 80)
    print("SO SÁNH NGÔN NGỮ TIỀN XỬ LÝ (PREPROCESSOR A VS PREPROCESSOR B)")
    print("=" * 80)

    count = 0
    for rel_path, gt_clean in gemini_map.items():
        raw_whisper = whisper_cache.get(rel_path, "")
        if not raw_whisper or raw_whisper.strip().lower() == gt_clean.strip().lower():
            continue

        sid = rel_path.split("/")[-1].replace(".wav", "")
        sample = samples_map.get(sid, {})
        if not sample:
            continue

        count += 1
        out_a = prep_a.preprocess(raw_whisper, sample)
        out_b = prep_b.preprocess(raw_whisper, sample)

        print(f"\n[Mẫu #{count}] ID: {sid} ({rel_path})")
        print(f"  🔴 Raw Whisper    : {raw_whisper}")
        print(f"  🟢 Preprocessor A : {out_a}")
        print(f"  🔵 Preprocessor B : {out_b}")
        print(f"  ⭐ Ground Truth   : {gt_clean}")

        if count >= args.limit:
            break

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
