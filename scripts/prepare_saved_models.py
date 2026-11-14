"""Prepare and verify offline inference weights inside `saved_models/` before `docker build`.

Usage examples:
  1) Download base/CT2 models directly (if no LoRA was trained):
     python scripts/prepare_saved_models.py

  2) Merge trained Qwen3-4B LoRA + Whisper LoRA into `saved_models/llm` and `saved_models/asr`:
     python scripts/prepare_saved_models.py \
         --llm-lora checkpoints/qwen3-4b-zac-lora-v5-expanded \
         --asr-lora checkpoints/whisper-lora

  3) Verify `saved_models/` is ready for offline Docker build:
     python scripts/prepare_saved_models.py --check-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _has_llm_weights(llm_dir: Path) -> bool:
    if not llm_dir.is_dir() or not (llm_dir / "config.json").is_file():
        return False
    return any(llm_dir.glob("*.safetensors")) or any(llm_dir.glob("*.bin"))


def _has_asr_weights(asr_dir: Path) -> bool:
    if not asr_dir.is_dir():
        return False
    if (asr_dir / "model.bin").is_file():  # faster-whisper / CTranslate2
        return True
    if (asr_dir / "config.json").is_file() and (
        any(asr_dir.glob("*.safetensors")) or any(asr_dir.glob("*.bin"))
    ):
        return True
    return False


def check_saved_models(saved_root: Path) -> bool:
    llm_dir = saved_root / "llm"
    asr_dir = saved_root / "asr"
    ok_llm = _has_llm_weights(llm_dir)
    ok_asr = _has_asr_weights(asr_dir)
    print(f"[Check] {llm_dir}: {'OK' if ok_llm else 'MISSING / INCOMPLETE'}")
    print(f"[Check] {asr_dir}: {'OK' if ok_asr else 'MISSING / INCOMPLETE'}")
    return ok_llm and ok_asr


def _copy_or_download(source: str, dest: Path) -> None:
    src_path = Path(source)
    if src_path.is_dir():
        print(f"[Prepare] Copying local model {src_path} -> {dest}...")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_path, dest)
        return

    from huggingface_hub import snapshot_download

    print(f"[Prepare] Downloading HuggingFace repo '{source}' -> {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=source, local_dir=str(dest))


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare saved_models/llm and saved_models/asr for offline Docker")
    ap.add_argument("--saved-dir", default=str(ROOT / "saved_models"))
    ap.add_argument("--llm-source", default="Qwen/Qwen3-4B", help="Base/merged LLM path or HF repo ID")
    ap.add_argument("--llm-lora", default=None, help="Optional Qwen LoRA checkpoint to merge into saved_models/llm")
    ap.add_argument(
        "--asr-source",
        default="deepdml/faster-whisper-large-v3-turbo-ct2",
        help="Base/merged ASR path or HF repo ID",
    )
    ap.add_argument("--asr-lora", default=None, help="Optional Whisper LoRA checkpoint to merge into saved_models/asr")
    ap.add_argument("--check-only", action="store_true", help="Only verify saved_models/ without downloading/merging")
    args = ap.parse_args()

    saved_root = Path(args.saved_dir)
    if args.check_only:
        ok = check_saved_models(saved_root)
        sys.exit(0 if ok else 1)

    llm_dest = saved_root / "llm"
    asr_dest = saved_root / "asr"

    if args.llm_lora:
        print(f"[Prepare] Merging LLM LoRA '{args.llm_lora}' into '{llm_dest}'...")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "merge_qwen_lora.py"),
                "--base-model",
                args.llm_source,
                "--lora-path",
                args.llm_lora,
                "--output-dir",
                str(llm_dest),
            ],
            check=True,
        )
    elif not _has_llm_weights(llm_dest):
        _copy_or_download(args.llm_source, llm_dest)
    else:
        print(f"[Prepare] {llm_dest} already populated, skipping.")

    if args.asr_lora:
        print(f"[Prepare] Merging ASR LoRA '{args.asr_lora}' into '{asr_dest}'...")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "merge_whisper_lora.py"),
                "--lora-path",
                args.asr_lora,
                "--output-dir",
                str(asr_dest),
            ],
            check=True,
        )
    elif not _has_asr_weights(asr_dest):
        _copy_or_download(args.asr_source, asr_dest)
    else:
        print(f"[Prepare] {asr_dest} already populated, skipping.")

    if not check_saved_models(saved_root):
        raise SystemExit("[ERROR] saved_models verification failed after preparation!")
    print("[OK] saved_models/llm and saved_models/asr are ready for offline Docker build.")


if __name__ == "__main__":
    main()
