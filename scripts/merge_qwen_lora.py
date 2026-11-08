"""Merge Qwen3-4B LoRA adapter into full standalone model weights for offline inference / Docker submission."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Qwen3-4B base model")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B", help="Base model ID or path")
    parser.add_argument("--lora-path", required=True, help="Path to trained LoRA checkpoint")
    parser.add_argument("--output-dir", required=True, help="Path to save merged standalone model")
    parser.add_argument("--device", default="auto", help="Device to use for merging")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}...")
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print("Done! Standalone model is ready for offline Docker deployment.")


if __name__ == "__main__":
    main()
