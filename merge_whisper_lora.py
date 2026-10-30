"""Script gộp (merge) Whisper LoRA adapter vào mô hình gốc và lưu lại thành Standalone Model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def merge_and_save(
    lora_path: str,
    output_dir: str,
    base_model: str | None = None,
):
    lp = Path(lora_path)
    if not (lp / "adapter_config.json").is_file():
        sub_ckpts = sorted(
            lp.glob("checkpoint-*/adapter_config.json"),
            key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else 0,
        )
        if sub_ckpts:
            resolved_sub = sub_ckpts[-1].parent.resolve()
            print(f"[Merge] Tự động chọn checkpoint LoRA mới nhất: {resolved_sub}")
            lora_path = str(resolved_sub)
            lp = Path(lora_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy adapter_config.json trong {lora_path}")

    # Detect base model if not specified
    if not base_model:
        cfg_path = lp / "adapter_config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            base_model = cfg.get("base_model_name_or_path")
        except Exception:
            pass
    if not base_model:
        base_model = "openai/whisper-large-v3-turbo"

    print(f"[Merge] Đang tải Base Model: {base_model}...")
    processor_src = lora_path if (lp / "processor_config.json").is_file() else base_model
    try:
        processor = AutoProcessor.from_pretrained(processor_src)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print(f"[Merge] Đang nạp và gộp LoRA adapter: {lora_path}...")
    peft_model = PeftModel.from_pretrained(model, lora_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"[Merge] Đang lưu mô hình đã gộp vào: {output_dir}...")
    merged_model.save_pretrained(out_p)
    processor.save_pretrained(out_p)
    print(f"[Merge] Hoàn tất thành công! Mô hình đã sẵn sàng tại '{output_dir}'.")


def main():
    ap = argparse.ArgumentParser(description="Gộp Whisper LoRA weights vào base model")
    ap.add_argument("--lora-path", default="whisper-turbo-vi-lora", help="Thư mục chứa LoRA adapter Whisper")
    ap.add_argument("--output-dir", default="whisper-turbo-vi-merged", help="Thư mục xuất model đã gộp")
    ap.add_argument("--base-model", default=None, help="Tên model base HF (mặc định tự đọc từ adapter_config.json)")
    args = ap.parse_args()

    merge_and_save(args.lora_path, args.output_dir, args.base_model)


if __name__ == "__main__":
    main()
