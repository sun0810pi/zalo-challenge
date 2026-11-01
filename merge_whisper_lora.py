"""Script gộp (merge) Whisper LoRA adapter vào mô hình gốc và lưu lại thành Standalone Model.
Tự động tìm kiếm đệ quy adapter_config.json trong mọi thư mục con (checkpoint-*/, output/, ...).
Tự động phát hiện nếu thư mục vốn đã là Full Model (không cần merge).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def inspect_whisper_path(target_path: str | Path) -> tuple[str, str | list[str]]:
    """Phân tích thư mục checkpoint: LoRA adapter, Full Model, hoặc đường dẫn không hợp lệ."""
    p = Path(target_path)
    if not p.exists():
        parent = p.parent
        existing = [f.name for f in parent.iterdir()] if parent.is_dir() else []
        raise FileNotFoundError(
            f"\n[LỖI] Không tìm thấy đường dẫn: '{target_path}'!\n"
            f"1. Nếu chạy trên Google Colab, bạn đã mount Google Drive chưa?\n"
            f"   from google.colab import drive\n"
            f"   drive.mount('/content/drive')\n"
            f"2. Các thư mục/file đang có trong '{parent}':\n"
            f"   {existing[:20]}\n"
        )

    # 1. Kiểm tra adapter_config.json trực tiếp
    if (p / "adapter_config.json").is_file():
        return "lora", str(p.resolve())

    # 2. Tìm kiếm đệ quy adapter_config.json trong các thư mục con (checkpoint-*/, final/, ...)
    sub_adapters = sorted(
        p.rglob("adapter_config.json"),
        key=lambda f: (
            int(re.findall(r"\d+", f.parent.name)[-1]) if re.findall(r"\d+", f.parent.name) else 0,
            f.stat().st_mtime,
        ),
    )
    if sub_adapters:
        best_adapter_dir = sub_adapters[-1].parent.resolve()
        print(f"[Merge] Tự động tìm thấy LoRA checkpoint mới nhất tại: {best_adapter_dir}")
        return "lora", str(best_adapter_dir)

    # 3. Kiểm tra xem thư mục có phải là Standalone Full Model sẵn rồi không
    weight_exts = ("*.safetensors", "*.bin")
    has_full_weights = any(p.glob(ext) for ext in weight_exts)
    if (p / "config.json").is_file() and has_full_weights:
        return "full_model", str(p.resolve())

    for sub_cfg in p.rglob("config.json"):
        if any(sub_cfg.parent.glob(ext) for ext in weight_exts):
            return "full_model", str(sub_cfg.parent.resolve())

    # 4. Không tìm thấy định dạng quen thuộc -> Liệt kê các file để chẩn đoán
    files_found = [str(f.relative_to(p)) for f in p.glob("**/*") if f.is_file()][:25]
    return "unknown", files_found


def merge_and_save(
    lora_path: str,
    output_dir: str,
    base_model: str | None = None,
):
    kind, resolved = inspect_whisper_path(lora_path)

    if kind == "full_model":
        print("\n" + "=" * 70)
        print("[THÔNG BÁO QUAN TRỌNG]")
        print(f"Thư mục '{resolved}' ĐÃ LÀ MÔ HÌNH HOÀN CHỈNH (Full Fine-Tuned Model)!")
        print("Trong thư mục này đã có sẵn trọng số đầy đủ (safetensors/bin + config.json).")
        print("-> BẠN KHÔNG CẦN CHẠY MERGE!")
        print()
        print("Để chạy inference trên Colab, bạn chỉ cần dùng trực tiếp lệnh:")
        print(f"  !python run_infer.py --split public_test --asr-model {resolved} ...")
        print("=" * 70 + "\n")
        return

    if kind == "unknown":
        print("\n" + "=" * 70)
        print(f"[LỖI] Không tìm thấy 'adapter_config.json' hay trọng số mô hình trong '{lora_path}'!")
        print(f"Các file đang có trong thư mục này:")
        for fn in resolved:
            print(f"  - {fn}")
        print("\nCách khắc phục:")
        print("1. Kiểm tra lại thư mục checkpoint bạn đã lưu khi train Whisper LoRA.")
        print("2. Trong Colab, hãy chạy lệnh sau để xem các file thực tế:")
        print(f"   !ls -la {lora_path}")
        print("=" * 70 + "\n")
        raise FileNotFoundError(f"Không tìm thấy adapter_config.json trong {lora_path}")

    # Tiến hành gộp LoRA adapter
    lora_dir = Path(resolved)
    if not base_model:
        cfg_path = lora_dir / "adapter_config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            base_model = cfg.get("base_model_name_or_path")
        except Exception:
            pass
    if not base_model:
        base_model = "openai/whisper-large-v3-turbo"

    print(f"[Merge] Đang tải Base Model: {base_model}...")
    processor_src = str(lora_dir) if (lora_dir / "processor_config.json").is_file() else base_model
    try:
        processor = AutoProcessor.from_pretrained(processor_src)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    print(f"[Merge] Đang nạp và gộp LoRA adapter từ '{lora_dir}'...")
    peft_model = PeftModel.from_pretrained(model, str(lora_dir))
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"[Merge] Đang lưu mô hình hoàn chỉnh vào: '{output_dir}'...")
    merged_model.save_pretrained(out_p)
    processor.save_pretrained(out_p)
    print(f"\n[Merge] THÀNH CÔNG! Mô hình hoàn chỉnh đã sẵn sàng tại '{output_dir}'.")
    print(f"Bây giờ bạn có thể chạy inference với:")
    print(f"  !python run_infer.py --split public_test --asr-model {output_dir} ...")


def main():
    ap = argparse.ArgumentParser(description="Gộp Whisper LoRA weights vào base model")
    ap.add_argument("--lora-path", required=True, help="Thư mục chứa LoRA adapter Whisper")
    ap.add_argument("--output-dir", default="models/whisper-finetuned", help="Thư mục xuất model đã gộp")
    ap.add_argument("--base-model", default=None, help="Tên model base HF (mặc định tự đọc từ adapter_config.json)")
    args = ap.parse_args()

    merge_and_save(args.lora_path, args.output_dir, args.base_model)


if __name__ == "__main__":
    main()
