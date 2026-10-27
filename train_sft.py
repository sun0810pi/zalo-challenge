"""LoRA / QLoRA SFT training script for Qwen3-4B / Qwen2.5-3B-Instruct on Google Colab Pro."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Qwen3-4B trên dữ liệu ReAct Tool-Calling ZAC 2026")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="Model gốc (< 5B tham số)")
    ap.add_argument("--data-dir", default="data/sft", help="Thư mục chứa sft_train.jsonl và sft_val.jsonl")
    ap.add_argument("--output-dir", default="checkpoints/qwen3-4b-zac-lora", help="Thư mục lưu LoRA adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument(
        "--use-4bit",
        action="store_true",
        help="Bật QLoRA 4-bit nếu chạy trên GPU 16GB (T4); trên L4/A100/A30 24GB+ có thể chạy thẳng BF16 LoRA",
    )
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(args.data_dir)
    if not (data_dir / "sft_train.jsonl").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {data_dir / 'sft_train.jsonl'}. Hãy chạy `python prepare_sft_data.py` trước!"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "sft_train.jsonl"),
            "validation": str(data_dir / "sft_val.jsonl"),
        },
    )

    def format_row(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}

    dataset = dataset.map(format_row, remove_columns=dataset["train"].column_names)

    # Tự động kiểm tra các tham số hợp lệ của SFTConfig theo phiên bản trl/transformers trên Colab
    valid_sft_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 15,
        "logging_steps": 10,
        "eval_strategy": "steps" if "eval_strategy" in valid_sft_params else "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "dataset_text_field": "text",
        "packing": False,
        "gradient_checkpointing": True,
        "seed": 42,
        "report_to": "none",
    }
    if "max_length" in valid_sft_params:
        sft_kwargs["max_length"] = args.max_seq_len
    elif "max_seq_length" in valid_sft_params:
        sft_kwargs["max_seq_length"] = args.max_seq_len

    filtered_sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in valid_sft_params}
    sft_config = SFTConfig(**filtered_sft_kwargs)

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": peft_config,
        "args": sft_config,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Hoàn tất huấn luyện! LoRA adapter đã lưu tại: {args.output_dir}")


if __name__ == "__main__":
    main()
