"""LoRA / QLoRA SFT training script for Qwen3-4B / Qwen2.5-3B-Instruct on Google Colab Pro."""
from __future__ import annotations

import argparse
import inspect
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Any

from src.agent import render_chatml_preserving_think


def _patch_colab_torchao():
    """Bypass Colab's pre-installed incompatible torchao 0.10.0 check in peft >= 0.21."""
    try:
        import peft.import_utils

        peft.import_utils.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao

        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass


class AssistantOnlyDataCollator:
    """Mask all system/user/tool_response tokens with -100 so loss is computed strictly on assistant `<think>` + `<tool_call>` turns."""

    def __init__(self, tokenizer: Any, max_seq_len: int = 8192):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.assistant_start_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def _find_subseq(self, seq: list[int], sub: list[int], start: int = 0) -> int:
        n, m = len(seq), len(sub)
        if m == 0 or m > n:
            return -1
        first = sub[0]
        for i in range(start, n - m + 1):
            if seq[i] == first and seq[i : i + m] == sub:
                return i
        return -1

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feat in features:
            if "input_ids" in feat:
                ids = list(feat["input_ids"])[: self.max_seq_len]
            else:
                ids = self.tokenizer.encode(
                    feat["text"],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_seq_len,
                )

            labels = [-100] * len(ids)
            pos = 0
            found_any = False
            while pos < len(ids):
                start_idx = self._find_subseq(ids, self.assistant_start_ids, pos)
                if start_idx == -1:
                    break
                content_start = start_idx + len(self.assistant_start_ids)
                end_idx = self._find_subseq(ids, self.im_end_ids, content_start)
                if end_idx == -1:
                    end_token_inclusive = len(ids)
                else:
                    end_token_inclusive = end_idx + len(self.im_end_ids)

                for j in range(content_start, end_token_inclusive):
                    labels[j] = ids[j]
                found_any = True
                pos = end_token_inclusive

            if not found_any:
                labels = list(ids)

            batch_input_ids.append(ids)
            batch_labels.append(labels)

        max_len = max(len(x) for x in batch_input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        padded_ids = []
        padded_mask = []
        padded_labels = []
        for ids, lbs in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_mask.append([1] * len(ids) + [0] * pad_len)
            padded_labels.append(lbs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


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

    _patch_colab_torchao()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(args.data_dir)
    if not (data_dir / "sft_train.jsonl").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {data_dir / 'sft_train.jsonl'}. Hãy chạy `python generate_reasoning_sft.py` trước!"
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
        # Bảo toàn đầy đủ thẻ <think>...</think> ở MỌI bước assistant trong multi-turn ReAct
        text = render_chatml_preserving_think(
            tokenizer,
            example["messages"],
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_row, remove_columns=dataset["train"].column_names)
    data_collator = AssistantOnlyDataCollator(tokenizer=tokenizer, max_seq_len=args.max_seq_len)

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
        "data_collator": data_collator,
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
