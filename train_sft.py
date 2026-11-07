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

    def __init__(self, tokenizer: Any, max_seq_len: int = 8192, strict_no_truncation: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.strict_no_truncation = strict_no_truncation
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

    def compute_labels_for_ids(self, ids: list[int]) -> list[int]:
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
                if self.strict_no_truncation:
                    raise ValueError(
                        f"Assistant turn truncated before '<|im_end|>' at sequence length {len(ids)} "
                        f"(max_seq_len={self.max_seq_len}). Increase --max-seq-len to >= 9216."
                    )
                end_token_inclusive = len(ids)
            else:
                end_token_inclusive = end_idx + len(self.im_end_ids)

            for j in range(content_start, end_token_inclusive):
                labels[j] = ids[j]
            found_any = True
            pos = end_token_inclusive

        if not found_any:
            raise ValueError(
                f"No assistant span found in sequence (length={len(ids)}). "
                "Ensure ChatML '<|im_start|>assistant' and '<|im_end|>' tokens are properly formatted."
            )
        return labels

    def compute_weights_for_ids(self, ids: list[int], labels: list[int]) -> list[float]:
        weights = [0.0] * len(ids)
        for j, (tok_id, lab) in enumerate(zip(ids, labels)):
            if lab == -100:
                continue
            tok_str = self.tokenizer.decode([tok_id]).strip()
            # Punctuation / JSON structure tokens get low weight
            if tok_str in ('{', '}', '"', '":', '",', ':', ',', '[', ']', '{"', '"}'):
                weights[j] = 0.5
            # Tool call tags and action decision tokens get highest weight
            elif any(k in tok_str for k in ('<tool_call>', '</tool_call>', 'sys.ask_missing_info', 'name')):
                weights[j] = 2.5
            # Numbers, entity IDs (evt_, c_, ord_, trk_, pl_), dates get high weight
            elif any(c.isdigit() for c in tok_str) or any(prefix in tok_str for prefix in ('evt_', 'c_', 'ord_', 'trk_', 'pl_')):
                weights[j] = 2.0
            else:
                weights[j] = 1.0
        return weights

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feat in features:
            if "input_ids" in feat and "labels" in feat:
                ids = list(feat["input_ids"])[: self.max_seq_len]
                labels = list(feat["labels"])[: self.max_seq_len]
                batch_input_ids.append(ids)
                batch_labels.append(labels)
                continue

            if "input_ids" in feat:
                full_ids = list(feat["input_ids"])
                if self.strict_no_truncation and len(full_ids) > self.max_seq_len:
                    raise ValueError(
                        f"Sequence length {len(full_ids)} exceeds max_seq_len={self.max_seq_len}. "
                        "Increase --max-seq-len to >= 9216 to prevent assistant target truncation."
                    )
                ids = full_ids[: self.max_seq_len]
            else:
                if self.strict_no_truncation:
                    full_ids = self.tokenizer.encode(feat["text"], add_special_tokens=False)
                    if len(full_ids) > self.max_seq_len:
                        raise ValueError(
                            f"Sequence length {len(full_ids)} exceeds max_seq_len={self.max_seq_len}. "
                            "Increase --max-seq-len to >= 9216 to prevent assistant target truncation."
                        )
                    ids = full_ids
                else:
                    ids = self.tokenizer.encode(
                        feat["text"],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_seq_len,
                    )

            labels = self.compute_labels_for_ids(ids)
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
    ap.add_argument("--output-dir", default="checkpoints/qwen3-4b-zac-lora-v3", help="Thư mục lưu LoRA adapter")
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
    ap.add_argument(
        "--optimized",
        action="store_true",
        help="Bật cấu hình tối ưu tốc độ trên NVIDIA L4 (pretokenize, group_by_length, sdpa, adamw_torch_fused, max_seq_len>=9216)",
    )
    ap.add_argument(
        "--pretokenize",
        action="store_true",
        help="Tokenize trước input_ids và labels 1 lần để loại bỏ CPU bottleneck trong DataCollator",
    )
    ap.add_argument(
        "--group-by-length",
        action="store_true",
        help="Gom nhóm các mẫu có độ dài tương đương vào cùng micro-batch để giảm ~80%% padding waste",
    )
    ap.add_argument(
        "--attn-implementation",
        default=None,
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Cơ chế Attention (sdpa hoặc flash_attention_2 giúp tăng tốc trên L4/A100)",
    )
    ap.add_argument(
        "--optim",
        default=None,
        help="Optimizer (ví dụ: adamw_torch_fused hoặc paged_adamw_8bit)",
    )
    ap.add_argument(
        "--non-reentrant-checkpointing",
        action="store_true",
        help="Sử dụng gradient_checkpointing_kwargs={'use_reentrant': False} để giảm overhead autograd",
    )
    ap.add_argument(
        "--verify-no-truncation",
        action="store_true",
        help="Kiểm tra nghiêm ngặt đảm bảo không có bất kỳ lượt assistant hoặc tool_call nào bị cắt cụt",
    )
    ap.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Bật hàm loss có trọng số theo token (TVAFT: Token-Weighted Loss cho Action Triage & Tool Schema)",
    )

    # Solutions 1, 5, and Baseline Preset Switches
    ap.add_argument(
        "--enable-hammer",
        action="store_true",
        help="[Solution 5] Tự động chọn candidate v4_hammer_when2call và bật weighted-loss",
    )
    ap.add_argument(
        "--enable-rsft",
        action="store_true",
        help="[Solution 1] Tự động chọn candidate rsft (Simulator Rejection Sampling, R=1.0)",
    )
    ap.add_argument(
        "--enable-concise",
        action="store_true",
        help="[Baseline v3] Tự động chọn candidate p1_concise",
    )

    args = ap.parse_args()

    # Apply solution presets if requested
    if args.enable_hammer:
        if args.data_dir == "data/sft":
            args.data_dir = "data/sft/candidates/v4_hammer_when2call"
        if args.output_dir == "checkpoints/qwen3-4b-zac-lora-v3":
            args.output_dir = "checkpoints/qwen3-4b-zac-lora-v4-hammer"
        args.weighted_loss = True
        print(f"[Solution 5: Hammer Masking] Đã chọn data={args.data_dir} | output={args.output_dir} | weighted_loss=True")
    elif args.enable_rsft:
        if args.data_dir == "data/sft":
            args.data_dir = "data/sft/candidates/rsft"
        if args.output_dir == "checkpoints/qwen3-4b-zac-lora-v3":
            args.output_dir = "checkpoints/qwen3-4b-zac-lora-rsft"
        print(f"[Solution 1: RSFT] Đã chọn data={args.data_dir} | output={args.output_dir}")
    elif args.enable_concise:
        if args.data_dir == "data/sft":
            args.data_dir = "data/sft/candidates/p1_concise"
        if args.output_dir == "checkpoints/qwen3-4b-zac-lora-v3":
            args.output_dir = "checkpoints/qwen3-4b-zac-lora-p1-concise"
        print(f"[Baseline v3: Concise] Đã chọn data={args.data_dir} | output={args.output_dir}")

    if args.optimized:
        args.pretokenize = True
        args.group_by_length = True
        args.non_reentrant_checkpointing = True
        args.verify_no_truncation = True
        if args.attn_implementation is None:
            args.attn_implementation = "sdpa"
        if args.optim is None:
            args.optim = "adamw_torch_fused"
        if args.max_seq_len < 9216:
            args.max_seq_len = 9216

    _patch_colab_torchao()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if args.optimized and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir)
    if not (data_dir / "sft_train.jsonl").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {data_dir / 'sft_train.jsonl'}. Hãy chuẩn bị dữ liệu SFT trước!"
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
    model_load_kwargs: dict[str, Any] = {
        "quantization_config": quant_config,
        "dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_load_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_load_kwargs,
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

    data_collator = AssistantOnlyDataCollator(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        strict_no_truncation=args.verify_no_truncation,
    )

    def format_row(example):
        text = render_chatml_preserving_think(
            tokenizer,
            example["messages"],
            add_generation_prompt=False,
        )
        if not args.pretokenize:
            return {"text": text}

        full_ids = tokenizer.encode(text, add_special_tokens=False)
        if args.verify_no_truncation and len(full_ids) > args.max_seq_len:
            raise ValueError(
                f"Sample length {len(full_ids)} exceeds max_seq_len={args.max_seq_len}. "
                "Increase --max-seq-len to >= 9216 so assistant targets are never truncated."
            )
        ids = full_ids[: args.max_seq_len]
        labels = data_collator.compute_labels_for_ids(ids)
        return {
            "text": text,
            "input_ids": ids,
            "labels": labels,
            "length": len(ids),
        }

    dataset = dataset.map(format_row, remove_columns=dataset["train"].column_names)

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
    if args.group_by_length:
        sft_kwargs["group_by_length"] = True
        sft_kwargs["length_column_name"] = "length"
    if args.optim:
        sft_kwargs["optim"] = args.optim
    if args.non_reentrant_checkpointing:
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if args.optimized and torch.cuda.is_available():
        sft_kwargs["tf32"] = True

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

    class TokenWeightedSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if "weights" not in inputs:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
            weights = inputs.pop("weights")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")

            import torch
            import torch.nn.functional as F

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            valid_mask = (shift_labels.view(-1) != -100).float()
            w = shift_weights.view(-1) * valid_mask
            weighted_loss = (loss * w).sum() / (w.sum() + 1e-8)
            return (weighted_loss, outputs) if return_outputs else weighted_loss

    trainer_cls = TokenWeightedSFTTrainer if args.weighted_loss else SFTTrainer
    trainer = trainer_cls(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Hoàn tất huấn luyện! LoRA adapter đã lưu tại: {args.output_dir}")


if __name__ == "__main__":
    main()
