"""RLVR (Reinforcement Learning with Verifiable Rewards) via GRPO for ZAC 2026 LifeMate.
Trained on Qwen3-4B (< 5B parameters) using the in-memory `zaic_tools.ToolEnv` simulator.

Hàm thưởng (Verifiable Reward):
    R(tau) = 0.5 * I(State Hash Khớp 100%) + 0.5 * Full Tool Call F1

GRPO (Group Relative Policy Optimization):
    - Sinh G candidate trajectories cho mỗi sample (G = 4).
    - Đo thưởng R_i trực tiếp trên ToolEnv không cần mạng Critic.
    - Chuẩn hóa Advantage theo nhóm: A_i = (R_i - mean(R)) / (std(R) + 1e-8).
    - Cập nhật LoRA policy với KL penalty đối chiếu reference model.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "train") not in sys.path:
    sys.path.insert(0, str(ROOT / "train"))

from evaluate import _load_zaic_tools, compute_call_f1
from zaic_tools import ToolEnv
from zaic_tools.validate import catalog_tools

from src.guardrail import normalize_arguments, parse_tool_calls
from src.prompt_builder import build_initial_messages
from src.router import SemanticToolRouter


class GRPODataset(Dataset):
    """Dataset serving prompts formatted in ChatML for GRPO trajectory generation."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        catalog_tools_map: dict[str, dict[str, Any]],
        catalog_apps_map: dict[str, dict[str, Any]],
        router: SemanticToolRouter,
        tokenizer: Any,
    ):
        self.samples = samples
        self.catalog_tools_map = catalog_tools_map
        self.catalog_apps_map = catalog_apps_map
        self.router = router
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        user_text = s.get("audio_text") or ""
        messages = build_initial_messages(
            sample=s,
            final_user_text=user_text,
            catalog_tools_map=self.catalog_tools_map,
            history_transcripts={},
            router=self.router,
            catalog_apps_map=self.catalog_apps_map,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "sample": s,
            "prompt_text": prompt_text,
            "sample_id": s["id"],
        }


def compute_verifiable_reward(
    sample: dict[str, Any],
    pred_text: str,
    catalog_tools_map: dict[str, dict[str, Any]],
    free_text_keys: set[str],
    _canon_scalar: Any,
    _tokens: Any,
    root: Path,
) -> tuple[float, float, float]:
    """Evaluates one model trajectory directly on in-memory ToolEnv.
    Returns: (total_reward, hash_match, f1)
    """
    qid = sample["id"]
    gt_label = sample.get("label") or {}
    gt_calls = gt_label.get("tool_calls") or []

    # Get Ground Truth State Hash
    gt_env = ToolEnv("train", qid, root=root)
    for c in gt_calls:
        gt_env.call(c.get("name", ""), c.get("arguments") or {})
    gt_hash = gt_env.get_state_hash()

    # Parse and execute predicted calls on fresh ToolEnv
    pred_calls_raw = parse_tool_calls(pred_text)
    cur_time = sample.get("context", {}).get("current_time", "")
    tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"

    pred_env = ToolEnv("train", qid, root=root)
    executed_calls = []
    has_exec_error = False

    for c in pred_calls_raw:
        t_name = c.get("name", "")
        t_args = normalize_arguments(t_name, c.get("arguments") or {}, catalog_tools_map, tz_offset=tz_offset)
        # Execute in environment
        res = pred_env.call(t_name, t_args)
        if not res.get("ok"):
            has_exec_error = True
        executed_calls.append({"name": t_name, "arguments": t_args})

    pred_hash = pred_env.get_state_hash()

    # 1. State Hash binary reward
    hash_match = 1.0 if (pred_hash == gt_hash) else 0.0

    # 2. Full Tool Call F1
    _, full_f1 = compute_call_f1(executed_calls, gt_calls, free_text_keys, _canon_scalar, _tokens)

    # 3. Overall Verifiable Reward (Strict ZAIC 2026 Hard-Gate)
    if hash_match < 1.0:
        reward = 0.0
    else:
        if not gt_calls:
            reward = 1.0 if not executed_calls else 0.0
        else:
            reward = full_f1

    # Penalty for schema or execution errors
    if has_exec_error and reward > 0.1:
        reward -= 0.1

    return reward, hash_match, full_f1


def main():
    ap = argparse.ArgumentParser(description="RLVR / GRPO Trainer with In-Memory ToolEnv Verifiable Rewards")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--base-lora", default=None, help="Đường dẫn LoRA checkpoint khởi điểm (VD: checkpoints/qwen3-4b-zac-lora-p1-concise)")
    ap.add_argument("--output-dir", default="checkpoints/qwen3-4b-zac-lora-grpo-rlvr")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4, help="Số lượng candidate rollout mỗi prompt (G trong GRPO)")
    ap.add_argument("--batch-size", type=int, default=1, help="Số prompts mỗi step huấn luyện")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--clip-eps", type=float, default=0.2, help="PPO/GRPO clipping epsilon")
    ap.add_argument("--kl-beta", type=float, default=0.04, help="Hệ số phạt KL divergence")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu train (0 = toàn bộ)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GRPO-RLVR] Khởi động RLVR với GRPO trên thiết bị: {device}")
    print(f"  - Model: {args.model}")
    print(f"  - Base LoRA: {args.base_lora}")
    print(f"  - Group Size (G): {args.group_size}")
    print(f"  - KL Beta: {args.kl_beta} | Clip Eps: {args.clip_eps} | LR: {args.lr}")

    # Load environment and catalogs
    zaic_tools, FREE_TEXT_KEYS, _canon_scalar, _tokens, catalog_tools_fn, _ = _load_zaic_tools(ROOT, "train")
    catalog_path = ROOT / "train" / "train_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_tools_map = catalog_tools_fn(catalog)
    catalog_apps_map = {app["id"]: app for app in catalog.get("apps", [])}
    router = SemanticToolRouter(top_k_recommended=4)

    # Load dataset
    raw_lines = (ROOT / "train" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in raw_lines if line.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"[GRPO-RLVR] Nạp {len(samples)} mẫu train cho vòng lặp tương tác RLVR.")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.base_lora and Path(args.base_lora).exists():
        print(f"[GRPO-RLVR] Nạp pretrained LoRA adapter từ {args.base_lora}...")
        model = PeftModel.from_pretrained(base_model, args.base_lora, is_trainable=True)
    else:
        print("[GRPO-RLVR] Khởi tạo LoRA adapter mới...")
        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_cfg)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    ds = GRPODataset(samples, catalog_tools_map, catalog_apps_map, router, tokenizer)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[GRPO-RLVR] Bắt đầu huấn luyện RLVR với hàm thưởng Verifiable từ ToolEnv...\n")

    for epoch in range(1, args.epochs + 1):
        total_epoch_reward = 0.0
        total_hash_matches = 0
        total_trajectories = 0

        for step, item in enumerate(ds, 1):
            sample = item["sample"]
            prompt_str = item["prompt_text"]

            # 1. GENERATION PHASE: Sinh G trajectories với temperature sampling
            prompt_inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Generate G sequences
            with torch.no_grad():
                gen_outputs = model.generate(
                    **prompt_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    num_return_sequences=args.group_size,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # 2. VERIFIABLE REWARD EVALUATION PHASE: Đo thưởng trên Simulator ToolEnv
            group_rewards = []
            group_texts = []
            group_tokens = []

            for g_idx in range(args.group_size):
                full_ids = gen_outputs[g_idx]
                resp_ids = full_ids[prompt_len:]
                resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                group_texts.append(resp_text)
                group_tokens.append(full_ids)

                r, h_match, f1 = compute_verifiable_reward(
                    sample=sample,
                    pred_text=resp_text,
                    catalog_tools_map=catalog_tools_map,
                    free_text_keys=FREE_TEXT_KEYS,
                    _canon_scalar=_canon_scalar,
                    _tokens=_tokens,
                    root=ROOT,
                )
                group_rewards.append(r)
                if h_match > 0.5:
                    total_hash_matches += 1
                total_trajectories += 1
                total_epoch_reward += r

            # 3. ADVANTAGE COMPUTATION (GRPO GROUP NORMALIZATION)
            r_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
            r_mean = r_tensor.mean()
            r_std = r_tensor.std() + 1e-8
            advantages = (r_tensor - r_mean) / r_std

            # 4. POLICY GRADIENT & KL LOSS COMPUTATION
            optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=device)

            for g_idx in range(args.group_size):
                adv = advantages[g_idx]
                if abs(adv.item()) < 1e-6:
                    continue

                full_seq = group_tokens[g_idx].unsqueeze(0)
                labels = full_seq.clone()
                labels[:, :prompt_len] = -100

                # Current policy logits
                curr_logits = model(full_seq).logits
                curr_shift = curr_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                log_probs = -F.cross_entropy(
                    curr_shift.view(-1, curr_shift.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(shift_labels.shape)

                # Mask out prompt tokens
                resp_mask = (shift_labels != -100).float()
                seq_log_prob = (log_probs * resp_mask).sum() / resp_mask.sum().clamp(min=1.0)

                # Reference policy KL penalty (disable LoRA to compute ref log_prob)
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_logits = model(full_seq).logits
                        ref_shift = ref_logits[:, :-1, :].contiguous()
                        ref_log_probs = -F.cross_entropy(
                            ref_shift.view(-1, ref_shift.size(-1)),
                            shift_labels.view(-1),
                            reduction="none",
                        ).view(shift_labels.shape)
                        ref_seq_log_prob = (ref_log_probs * resp_mask).sum() / resp_mask.sum().clamp(min=1.0)

                # Approximate KL divergence
                kl_div = seq_log_prob - ref_seq_log_prob

                # GRPO surrogate loss: - Advantage * log_prob + beta * KL
                loss_g = -adv * seq_log_prob + args.kl_beta * kl_div
                step_loss += loss_g / args.group_size

            if step_loss.item() != 0.0:
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if step % 20 == 0 or step == len(ds):
                avg_r = total_epoch_reward / max(1, total_trajectories)
                hash_acc = (total_hash_matches / max(1, total_trajectories)) * 100
                print(
                    f"Epoch {epoch}/{args.epochs} | Bước [{step}/{len(ds)}] "
                    f"| Mean Reward={avg_r:.4f} | State Hash Khớp={hash_acc:.1f}% "
                    f"| Loss={step_loss.item():.4f}"
                )

        # Save checkpoint per epoch
        epoch_dir = out_dir / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"[GRPO-RLVR] Lưu checkpoint sau Epoch {epoch} -> {epoch_dir}")

    # Save final model
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n[GRPO-RLVR] Hoàn tất huấn luyện RLVR! Model đã lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
