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

from src.agent import render_chatml_preserving_think
from src.entity_normalizer import EntityNormalizer
from src.guardrail import PureSchemaStateTracker, normalize_arguments, parse_tool_calls
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
        prompt_text = render_chatml_preserving_think(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
        )
        return {
            "sample": s,
            "prompt_text": prompt_text,
            "sample_id": s["id"],
        }


def compute_kl_k3(
    curr_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    resp_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute non-negative Schulman K3 KL divergence over masked assistant tokens."""
    log_ratio = torch.clamp(ref_log_probs - curr_log_probs, min=-20.0, max=20.0)
    k3_kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (k3_kl * resp_mask).sum() / resp_mask.sum().clamp(min=1.0)


def evaluate_executed_trajectory(
    sample: dict[str, Any],
    executed_calls: list[dict[str, Any]],
    pred_env: Any,
    has_exec_error: bool,
    free_text_keys: set[str],
    _canon_scalar: Any,
    _tokens: Any,
    root: Path,
) -> tuple[float, float, float]:
    """Compute verifiable ToolEnv state-hash + F1 reward from an already executed rollout."""
    qid = sample["id"]
    gt_label = sample.get("label") or {}
    gt_calls = gt_label.get("tool_calls") or []

    gt_env = ToolEnv("train", qid, root=root)
    for c in gt_calls:
        gt_env.call(c.get("name", ""), c.get("arguments") or {})
    gt_hash = gt_env.get_state_hash()

    pred_hash = pred_env.get_state_hash()
    hash_match = 1.0 if (pred_hash == gt_hash) else 0.0
    _, full_f1 = compute_call_f1(executed_calls, gt_calls, free_text_keys, _canon_scalar, _tokens)

    if hash_match < 1.0:
        reward = 0.0
    elif not gt_calls:
        reward = 1.0 if not executed_calls else 0.0
    else:
        reward = full_f1

    if has_exec_error and reward > 0.1:
        reward -= 0.1

    return reward, hash_match, full_f1


def compute_verifiable_reward(
    sample: dict[str, Any],
    pred_text: str,
    catalog_tools_map: dict[str, dict[str, Any]],
    free_text_keys: set[str],
    _canon_scalar: Any,
    _tokens: Any,
    root: Path,
) -> tuple[float, float, float]:
    """Evaluates one model text trajectory directly on in-memory ToolEnv.
    Returns: (total_reward, hash_match, f1)
    """
    qid = sample["id"]
    pred_calls_raw = parse_tool_calls(pred_text)
    cur_time = sample.get("context", {}).get("current_time", "")
    tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"

    pred_env = ToolEnv("train", qid, root=root)
    executed_calls = []
    has_exec_error = False

    for c in pred_calls_raw:
        t_name = c.get("name", "")
        t_args = normalize_arguments(t_name, c.get("arguments") or {}, catalog_tools_map, tz_offset=tz_offset)
        res = pred_env.call(t_name, t_args)
        if not res.get("ok"):
            has_exec_error = True
        executed_calls.append({"name": t_name, "arguments": t_args})

    return evaluate_executed_trajectory(
        sample=sample,
        executed_calls=executed_calls,
        pred_env=pred_env,
        has_exec_error=has_exec_error,
        free_text_keys=free_text_keys,
        _canon_scalar=_canon_scalar,
        _tokens=_tokens,
        root=root,
    )


def rollout_multiturn_trajectory(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    prompt_text: str,
    catalog_tools_map: dict[str, dict[str, Any]],
    device: torch.device,
    max_steps: int = 6,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    root: Path = ROOT,
) -> tuple[list[int], list[int], list[dict[str, Any]], Any, bool]:
    """Run an interactive multi-turn ReAct rollout against ToolEnv with exact token action masks."""
    qid = sample["id"]
    cur_time = sample.get("context", {}).get("current_time", "")
    tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"

    pred_env = ToolEnv("train", qid, root=root)
    state_tracker = PureSchemaStateTracker(catalog_tools_map)
    normalizer = EntityNormalizer(enabled=True)

    seq_ids: list[int] = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    action_mask: list[int] = [0] * len(seq_ids)
    executed_calls: list[dict[str, Any]] = []
    has_exec_error = False

    stop_token_ids = [tok for tok in (tokenizer.eos_token_id,) if tok is not None]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in stop_token_ids:
        stop_token_ids.append(im_end_id)

    for _ in range(max_steps):
        inp = torch.tensor([seq_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=inp,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=stop_token_ids or tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out_ids[0, len(seq_ids):].tolist()
        if not gen_ids:
            break

        seq_ids.extend(gen_ids)
        action_mask.extend([1] * len(gen_ids))

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        calls = parse_tool_calls(gen_text)
        if not calls:
            break

        tc = calls[0]
        t_name = tc.get("name", "")
        t_args = normalize_arguments(
            t_name,
            tc.get("arguments") or {},
            catalog_tools_map,
            tz_offset=tz_offset,
        )
        res = pred_env.call(t_name, t_args)
        if not res.get("ok"):
            has_exec_error = True
        executed_calls.append({"name": t_name, "arguments": t_args})
        state_tracker.record_observation(t_name, t_args, res)

        if t_name == "sys.ask_missing_info":
            break

        prompt_res = normalizer.recover_empty_read(
            t_name,
            t_args,
            res,
            sample,
            state_tracker.observed_entities,
        )
        obs_text = (
            f"\n<|im_start|>user\n"
            f"<tool_response>\n{json.dumps(prompt_res, ensure_ascii=False)}\n</tool_response>"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        obs_ids = list(tokenizer.encode(obs_text, add_special_tokens=False))
        seq_ids.extend(obs_ids)
        action_mask.extend([0] * len(obs_ids))

    return seq_ids, action_mask, executed_calls, pred_env, has_exec_error


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

    # Load dataset and exclude SFT validation split to prevent train/val leakage
    val_ids_path = ROOT / "data" / "sft" / "val_ids.json"
    val_ids = set(json.loads(val_ids_path.read_text(encoding="utf-8"))) if val_ids_path.is_file() else set()
    raw_lines = (ROOT / "train" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in raw_lines if line.strip()]
    if val_ids:
        samples = [s for s in samples if s["id"] not in val_ids]
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

            # 1 & 2. MULTI-TURN REACT ROLLOUT + VERIFIABLE REWARD EVALUATION
            group_rewards = []
            group_rollouts: list[tuple[list[int], list[int]]] = []

            for _g_idx in range(args.group_size):
                seq_ids, action_mask, executed_calls, pred_env, has_exec_error = rollout_multiturn_trajectory(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    prompt_text=prompt_str,
                    catalog_tools_map=catalog_tools_map,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    root=ROOT,
                )
                group_rollouts.append((seq_ids, action_mask))

                r, h_match, _f1 = evaluate_executed_trajectory(
                    sample=sample,
                    executed_calls=executed_calls,
                    pred_env=pred_env,
                    has_exec_error=has_exec_error,
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

            # 4. POLICY GRADIENT & K3 KL LOSS COMPUTATION (ASSISTANT ACTION TOKENS ONLY)
            optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=device)

            for g_idx in range(args.group_size):
                adv = advantages[g_idx]
                if abs(adv.item()) < 1e-6:
                    continue

                seq_ids, action_mask = group_rollouts[g_idx]
                if len(seq_ids) < 2 or not any(action_mask[1:]):
                    continue

                full_seq = torch.tensor([seq_ids], dtype=torch.long, device=device)
                labels = full_seq.clone()
                action_t = torch.tensor([action_mask], dtype=torch.bool, device=device)
                labels[~action_t] = -100

                # Current policy logits
                curr_logits = model(full_seq).logits
                curr_shift = curr_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                log_probs = -F.cross_entropy(
                    curr_shift.view(-1, curr_shift.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(shift_labels.shape)

                # Mask out initial prompt and <tool_response> tokens
                resp_mask = (shift_labels != -100).float()
                seq_log_prob = (log_probs * resp_mask).sum() / resp_mask.sum().clamp(min=1.0)

                # Reference policy K3 KL penalty (non-negative)
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_logits = model(full_seq).logits
                        ref_shift = ref_logits[:, :-1, :].contiguous()
                        ref_log_probs = -F.cross_entropy(
                            ref_shift.view(-1, ref_shift.size(-1)),
                            shift_labels.view(-1),
                            reduction="none",
                            ignore_index=-100,
                        ).view(shift_labels.shape)

                kl_div = compute_kl_k3(log_probs, ref_log_probs, resp_mask)

                # GRPO surrogate loss: - Advantage * log_prob + beta * K3_KL
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
