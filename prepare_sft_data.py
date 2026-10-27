"""Convert train/train.jsonl into causal multi-step ReAct ChatML trajectories for Qwen3-4B / Qwen2.5-3B SFT."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from src.prompt_builder import build_initial_messages


def build_sft_trajectory(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a causal multi-turn ChatML conversation where each tool call sees previous tool_results."""
    # Extract ground-truth history user audio_text
    history_transcripts = {}
    for turn in sample.get("history") or []:
        if turn.get("role") == "user" and turn.get("audio") and turn.get("audio_text"):
            history_transcripts[turn["audio"]] = turn["audio_text"]

    messages = build_initial_messages(
        sample=sample,
        final_user_text=sample.get("audio_text", ""),
        catalog_tools_map=catalog_tools_map,
        history_transcripts=history_transcripts,
    )

    label = sample.get("label") or {}
    t_calls = label.get("tool_calls") or []
    t_results = label.get("tool_results") or []

    if not t_calls:
        messages.append(
            {
                "role": "assistant",
                "content": "Yêu cầu không cần thực hiện thao tác trên ứng dụng.",
            }
        )
        return messages

    for tc, tr in zip(t_calls, t_results):
        call_payload = {"name": tc["name"], "arguments": tc.get("arguments") or {}}
        messages.append(
            {
                "role": "assistant",
                "content": f"<tool_call>\n{json.dumps(call_payload, ensure_ascii=False)}\n</tool_call>",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"<tool_response>\n{json.dumps(tr, ensure_ascii=False)}\n</tool_response>",
            }
        )

    # Final closing turn after all tool calls finish
    last_call_name = t_calls[-1]["name"]
    if last_call_name == "sys.ask_missing_info":
        messages.append(
            {
                "role": "assistant",
                "content": "Vui lòng cung cấp thêm thông tin chi tiết để tôi thực hiện yêu cầu.",
            }
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "content": "Đã hoàn thành yêu cầu của bạn.",
            }
        )
    return messages


def main():
    ap = argparse.ArgumentParser(description="Tạo dữ liệu SFT multi-turn từ train/train.jsonl")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--val-size", type=int, default=145, help="Số mẫu tách làm tập validation (mặc định 145 / 1545)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    sys.path.insert(0, str((root / "train").resolve()))
    from zaic_tools.validate import catalog_tools  # noqa: E402

    catalog = json.loads((root / "train/train_tools_catalog.json").read_text(encoding="utf-8"))
    catalog_tools_map = catalog_tools(catalog)

    samples = [
        json.loads(line)
        for line in (root / "train/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_idx = set(indices[: args.val_size])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = []
    val_rows = []
    val_ids = []

    for i, s in enumerate(samples):
        traj = {"id": s["id"], "messages": build_sft_trajectory(s, catalog_tools_map)}
        if i in val_idx:
            val_rows.append(traj)
            val_ids.append(s["id"])
        else:
            train_rows.append(traj)

    (out_dir / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "sft_val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "val_ids.json").write_text(
        json.dumps(val_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã tạo {len(train_rows)} mẫu train -> {out_dir / 'sft_train.jsonl'}")
    print(f"Đã tạo {len(val_rows)} mẫu val   -> {out_dir / 'sft_val.jsonl'} (danh sách ID lưu tại {out_dir / 'val_ids.json'})")


if __name__ == "__main__":
    main()
