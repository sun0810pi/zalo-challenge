"""Generator of Structured Micro-Reasoning ReAct SFT Trajectories for Qwen3-4B / Qwen2.5-3B.
Converts train/train.jsonl into causal, step-by-step thinking trajectories that teach the model
schema-grounded reasoning, read-before-write, self-cancellation detection, and disambiguation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from src.prompt_builder import build_initial_messages

NEGATION_KEYWORDS = [
    "à mà thôi",
    "thôi không cần",
    "thôi khỏi",
    "khoan đã",
    "khoan hãy",
    "đừng làm",
    "cứ để nguyên",
    "cứ để lịch đó",
    "chỉ cần xem",
    "chỉ xem",
    "để lại đã",
    "khoan trả",
]

CONDITIONAL_KEYWORDS = [
    "nếu mai mưa",
    "nếu mưa",
    "nếu không mưa",
    "nếu giá vàng",
    "nếu xăng",
    "nếu trên",
    "nếu dưới",
    "nếu có",
]


def _detect_cancellation(text: str) -> str:
    text_l = text.lower()
    for kw in NEGATION_KEYWORDS:
        if kw in text_l:
            return f"Phát hiện người dùng đổi ý qua cụm từ '{kw}'. Hủy thao tác thay đổi dữ liệu."
    return "Không có phát biểu đổi ý / tự hủy."


def _detect_conditional(text: str) -> str:
    text_l = text.lower()
    for kw in CONDITIONAL_KEYWORDS:
        if kw in text_l:
            return f"Phát hiện câu điều kiện '{kw}'. Cần kiểm tra điều kiện trước."
    return "Yêu cầu thực hiện trực tiếp, không ràng buộc điều kiện."


def _build_step_reasoning(
    step_idx: int,
    total_steps: int,
    tool_call: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    user_text: str,
    prev_tool_result: dict[str, Any] | None = None,
) -> str:
    """Generate a sharp, structured 3-point micro-reasoning trace (<think>...)."""
    t_name = tool_call["name"]
    spec = catalog_tools_map.get(t_name) or {}
    t_kind = spec.get("kind", "write")
    params = spec.get("parameters", {}).get("properties", {})

    lines = ["<think>"]

    # Slot 1: Intent & Condition analysis
    if step_idx == 0:
        cancellation = _detect_cancellation(user_text)
        conditional = _detect_conditional(user_text)
        lines.append(f"- Ý định & Điều kiện: {conditional} | {cancellation}")
    else:
        # Subsequent step: observe previous output
        obs = "Thành công"
        if prev_tool_result:
            if "events" in prev_tool_result:
                obs = f"Đã tìm thấy {len(prev_tool_result['events'])} sự kiện trong lịch"
            elif "contacts" in prev_tool_result:
                obs = f"Đã tìm thấy {len(prev_tool_result['contacts'])} liên hệ trong danh bạ"
            elif "devices" in prev_tool_result:
                obs = f"Đã lấy được danh sách thiết bị và trạng thái"
            elif "forecast" in prev_tool_result:
                obs = f"Đã có thông tin dự báo thời tiết"
            elif "bills" in prev_tool_result:
                obs = f"Đã có danh sách hóa đơn"
            elif "quotes" in prev_tool_result:
                obs = f"Đã lấy được báo giá xe"
        lines.append(f"- Quan sát bước trước: {obs}. Tiếp tục thực hiện mục tiêu.")

    # Slot 2: Schema Grounding & Dependency Check
    if t_name == "sys.ask_missing_info":
        lines.append(
            f"- Tra cứu Schema: Thông tin người dùng cung cấp chưa đủ hoặc có nhiều kết quả trùng nhau (mơ hồ). "
            f"Theo quy tắc hệ thống, gọi sys.ask_missing_info để hỏi lại người dùng."
        )
    elif t_kind == "read":
        # Check why we are reading
        ref_keys = [k for k, v in params.items() if "x-ref" in v]
        lines.append(
            f"- Tra cứu Schema: Cần tra cứu dữ liệu thực tế bằng tool '{t_name}' (loại read) "
            f"để lấy ID/thông tin chính xác, tuyệt đối không tự bịa ID."
        )
    else:
        # Write tool
        args = tool_call.get("arguments") or {}
        id_args = [f"{k}={v}" for k, v in args.items() if "id" in k]
        id_desc = f" ({', '.join(id_args)})" if id_args else ""
        lines.append(
            f"- Tra cứu Schema: Đã có đủ tham số cho tool '{t_name}' (loại write){id_desc}. Tiến hành thực thi."
        )

    # Slot 3: Decision
    lines.append(f"- Quyết định: Gọi tool {t_name}.")
    lines.append("</think>")
    return "\n".join(lines)


def _build_final_reasoning(last_tool_name: str | None) -> str:
    lines = ["<think>"]
    if last_tool_name == "sys.ask_missing_info":
        lines.append("- Kiểm tra hoàn thành: Đã gửi yêu cầu hỏi lại thông tin cần thiết tới người dùng.")
        lines.append("- Quyết định: Dừng gọi tool và chờ phản hồi từ người dùng.")
    elif last_tool_name is None:
        lines.append("- Ý định & Điều kiện: Yêu cầu ngoài phạm vi ứng dụng đã cài hoặc đã bị người dùng hủy bỏ.")
        lines.append("- Quyết định: Không gọi tool nào, trả lời kết thúc trực tiếp.")
    else:
        lines.append("- Kiểm tra hoàn thành: Tất cả các thao tác cần thiết đã được thực thi đầy đủ và hợp lệ.")
        lines.append("- Quyết định: Dừng gọi tool và thông báo hoàn thành cho người dùng.")
    lines.append("</think>")
    return "\n".join(lines)


def build_reasoning_trajectory(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a causal ReAct conversation with Structured Micro-Reasoning for each step."""
    history_transcripts = {}
    for turn in sample.get("history") or []:
        if turn.get("role") == "user" and turn.get("audio") and turn.get("audio_text"):
            history_transcripts[turn["audio"]] = turn["audio_text"]

    user_text = sample.get("audio_text", "")
    messages = build_initial_messages(
        sample=sample,
        final_user_text=user_text,
        catalog_tools_map=catalog_tools_map,
        history_transcripts=history_transcripts,
    )

    label = sample.get("label") or {}
    t_calls = label.get("tool_calls") or []
    t_results = label.get("tool_results") or []

    if not t_calls:
        think = _build_final_reasoning(None)
        messages.append(
            {
                "role": "assistant",
                "content": f"{think}\nYêu cầu không cần thực hiện thao tác trên ứng dụng.",
            }
        )
        return messages

    prev_res = None
    for idx, (tc, tr) in enumerate(zip(t_calls, t_results)):
        call_payload = {"name": tc["name"], "arguments": tc.get("arguments") or {}}
        think = _build_step_reasoning(
            step_idx=idx,
            total_steps=len(t_calls),
            tool_call=tc,
            catalog_tools_map=catalog_tools_map,
            user_text=user_text,
            prev_tool_result=prev_res,
        )
        call_block = f"<tool_call>\n{json.dumps(call_payload, ensure_ascii=False)}\n</tool_call>"
        messages.append({"role": "assistant", "content": f"{think}\n{call_block}"})
        messages.append({"role": "user", "content": f"<tool_response>\n{json.dumps(tr, ensure_ascii=False)}\n</tool_response>"})
        prev_res = tr

    # Final closing turn
    last_call_name = t_calls[-1]["name"]
    final_think = _build_final_reasoning(last_call_name)
    if last_call_name == "sys.ask_missing_info":
        messages.append(
            {
                "role": "assistant",
                "content": f"{final_think}\nVui lòng cung cấp thêm thông tin chi tiết để tôi thực hiện yêu cầu.",
            }
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "content": f"{final_think}\nĐã hoàn thành yêu cầu của bạn.",
            }
        )
    return messages


def main():
    ap = argparse.ArgumentParser(description="Tạo dữ liệu SFT ReAct có suy luận cấu trúc (Micro-Reasoning)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--val-size", type=int, default=145)
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
        traj = {"id": s["id"], "messages": build_reasoning_trajectory(s, catalog_tools_map)}
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
    print(f"Đã tạo {len(train_rows)} mẫu train có suy luận (Micro-Reasoning) -> {out_dir / 'sft_train.jsonl'}")
    print(f"Đã tạo {len(val_rows)} mẫu val có suy luận   -> {out_dir / 'sft_val.jsonl'}")


if __name__ == "__main__":
    main()
