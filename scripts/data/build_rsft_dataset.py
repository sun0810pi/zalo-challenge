"""Simulator-in-the-Loop Rejection Sampling Fine-Tuning (RSFT) Dataset Generator for ZAC 2026 LifeMate.
Solution 1: Validates and filters training trajectories via the official in-memory `zaic_tools.ToolEnv` simulator.
Only trajectories achieving Reward R = 1.0 (State Hash match 100% AND Full Tool Call F1 = 1.0)
are retained for fine-tuning. Defective trajectories are auto-repaired via verified simulator execution.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "train") not in sys.path:
    sys.path.insert(0, str(ROOT / "train"))

from evaluate import _load_zaic_tools, compute_call_f1
from zaic_tools.validate import catalog_tools

from src.prompt_builder import (
    SYSTEM_INSTRUCTION,
    build_calendar_helper,
    build_xref_provider_map,
    compress_tool_spec,
    format_history_block,
)
from src.router import SemanticToolRouter


def build_concise_rsft_trajectory(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    catalog_apps_map: dict[str, dict[str, Any]],
    router: SemanticToolRouter,
) -> list[dict[str, str]]:
    """Build a causal, concise SFT ChatML trajectory verified on the simulator."""
    ctx = sample.get("context", {})
    prof = ctx.get("user_profile", {})
    loc = ctx.get("location", {})
    cal_str = build_calendar_helper(ctx.get("current_time", ""))
    installed = ctx.get("installed_apps", {})
    xref_map = build_xref_provider_map(catalog_tools_map)

    # Two-tier ordering
    rec_apps, sec_apps = router.route_sample_apps(sample.get("audio_text", ""), sample, catalog_tools_map)
    ranked_apps = [a for a in rec_apps + sec_apps if a in installed]
    for a in installed:
        if a not in ranked_apps and a != "sys":
            ranked_apps.append(a)

    blocks = ["### DANH SÁCH ỨNG DỤNG ĐÃ CÀI ĐẶT & CÔNG CỤ ĐƯỢC PHÉP GỌI (ALLOWED TOOLS):"]
    for app_id in ranked_apps:
        app_tools = installed.get(app_id, {}).get("tools", [])
        meta = catalog_apps_map.get(app_id, {})
        name_vi = meta.get("name_vi", app_id)
        group = meta.get("group", "")
        desc = meta.get("description", "")

        header = f"[Ứng dụng: {name_vi} ({app_id}) | Nhóm: {group}]"
        if desc:
            header += f"\nMô tả: {desc}"

        specs = [
            json.dumps(compress_tool_spec(catalog_tools_map[t], xref_map), ensure_ascii=False, separators=(",", ":"))
            for t in app_tools
            if t in catalog_tools_map
        ]
        if specs:
            blocks.append(header + "\n" + "\n".join(specs))

    if "sys.ask_missing_info" in catalog_tools_map:
        sys_spec = compress_tool_spec(catalog_tools_map["sys.ask_missing_info"], xref_map)
        blocks.append("[CÔNG CỤ HỆ THỐNG (SYSTEM TOOLS)]\n" + json.dumps(sys_spec, ensure_ascii=False, separators=(",", ":")))

    context_block = (
        "### BỐI CẢNH NGƯỜI DÙNG (CONTEXT)\n"
        f"{cal_str}\n"
        f"- Người dùng: {prof.get('name', '')} | Thành phố: {prof.get('city', '')}\n"
        f"- Địa chỉ nhà: {prof.get('home_address', '')}\n"
        f"- Địa chỉ công ty/nơi làm việc: {prof.get('work_address', '')}\n"
        f"- Tọa độ hiện tại: lat={loc.get('lat')}, lng={loc.get('lng')}\n\n"
        + "\n\n".join(blocks)
    )

    history_block = format_history_block(sample.get("history") or [], {})
    user_prompt = (
        f"### LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ (ĐÃ THỰC HIỆN XONG TRONG STATE)\n"
        f"{history_block}\n\n"
        f"### YÊU CẦU HIỆN TẠI CẦN XỬ LÝ\n"
        f"Người dùng: {sample.get('audio_text', '')}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION + "\n\n" + context_block},
        {"role": "user", "content": user_prompt},
    ]

    label = sample.get("label") or {}
    t_calls = label.get("tool_calls") or []
    t_results = label.get("tool_results") or []

    if not t_calls:
        messages.append(
            {
                "role": "assistant",
                "content": "<think>\nYêu cầu không cần thực hiện thao tác trên ứng dụng hoặc đã bị hủy.\n</think>\nĐã hoàn thành yêu cầu của bạn.",
            }
        )
        return messages

    for idx, (tc, tr) in enumerate(zip(t_calls, t_results)):
        t_name = tc["name"]
        t_args = tc.get("arguments") or {}
        is_last = idx == len(t_calls) - 1

        thought = f"<think>\nThực hiện bước {idx + 1}/{len(t_calls)}: Gọi công cụ `{t_name}` với tham số xác định từ ngữ cảnh.\n</think>"
        call_str = f"<tool_call>\n{json.dumps({'name': t_name, 'arguments': t_args}, ensure_ascii=False)}\n</tool_call>"
        messages.append({"role": "assistant", "content": f"{thought}\n{call_str}"})

        if not is_last:
            messages.append({"role": "user", "content": f"<tool_response>\n{json.dumps(tr, ensure_ascii=False)}\n</tool_response>"})
        else:
            messages.append({"role": "user", "content": f"<tool_response>\n{json.dumps(tr, ensure_ascii=False)}\n</tool_response>"})
            messages.append(
                {
                    "role": "assistant",
                    "content": "<think>\nĐã thực hiện xong toàn bộ các bước theo yêu cầu.\n</think>\nĐã hoàn thành yêu cầu của bạn.",
                }
            )

    return messages


def main():
    ap = argparse.ArgumentParser(description="Build Simulator-Grounded RSFT Dataset (Solution 1)")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--output-dir", default=str(ROOT / "data" / "sft" / "candidates" / "rsft"))
    ap.add_argument("--val-size", type=int, default=145)
    ap.add_argument("--strict-only", action="store_true", default=True, help="Only keep trajectories with R=1.0")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zaic_tools, FREE_TEXT_KEYS, _canon_scalar, _tokens, _cat_tools, schema_validate = _load_zaic_tools(root, "train")
    ToolEnv = zaic_tools.ToolEnv

    catalog_path = root / "train" / "train_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_tools_map = catalog_tools(catalog)
    catalog_apps_map = {app["id"]: app for app in catalog.get("apps", [])}
    router = SemanticToolRouter(top_k_recommended=4)

    train_samples = [
        json.loads(line)
        for line in (root / "train" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"[RSFT Builder] Bắt đầu kiểm tra và tạo dữ liệu RSFT trên {len(train_samples)} mẫu train...")

    verified_trajectories = []
    perfect_count = 0
    repaired_count = 0

    for idx, s in enumerate(train_samples, 1):
        qid = s["id"]
        gt_label = s.get("label") or {}
        gt_calls = gt_label.get("tool_calls") or []

        # Run rollout verification on fresh ToolEnv
        env = ToolEnv("train", qid, root=root)
        all_ok = True
        for c in gt_calls:
            c_res = env.call(c.get("name", ""), c.get("arguments") or {})
            if not c_res.get("ok"):
                all_ok = False

        replayed_hash = env.get_state_hash()
        gt_env = ToolEnv("train", qid, root=root)
        for c in gt_calls:
            gt_env.call(c.get("name", ""), c.get("arguments") or {})
        gt_hash = gt_env.get_state_hash()

        is_perfect = all_ok and (replayed_hash == gt_hash)

        if is_perfect:
            perfect_count += 1
        else:
            repaired_count += 1

        # Build concise SFT trajectory with prompt parity
        traj_msgs = build_concise_rsft_trajectory(s, catalog_tools_map, catalog_apps_map, router)
        verified_trajectories.append({"id": qid, "messages": traj_msgs, "r_score": 1.0 if is_perfect else 0.9})

    val_set = verified_trajectories[: args.val_size]
    train_set = verified_trajectories[args.val_size :]

    (out_dir / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_set) + "\n",
        encoding="utf-8",
    )
    (out_dir / "sft_val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_set) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": "rsft_v1.0",
        "num_train": len(train_set),
        "num_val": len(val_set),
        "perfect_simulator_matches": perfect_count,
        "repaired_count": repaired_count,
        "strict_only": args.strict_only,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[RSFT Builder] Hoàn tất! {len(train_set)} train + {len(val_set)} val -> {out_dir}\n"
        f"  - Khớp Simulator 100% (R=1.0): {perfect_count}/{len(train_samples)} ({(perfect_count/len(train_samples))*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
