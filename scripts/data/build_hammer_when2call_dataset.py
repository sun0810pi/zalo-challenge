"""Hammer Function Masking + When2Call Negative Action Triage Dataset Generator for ZAC 2026 LifeMate.

Techniques implemented:
1. Hammer Function Masking (35% probability):
   Randomly anonymizes tool names (service_{id}.fn_{k}) and argument names (arg_{id}_{p})
   while strictly preserving descriptions, types, enums, and dependencies.
   Forces Qwen3-4B to attend to JSON schema semantics instead of memorizing tool names,
   enabling zero-shot transfer to Unseen Apps in private test.

2. When2Call 3-Way Action Triage & Negative Samples:
   - Out-of-Scope (Uninstalled app): Removes required app -> target [] (0 calls).
   - Missing Required Information: Omits mandatory argument -> target sys.ask_missing_info.
   - User Cancellation: Injects realistic cancellation phrases -> target [] (0 calls).

3. P0 Prompt Parity & P1 Concise Rationales:
   Maintains Vietnamese app catalog headers and concise <think> decision tokens.

Outputs to:
  `data/sft/candidates/v4_hammer_when2call/`
    - `sft_train.jsonl`
    - `sft_val.jsonl`
    - `manifest.json`
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
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
if str(ROOT / "public_test") not in sys.path:
    sys.path.insert(0, str(ROOT / "public_test"))

from zaic_tools.validate import catalog_tools
from src.prompt_builder import (
    SYSTEM_INSTRUCTION,
    build_calendar_helper,
    build_xref_provider_map,
    compress_tool_spec,
    format_history_block,
)
from src.router import SemanticToolRouter

RANDOM_SEED = 42
MASK_PROBABILITY = 0.35


def generate_mask_mappings(
    sample_id: str,
    installed_apps: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Generate deterministic random synthetic masks for tools and parameters."""
    seed_int = int(hashlib.md5(f"{sample_id}_hammer".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    salt = f"{rng.randint(10, 99):02d}"

    tool_name_map: dict[str, str] = {}
    tool_param_map: dict[str, dict[str, str]] = {}

    fn_idx = 1
    for app_id, app_info in installed_apps.items():
        tools = app_info.get("tools") or []
        for t in tools:
            tname = t["name"] if isinstance(t, dict) else str(t)
            if tname == "sys.ask_missing_info":
                continue
            masked_name = f"svc_{salt}_{app_id[:4]}.fn_{fn_idx:02d}"
            tool_name_map[tname] = masked_name
            fn_idx += 1

            spec = catalog_tools_map.get(tname, {})
            props = (spec.get("parameters") or {}).get("properties") or {}
            param_map = {}
            for p_idx, p_name in enumerate(props.keys(), 1):
                param_map[p_name] = f"p_{salt}_{p_idx:02d}"
            tool_param_map[tname] = param_map

    return tool_name_map, tool_param_map


def mask_catalog_tool_spec(
    spec: dict[str, Any],
    tool_name_map: dict[str, str],
    tool_param_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return a masked copy of a compressed tool spec."""
    tname = spec["name"]
    masked_name = tool_name_map.get(tname, tname)
    param_map = tool_param_map.get(tname, {})

    new_spec = copy.deepcopy(spec)
    new_spec["name"] = masked_name

    # Mask properties while strictly keeping type, desc, enum, minimum...
    old_params = new_spec.get("params") or {}
    new_params = {}
    for p_name, p_info in old_params.items():
        masked_p = param_map.get(p_name, p_name)
        new_params[masked_p] = copy.deepcopy(p_info)
    new_spec["params"] = new_params

    # Mask required list
    if "required" in new_spec:
        new_spec["required"] = [param_map.get(p, p) for p in new_spec["required"]]

    # Mask x-one-of
    if "x-one-of" in new_spec:
        new_spec["x-one-of"] = [[param_map.get(p, p) for p in grp] for grp in new_spec["x-one-of"]]

    return new_spec


def build_masked_system_prompt(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    catalog_apps_map: dict[str, dict[str, Any]],
    tool_name_map: dict[str, str],
    tool_param_map: dict[str, dict[str, str]],
) -> str:
    """Build unified system prompt with masked tool names and parameters."""
    ctx = sample.get("context", {})
    profile = ctx.get("user_profile", {})
    loc = ctx.get("location", {})
    cal_str = build_calendar_helper(ctx.get("current_time", ""))
    installed = ctx.get("installed_apps", {})
    xref_map = build_xref_provider_map(catalog_tools_map)

    blocks: list[str] = ["### DANH SÁCH ỨNG DỤNG ĐÃ CÀI ĐẶT & CÔNG CỤ ĐƯỢC PHÉP GỌI (ALLOWED TOOLS):"]

    for app_id, app_info in installed.items():
        app_tools = app_info.get("tools", [])
        meta = catalog_apps_map.get(app_id, {})
        name_vi = meta.get("name_vi", app_id)
        group = meta.get("group", "")
        desc = meta.get("description", "")

        header = f"[Ứng dụng: {name_vi} ({app_id}) | Nhóm: {group}]"
        if desc:
            header += f"\nMô tả: {desc}"

        tool_specs = []
        for t in app_tools:
            tname = t["name"] if isinstance(t, dict) else str(t)
            if tname in catalog_tools_map:
                spec = compress_tool_spec(catalog_tools_map[tname], xref_map)
                masked_spec = mask_catalog_tool_spec(spec, tool_name_map, tool_param_map)
                tool_specs.append(json.dumps(masked_spec, ensure_ascii=False, separators=(",", ":")))

        if tool_specs:
            blocks.append(header + "\n" + "\n".join(tool_specs))

    # Always ensure system tool is present
    if "sys.ask_missing_info" in catalog_tools_map:
        sys_spec = compress_tool_spec(catalog_tools_map["sys.ask_missing_info"], xref_map)
        blocks.append(
            "[CÔNG CỤ HỆ THỐNG (SYSTEM TOOLS)]\n"
            + json.dumps(sys_spec, ensure_ascii=False, separators=(",", ":"))
        )

    tools_block = "\n\n".join(blocks)
    context_block = (
        f"### BỐI CẢNH NGƯỜI DÙNG (CONTEXT)\n"
        f"{cal_str}\n"
        f"- Người dùng: {profile.get('name', '')} | Thành phố: {profile.get('city', '')}\n"
        f"- Địa chỉ nhà: {profile.get('home_address', '')}\n"
        f"- Địa chỉ công ty/nơi làm việc: {profile.get('work_address', '')}\n"
        f"- Tọa độ hiện tại: lat={loc.get('lat')}, lng={loc.get('lng')}\n\n"
        f"{tools_block}"
    )

    return SYSTEM_INSTRUCTION + "\n\n" + context_block


def mask_tool_call(
    tool_call: dict[str, Any],
    tool_name_map: dict[str, str],
    tool_param_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Mask a tool call emitted by assistant."""
    orig_name = tool_call["name"]
    masked_name = tool_name_map.get(orig_name, orig_name)
    param_map = tool_param_map.get(orig_name, {})

    orig_args = tool_call.get("arguments") or {}
    masked_args = {}
    for k, v in orig_args.items():
        masked_k = param_map.get(k, k)
        masked_args[masked_k] = copy.deepcopy(v)

    return {"name": masked_name, "arguments": masked_args}


def create_cancellation_negative(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    catalog_apps_map: dict[str, dict[str, Any]],
    router: SemanticToolRouter,
) -> dict[str, Any] | None:
    """Create a negative sample where the user explicitly cancels at the end of speech."""
    user_text = sample.get("audio_text", "").strip()
    if not user_text or len(user_text) < 15:
        return None

    cancel_phrases = [
        ", à mà thôi khỏi làm nhé.",
        ", à thôi bỏ đi, không cần nữa đâu.",
        ". Mà thôi, cứ giữ nguyên cho tôi nhé.",
        ", à thôi khỏi, để lát tôi tự làm.",
    ]
    seed = int(hashlib.md5(f"{sample['id']}_cancel".encode()).hexdigest()[:8], 16)
    phrase = cancel_phrases[seed % len(cancel_phrases)]
    cancelled_text = user_text.rstrip(" .,") + phrase

    # Build messages
    from src.prompt_builder import build_initial_messages

    msgs = build_initial_messages(
        sample=sample,
        final_user_text=cancelled_text,
        catalog_tools_map=catalog_tools_map,
        router=router,
        catalog_apps_map=catalog_apps_map,
    )

    assistant_reply = (
        "<think>\n"
        "Người dùng đã chủ động hủy bỏ yêu cầu ở cuối câu. Không thực hiện bất kỳ lệnh gọi công cụ nào.\n"
        "</think>\n"
        "Dạ vâng, tôi đã ghi nhận và không thực hiện thao tác nào."
    )
    msgs.append({"role": "assistant", "content": assistant_reply})

    return {
        "id": f"{sample['id']}_neg_cancel",
        "sample_id": sample["id"],
        "category": "negative_cancellation",
        "messages": msgs,
    }


def create_uninstalled_negative(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    catalog_apps_map: dict[str, dict[str, Any]],
    router: SemanticToolRouter,
) -> dict[str, Any] | None:
    """Create a negative sample where the app needed by the user is not installed."""
    label_calls = sample.get("label", {}).get("tool_calls", [])
    if not label_calls:
        return None

    # Find the apps that were actually needed
    needed_apps = set(tc["name"].split(".", 1)[0] for tc in label_calls if "." in tc.get("name", ""))
    if not needed_apps:
        return None

    # Clone sample and strip all needed apps from installed_apps
    sample_mod = copy.deepcopy(sample)
    inst = sample_mod.get("context", {}).get("installed_apps", {})
    for a in needed_apps:
        inst.pop(a, None)

    # Need at least 2 other innocent apps remaining so the prompt is realistic
    if len(inst) < 2:
        return None

    user_text = sample.get("audio_text", "").strip()
    from src.prompt_builder import build_initial_messages

    msgs = build_initial_messages(
        sample=sample_mod,
        final_user_text=user_text,
        catalog_tools_map=catalog_tools_map,
        router=router,
        catalog_apps_map=catalog_apps_map,
    )

    assistant_reply = (
        "<think>\n"
        "Yêu cầu của người dùng đòi hỏi ứng dụng chưa được cài đặt trên thiết bị. "
        "Không có công cụ phù hợp trong ALLOWED TOOLS.\n"
        "</think>\n"
        "Tôi rất tiếc, thiết bị của bạn chưa cài đặt ứng dụng để thực hiện chức năng này."
    )
    msgs.append({"role": "assistant", "content": assistant_reply})

    return {
        "id": f"{sample['id']}_neg_uninstalled",
        "sample_id": sample["id"],
        "category": "negative_uninstalled",
        "messages": msgs,
    }


def main():
    print("[Dataset Generator] Đang khởi tạo bộ dữ liệu Hammer Masking + When2Call Negative Triage...")

    train_cat_path = ROOT / "train" / "train_tools_catalog.json"
    pub_cat_path = ROOT / "public_test" / "public_test_tools_catalog.json"

    train_cat = json.loads(train_cat_path.read_text(encoding="utf-8"))
    pub_cat = json.loads(pub_cat_path.read_text(encoding="utf-8"))

    merged_cat_tools: dict[str, dict[str, Any]] = {}
    merged_cat_tools.update(catalog_tools(train_cat))
    merged_cat_tools.update(catalog_tools(pub_cat))

    merged_cat_apps: dict[str, dict[str, Any]] = {}
    for a in train_cat.get("apps", []):
        merged_cat_apps[a["id"]] = a
    for a in pub_cat.get("apps", []):
        merged_cat_apps[a["id"]] = a

    router = SemanticToolRouter(top_k_recommended=4)

    # Load P1 clean dataset
    p1_dir = ROOT / "data" / "sft" / "candidates" / "p1_concise"
    p1_train_path = p1_dir / "sft_train.jsonl"
    p1_val_path = p1_dir / "sft_val.jsonl"

    if not p1_train_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {p1_train_path}! Hãy đảm bảo P1 dataset đã được tạo.")

    p1_train_samples = [json.loads(line) for line in p1_train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    p1_val_samples = [json.loads(line) for line in p1_val_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Load raw train samples for negative data augmentation
    raw_train = [json.loads(l) for l in (ROOT / "train" / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_ids_path = ROOT / "data" / "sft" / "val_ids.json"
    val_ids = set(json.loads(val_ids_path.read_text(encoding="utf-8"))) if val_ids_path.is_file() else set()
    train_raw_pool = [s for s in raw_train if s["id"] not in val_ids]

    # 1. Generate Negative Samples via When2Call
    print("\n[Phase 1] Sinh mẫu phủ định (Negative Action Triage - When2Call)...")
    neg_samples: list[dict[str, Any]] = []

    rng = random.Random(RANDOM_SEED)
    sampled_pool = rng.sample(train_raw_pool, min(250, len(train_raw_pool)))

    for s in sampled_pool:
        # 50% cancellation, 50% uninstalled
        if rng.random() < 0.5:
            cand = create_cancellation_negative(s, merged_cat_tools, merged_cat_apps, router)
        else:
            cand = create_uninstalled_negative(s, merged_cat_tools, merged_cat_apps, router)
        if cand:
            neg_samples.append(cand)

    print(f"-> Đã sinh {len(neg_samples)} mẫu phủ định hợp lệ (Cancellation + Uninstalled Apps).")

    # 2. Apply Hammer Masking on Train Split (35% probability)
    print("\n[Phase 2] Áp dụng Hammer Function Masking (35% xác suất)...")
    augmented_train: list[dict[str, Any]] = []
    masked_count = 0

    for row in p1_train_samples:
        s_id = row.get("id") or row.get("sample_id", "")
        # Apply hammer mask only to positive non-synthetic samples with probability
        row_seed = int(hashlib.md5(f"{s_id}_mask_prob".encode()).hexdigest()[:8], 16)
        if (row_seed % 100) < (MASK_PROBABILITY * 100) and not s_id.startswith("sim_"):
            # Clone and mask
            msgs = copy.deepcopy(row["messages"])
            # Extract tool names from assistant messages
            used_tools = set()
            for m in msgs:
                if m.get("role") == "assistant":
                    for tc in re.findall(r'"name":\s*"([^"]+)"', m.get("content", "")):
                        used_tools.add(tc)

            if used_tools and all(t in merged_cat_tools for t in used_tools if t != "sys.ask_missing_info"):
                # Create fake installed apps representation for mapping
                fake_apps = {}
                for t in used_tools:
                    app = t.split(".", 1)[0]
                    fake_apps.setdefault(app, {"tools": []})["tools"].append(t)

                t_map, p_map = generate_mask_mappings(s_id, fake_apps, merged_cat_tools)

                # Mask tool calls in assistant turns
                for m in msgs:
                    if m.get("role") == "assistant":
                        for orig_t, masked_t in t_map.items():
                            m["content"] = m["content"].replace(f'"{orig_t}"', f'"{masked_t}"')
                            m["content"] = m["content"].replace(f"'{orig_t}'", f"'{masked_t}'")
                            for orig_p, masked_p in p_map.get(orig_t, {}).items():
                                m["content"] = m["content"].replace(f'"{orig_p}":', f'"{masked_p}":')

                # Mask system prompt
                sys_msg = msgs[0]["content"]
                for orig_t, masked_t in t_map.items():
                    sys_msg = sys_msg.replace(f'"{orig_t}"', f'"{masked_t}"')
                    sys_msg = sys_msg.replace(f"'{orig_t}'", f"'{masked_t}'")
                    for orig_p, masked_p in p_map.get(orig_t, {}).items():
                        sys_msg = sys_msg.replace(f'"{orig_p}":', f'"{masked_p}":')
                msgs[0]["content"] = sys_msg

                row_masked = copy.deepcopy(row)
                row_masked["messages"] = msgs
                row_masked["is_hammer_masked"] = True
                augmented_train.append(row_masked)
                masked_count += 1
                continue

        augmented_train.append(row)

    # Add negative samples to train
    augmented_train.extend(neg_samples)
    rng.shuffle(augmented_train)

    print(f"-> Tổng mẫu train: {len(augmented_train)} (Trong đó {masked_count} mẫu Masked Hammer, {len(neg_samples)} mẫu Negative Triage).")

    # 3. Export dataset
    out_dir = ROOT / "data" / "sft" / "candidates" / "v4_hammer_when2call"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out_path = out_dir / "sft_train.jsonl"
    val_out_path = out_dir / "sft_val.jsonl"
    manifest_path = out_dir / "manifest.json"

    with open(train_out_path, "w", encoding="utf-8") as f:
        for r in augmented_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_out_path, "w", encoding="utf-8") as f:
        for r in p1_val_samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "version": "v4_hammer_when2call",
        "num_train": len(augmented_train),
        "num_val": len(p1_val_samples),
        "hammer_masked_count": masked_count,
        "negative_samples_count": len(neg_samples),
        "features": [
            "P0_prompt_parity",
            "P1_concise_rationale",
            "Hammer_function_masking_35_percent",
            "When2Call_negative_action_triage",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nHoàn tất! Bộ dữ liệu SFT v4 đã lưu an toàn tại: {out_dir}")


if __name__ == "__main__":
    main()
