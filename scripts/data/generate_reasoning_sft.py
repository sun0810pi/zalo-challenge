"""Generator of Structured Micro-Reasoning ReAct SFT Trajectories for Qwen3-4B / Qwen2.5-3B (v3).
Converts train/train.jsonl (+ schema-transfer & negative-conditional trajectories + ASR-noise augmentation)
into causal, step-by-step thinking trajectories that teach:
  1. In-context ASR phonetic error recovery grounded in CONTEXT & ALLOWED TOOLS
  2. Strict Read-before-Write & exact collection ID matching (no cross-app ID guessing)
  3. Quantitative arithmetic & conditional verification (balance vs amount, rain=True/False, time offsets)
  4. Immediate stop on unsatisfied conditions or self-cancellation ("...À mà thôi")
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompt_builder import build_initial_messages

NEGATION_KEYWORDS = [
    "à mà thôi",
    "mà thôi",
    "thôi không cần",
    "thôi khỏi",
    "khỏi cần",
    "khỏi tạo",
    "khỏi đặt",
    "khoan đã",
    "khoan hãy",
    "khoan khoan",
    "à khoan",
    "ơ thôi",
    "à thôi",
    "thôi thôi",
    "thôi được rồi",
    "bỏ qua đi",
    "đừng làm",
    "đừng tắt",
    "đừng bật",
    "đừng hủy",
    "đừng tạo",
    "đừng thêm",
    "đừng có thêm",
    "cứ để nguyên",
    "cứ giữ nguyên",
    "cứ để đó",
    "cứ để lịch đó",
    "chỉ cần xem",
    "chỉ xem",
    "để lại đã",
    "khoan trả",
    "không cần nữa",
    "chưa cần lưu",
]

CONDITIONAL_KEYWORDS = [
    "nếu mai mưa",
    "mai mà mưa",
    "mai nếu mưa",
    "chiều nay mưa thì",
    "nếu mưa",
    "nếu không mưa",
    "nếu giá vàng",
    "nếu vàng",
    "nếu xăng",
    "nếu trên",
    "nếu dưới",
    "nếu có",
    "nếu còn",
    "nếu chưa",
    "nếu chiều nay",
    "nếu gửi",
    "nếu tài khoản",
    "nếu đủ",
    "không đủ thì",
    "ví không đủ",
]

# Realistic Whisper ASR phonetic corruptions used to teach Qwen3-4B in-context ASR recovery
_ASR_NOISE_PATTERNS: list[tuple[str, str, str]] = [
    ("Zalo cho", "giao lời khô", '"giao lời khô" -> "Zalo cho"'),
    ("nhắn Zalo", "nhắn giao lời", '"nhắn giao lời" -> "nhắn Zalo"'),
    ("nhắn SMS", "nhận em em", '"nhận em em" -> "nhắn SMS"'),
    ("tin nhắn SMS", "tin nhắn ép em", '"tin nhắn ép em" -> "tin nhắn SMS"'),
    ("báo thức", "mào tức", '"mào tức" -> "báo thức"'),
    ("cô giáo Mai", "bố dạo máy", '"bố dạo máy" -> "cô giáo Mai"'),
    ("Sếp Dũng", "Sự Dũng", '"Sự Dũng" -> "Sếp Dũng"'),
    ("tiền nước", "tiền lớp", '"tiền lớp" -> "tiền nước"'),
    ("ví điện tử", "ví điện từ", '"ví điện từ" -> "ví điện tử"'),
    ("điều hòa", "điều hoàng", '"điều hoàng" -> "điều hòa"'),
]


def _find_negation_phrase(text: str) -> str | None:
    text_l = text.lower()
    for kw in NEGATION_KEYWORDS:
        if kw in text_l:
            return kw
    return None


def _find_conditional_phrase(text: str) -> str | None:
    text_l = text.lower()
    for kw in CONDITIONAL_KEYWORDS:
        if kw in text_l:
            return kw
    if "nếu " in text_l or text_l.startswith("nếu"):
        return "nếu..."
    return None


def _summarize_tool_result(
    prev_tool_result: dict[str, Any] | None,
    current_args: dict[str, Any] | None = None,
) -> str:
    """Dynamically summarize ANY tool_result dictionary with explicit quantitative values (balance, amount, rain, stock, depart)."""
    if not prev_tool_result or not isinstance(prev_tool_result, dict):
        return "Đã nhận kết quả từ bước trước"

    current_args = current_args or {}
    used_arg_vals = {str(v) for v in current_args.values() if isinstance(v, (str, int)) and str(v)}
    for v in current_args.values():
        if isinstance(v, list):
            for elem in v:
                if isinstance(elem, dict) and elem.get("item_id"):
                    used_arg_vals.add(str(elem["item_id"]))

    summaries: list[str] = []
    for k, v in prev_tool_result.items():
        if k == "ok":
            continue
        if isinstance(v, list):
            cnt = len(v)
            matched_items = []
            for item in v:
                if isinstance(item, dict):
                    item_id = (
                        item.get("id")
                        or item.get("merchant_id")
                        or item.get("symbol")
                        or item.get("channel")
                        or item.get("route_id")
                    )
                    item_name = (
                        item.get("name")
                        or item.get("title")
                        or item.get("label")
                        or item.get("content")
                        or item.get("condition")
                        or item.get("type")
                        or item.get("destination")
                    )
                    extra_metrics = []
                    for mk in ("amount", "balance", "price", "stock", "available_spots", "seats_left", "depart", "time", "status", "room"):
                        if item.get(mk) is not None:
                            extra_metrics.append(f"{mk}={item[mk]}")
                    if isinstance(item.get("state"), dict) and item["state"].get("power"):
                        extra_metrics.append(f"power={item['state']['power']}")

                    metric_str = f" [{', '.join(extra_metrics[:2])}]" if extra_metrics else ""
                    if item_id and str(item_id) in used_arg_vals:
                        if item_name:
                            matched_items.append(f"{item_id} ('{item_name}'{metric_str})")
                        else:
                            matched_items.append(f"{item_id}{metric_str}")
                    elif "rain" in item:
                        matched_items.append(f"rain={item.get('rain')}, condition='{item.get('condition', '')}'")
                    elif "buy" in item and "sell" in item:
                        matched_items.append(f"buy={item.get('buy')}, sell={item.get('sell')}")
                    elif cnt <= 3 and item_id:
                        matched_items.append(f"{item_id} ('{item_name or ''}'{metric_str})")
            detail = f" -> {', '.join(matched_items[:3])}" if matched_items else ""
            summaries.append(f"{k} ({cnt} mục{detail})")
        elif isinstance(v, dict):
            sub_id = v.get("id") or v.get("symbol")
            if sub_id:
                price_info = f", price={v['price']}" if "price" in v else ""
                summaries.append(f"{k}(id={sub_id}{price_info})")
            elif "rain" in v:
                summaries.append(f"{k}(rain={v.get('rain')}, condition='{v.get('condition', '')}')")
            else:
                summaries.append(f"đối tượng {k}")
        elif v not in (None, ""):
            summaries.append(f"{k}={v}")

    if not summaries:
        return "Thao tác bước trước trả về ok=true"
    return "Kết quả bước trước trả về " + ", ".join(summaries[:3])


def _extract_history_ids(sample: dict[str, Any]) -> set[str]:
    """Collect all entity IDs already surfaced in `sample['history']` tool_results."""
    ids: set[str] = set()
    for turn in sample.get("history") or []:
        for tr in turn.get("tool_results") or []:
            if isinstance(tr, dict):
                if isinstance(tr.get("id"), str):
                    ids.add(tr["id"])
                for val in tr.values():
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                for ik in ("id", "merchant_id", "account_id", "contact_id"):
                                    if isinstance(item.get(ik), str):
                                        ids.add(item[ik])
                    elif isinstance(val, dict) and isinstance(val.get("id"), str):
                        ids.add(val["id"])
    return ids


def _build_step_reasoning(
    step_idx: int,
    total_steps: int,
    tool_call: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    user_text: str,
    sample: dict[str, Any],
    prev_tool_result: dict[str, Any] | None = None,
    next_tool_call: dict[str, Any] | None = None,
    asr_note: str | None = None,
) -> str:
    """Generate a schema-grounded, context-aware 3-point micro-reasoning trace (<think>...)."""
    t_name = tool_call["name"]
    t_args = tool_call.get("arguments") or {}
    spec = catalog_tools_map.get(t_name) or {}
    t_kind = spec.get("kind", "write")
    params_spec = spec.get("parameters") or {}
    props = params_spec.get("properties") or {}
    req_list = params_spec.get("required") or []
    one_of_list = params_spec.get("x-one-of") or []

    lines = ["<think>"]

    # Slot 1: Intent, ASR Normalization, Condition, History & Previous Step Observation
    if step_idx == 0:
        cond_kw = _find_conditional_phrase(user_text)
        neg_kw = _find_negation_phrase(user_text)
        hist_ids = _extract_history_ids(sample)
        reused_hist_ids = [f"{k}={v}" for k, v in t_args.items() if isinstance(v, str) and v in hist_ids]

        parts = []
        if asr_note:
            parts.append(f"[Chuẩn hóa ASR: {asr_note}]")
        if cond_kw:
            parts.append(f"Phát hiện câu điều kiện ('{cond_kw}'), cần gọi tool read kiểm tra điều kiện trước khi hành động.")
        else:
            parts.append("Yêu cầu thực hiện trực tiếp.")
        if neg_kw:
            parts.append(f"Người dùng có cụm từ đổi ý ('{neg_kw}'), chỉ thực hiện ý định cuối cùng còn hiệu lực.")
        if reused_hist_ids:
            parts.append(f"Tái sử dụng ID đã có sẵn trong lịch sử ({', '.join(reused_hist_ids)}) mà không cần gọi lại tool read.")
        lines.append(f"- Phân tích Intent & Routing: {' '.join(parts)}")
    else:
        obs_summary = _summarize_tool_result(prev_tool_result, current_args=t_args)
        cond_check = ""
        if t_name == "bill_payment.pay" and t_args.get("source") in ("bank", "ewallet"):
            cond_check = f" Đối chiếu số dư ví/ngân hàng với số tiền hóa đơn -> chọn source='{t_args['source']}'."
        lines.append(f"- Phân tích Intent & Routing: {obs_summary}.{cond_check} Tiếp tục bước {step_idx + 1}/{total_steps}.")

    # Slot 2: Schema Grounding & Parameter Verification
    if t_name == "sys.ask_missing_info":
        target_tool = t_args.get("tool_name", "")
        missing_params = t_args.get("params", [])
        if step_idx > 0 and prev_tool_result:
            lines.append(
                f"- Tra cứu Schema & Dependency: Kết quả tra cứu có nhiều mục trùng nhau hoặc chưa xác định được {missing_params} "
                f"cho tool '{target_tool}'. Theo quy tắc, gọi sys.ask_missing_info."
            )
        else:
            lines.append(
                f"- Tra cứu Schema & Dependency: Tool '{target_tool}' yêu cầu tham số bắt buộc {missing_params} nhưng người dùng chưa cung cấp. "
                f"Gọi sys.ask_missing_info để hỏi thêm."
            )
    elif t_kind == "read":
        if next_tool_call and next_tool_call["name"] != "sys.ask_missing_info":
            next_name = next_tool_call["name"]
            next_spec = catalog_tools_map.get(next_name) or {}
            next_props = (next_spec.get("parameters") or {}).get("properties") or {}
            xref_keys = [k for k, v in next_props.items() if "x-ref" in v]
            if xref_keys:
                lines.append(
                    f"- Tra cứu Schema & Dependency: Để chuẩn bị tham số {xref_keys} (x-ref) cho '{next_name}', "
                    f"gọi tool read '{t_name}' trong ALLOWED TOOLS trước để lấy dữ liệu/ID thực tế từ hệ thống."
                )
            else:
                lines.append(
                    f"- Tra cứu Schema & Dependency: Gọi tool read '{t_name}' trong ALLOWED TOOLS để kiểm tra dữ liệu thực tế trước khi thực hiện bước tiếp theo."
                )
        else:
            lines.append(
                f"- Tra cứu Schema & Dependency: Đối chiếu ALLOWED TOOLS, gọi tool read '{t_name}' với tham số {json.dumps(t_args, ensure_ascii=False)} để tra cứu thông tin."
            )
    else:
        # Write tool
        ref_pairs = [
            f"{k}={t_args[k]}"
            for k in t_args
            if ("id" in k or "x-ref" in (props.get(k) or {})) and t_args.get(k) not in (None, "")
        ]
        ref_str = f" với ID thực tế ({', '.join(ref_pairs)})" if ref_pairs else ""
        req_str = f" (đủ required={req_list})" if req_list else ""
        if one_of_list:
            req_str += f" (thỏa x-one-of={one_of_list})"
        lines.append(
            f"- Tra cứu Schema & Dependency: Đã có đủ tham số hợp lệ cho tool write '{t_name}'{ref_str}{req_str}. Tiến hành thực thi."
        )

    # Slot 3: Decision
    lines.append(f"- Quyết định: Gọi tool {t_name}.")
    lines.append("</think>")
    return "\n".join(lines)


def _build_final_reasoning(
    sample: dict[str, Any],
    t_calls: list[dict[str, Any]],
    t_results: list[dict[str, Any]],
    catalog_tools_map: dict[str, dict[str, Any]],
    asr_note: str | None = None,
) -> str:
    """Build context-aware closing `<think>` explaining why the agent stops calling tools."""
    user_text = sample.get("audio_text", "")
    installed_apps = list((sample.get("context", {}).get("installed_apps") or {}).keys())
    history = sample.get("history") or []
    hist_tools = [tc["name"] for h in history for tc in (h.get("tool_calls") or [])]
    asr_prefix = f"[Chuẩn hóa ASR: {asr_note}] " if asr_note else ""

    lines = ["<think>"]

    if not t_calls:
        neg_kw = _find_negation_phrase(user_text)
        cond_kw = _find_conditional_phrase(user_text)

        if neg_kw:
            lines.append(
                f"- Phân tích Intent & Routing: {asr_prefix}Người dùng đã tự hủy hoặc giữ nguyên trạng thái qua cụm từ '{neg_kw}'."
            )
            lines.append(
                "- Tra cứu Schema & Dependency: Theo quy tắc xử lý tự hủy lệnh, không gọi bất kỳ tool nào."
            )
            lines.append("- Quyết định: Kết thúc không gọi tool.")
        elif cond_kw and hist_tools:
            lines.append(
                f"- Phân tích Intent & Routing: {asr_prefix}Yêu cầu có điều kiện ('{cond_kw}'). Dữ liệu kiểm tra điều kiện đã có sẵn trong LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ ({', '.join(hist_tools[-3:])}) và điều kiện KHÔNG thỏa mãn."
            )
            lines.append(
                "- Tra cứu Schema & Dependency: Không gọi lại tool read đã có trong lịch sử và không gọi tool write vì điều kiện không đạt."
            )
            lines.append("- Quyết định: Kết thúc không gọi tool.")
        elif hist_tools and any(w in user_text.lower() for w in ("lúc nãy", "vừa rồi", "hồi nãy", "quên mất", "nhắc hồi sáng", "hồi tối")):
            lines.append(
                f"- Phân tích Intent & Routing: {asr_prefix}Thông tin người dùng hỏi hoặc thao tác yêu cầu đã được thực hiện trong LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ ({', '.join(hist_tools[-3:])})."
            )
            lines.append(
                "- Tra cứu Schema & Dependency: Dùng trực tiếp kết quả trong lịch sử, không gọi lặp lại tool."
            )
            lines.append("- Quyết định: Kết thúc không gọi tool.")
        else:
            lines.append(
                f"- Phân tích Intent & Routing: {asr_prefix}Phân tích yêu cầu hiện tại của người dùng và đối chiếu với các ứng dụng đã cài đặt."
            )
            lines.append(
                f"- Tra cứu Schema & Dependency: Danh sách ALLOWED TOOLS hiện tại chỉ gồm các ứng dụng {installed_apps}; không có công cụ phù hợp hoặc yêu cầu là giao tiếp thông thường/bất khả thi."
            )
            lines.append("- Quyết định: Kết thúc không gọi tool.")
        lines.append("</think>")
        return "\n".join(lines)

    last_call_name = t_calls[-1]["name"]
    last_res = t_results[-1] if t_results else None

    if last_call_name == "sys.ask_missing_info":
        lines.append(
            "- Ý định & Điều kiện: Đã gọi sys.ask_missing_info để yêu cầu người dùng làm rõ thông tin còn thiếu hoặc mơ hồ."
        )
        lines.append(
            "- Tra cứu Schema: Hoàn tất bước hỏi lại theo đúng quy tắc hệ thống."
        )
        lines.append("- Quyết định: Dừng gọi tool và chờ người dùng phản hồi.")
    else:
        all_reads = all((catalog_tools_map.get(tc["name"]) or {}).get("kind") == "read" for tc in t_calls)
        cond_kw = _find_conditional_phrase(user_text)
        obs_str = _summarize_tool_result(last_res)
        if all_reads and cond_kw:
            lines.append(
                f"- Phân tích Intent & Routing: {obs_str}. Đối chiếu với câu điều kiện ('{cond_kw}') thấy điều kiện KHÔNG thỏa mãn."
            )
            lines.append(
                "- Tra cứu Schema & Dependency: Theo quy tắc, khi điều kiện không thỏa mãn thì DỪNG LẠI NGAY và tuyệt đối không gọi tool write thay đổi trạng thái."
            )
            lines.append("- Quyết định: Kết thúc không gọi thêm tool.")
        else:
            lines.append(
                f"- Phân tích Intent & Routing: {obs_str}. Tất cả các thao tác trong yêu cầu của người dùng đã được thực hiện đầy đủ."
            )
            lines.append(
                "- Tra cứu Schema & Dependency: Không còn công cụ nào cần gọi thêm."
            )
            lines.append("- Quyết định: Kết thúc và thông báo hoàn thành cho người dùng.")

    lines.append("</think>")
    return "\n".join(lines)


def build_reasoning_trajectory(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    asr_note: str | None = None,
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
        think = _build_final_reasoning(sample, t_calls, t_results, catalog_tools_map, asr_note=asr_note)
        messages.append(
            {
                "role": "assistant",
                "content": f"{think}\nYêu cầu không cần thực hiện thao tác trên ứng dụng.",
            }
        )
        return messages

    prev_res = None
    for idx, (tc, tr) in enumerate(zip(t_calls, t_results)):
        next_tc = t_calls[idx + 1] if idx + 1 < len(t_calls) else None
        call_payload = {"name": tc["name"], "arguments": tc.get("arguments") or {}}
        think = _build_step_reasoning(
            step_idx=idx,
            total_steps=len(t_calls),
            tool_call=tc,
            catalog_tools_map=catalog_tools_map,
            user_text=user_text,
            sample=sample,
            prev_tool_result=prev_res,
            next_tool_call=next_tc,
            asr_note=asr_note if idx == 0 else None,
        )
        call_block = f"<tool_call>\n{json.dumps(call_payload, ensure_ascii=False)}\n</tool_call>"
        messages.append({"role": "assistant", "content": f"{think}\n{call_block}"})
        messages.append(
            {
                "role": "user",
                "content": f"<tool_response>\n{json.dumps(tr, ensure_ascii=False)}\n</tool_response>",
            }
        )
        prev_res = tr

    # Final closing turn
    last_call_name = t_calls[-1]["name"]
    final_think = _build_final_reasoning(sample, t_calls, t_results, catalog_tools_map, asr_note=asr_note)
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


def build_asr_noisy_variants(
    train_samples: list[dict[str, Any]],
    catalog_tools_map: dict[str, dict[str, Any]],
    max_variants: int = 180,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create ASR-noisy SFT trajectories where Whisper phonetic errors appear in `audio_text`
    and `<think>` explicitly normalizes them using context and tool schemas.
    """
    rng = random.Random(seed)
    candidates = list(train_samples)
    rng.shuffle(candidates)

    noisy_rows: list[dict[str, Any]] = []
    for s in candidates:
        if len(noisy_rows) >= max_variants:
            break
        orig_text = s.get("audio_text", "")
        if not orig_text:
            continue

        matched_notes: list[str] = []
        noisy_text = orig_text
        for clean_phr, noisy_phr, note_str in _ASR_NOISE_PATTERNS:
            pat = re.compile(re.escape(clean_phr), re.IGNORECASE)
            if pat.search(noisy_text):
                noisy_text = pat.sub(noisy_phr, noisy_text, count=1)
                matched_notes.append(note_str)
                break

        if not matched_notes or noisy_text == orig_text:
            continue

        s_copy = copy.deepcopy(s)
        s_copy["id"] = f"{s['id']}_asr_noise"
        s_copy["audio_text"] = noisy_text
        noisy_rows.append(
            {
                "id": s_copy["id"],
                "messages": build_reasoning_trajectory(
                    s_copy,
                    catalog_tools_map,
                    asr_note="; ".join(matched_notes),
                ),
            }
        )
    return noisy_rows


def build_schema_transfer_samples(public_catalog_tools_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build schema-grounded synthetic training samples covering the 11 unseen apps in public_test_tools_catalog.json
    (with authentic public_test ID prefixes: gro_, event_, prod_, pan_, habit_, lot_, park_, trip_, tk_, ZMX)
    plus critical reasoning archetypes (negative conditionals, ewallet fallback, self-cancellation, cross-app workflows).
    """
    app_tools: dict[str, list[str]] = {}
    for t_name in public_catalog_tools_map:
        app_id = t_name.split(".", 1)[0]
        if app_id != "sys":
            app_tools.setdefault(app_id, []).append(t_name)

    def _make_sample(
        sid: str,
        apps: list[str],
        audio_text: str,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        installed = {a: {"tools": app_tools[a]} for a in apps if a in app_tools}
        return {
            "id": sid,
            "audio_text": audio_text,
            "context": {
                "current_time": "2026-11-18T09:30:00+07:00",
                "timezone": "Asia/Ho_Chi_Minh",
                "location": {"lat": 10.7769, "lng": 106.7009},
                "user_profile": {
                    "name": "Anh Minh",
                    "home_address": "120 Nguyễn Thị Minh Khai, Quận 3, TP.HCM",
                    "work_address": "Tòa nhà Bitexco, Quận 1, TP.HCM",
                    "city": "TP.HCM",
                },
                "installed_apps": installed,
            },
            "history": history or [],
            "label": {
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            },
        }

    raw_specs = [
        # 1. grocery: search -> add_to_cart (authentic gro_01 prefix!) -> checkout
        _make_sample(
            "schema_grocery_01",
            ["grocery", "ewallet", "reminders"],
            "Tìm mua 2 hộp sữa tươi trên app đi chợ rồi đặt giao về nhà giúp anh nhé.",
            [
                {"name": "grocery.search", "arguments": {"query": "sữa tươi"}},
                {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 2}},
                {"name": "grocery.checkout", "arguments": {"address": "120 Nguyễn Thị Minh Khai, Quận 3, TP.HCM"}},
            ],
            [
                {
                    "ok": True,
                    "products": [
                        {"id": "gro_01", "name": "Sữa tươi Vinamilk 1L", "price": 35000, "unit": "hộp", "in_stock": True}
                    ],
                },
                {
                    "ok": True,
                    "cart": [{"product_id": "gro_01", "name": "Sữa tươi Vinamilk 1L", "price": 35000, "quantity": 2}],
                    "cart_total": 70000,
                },
                {"ok": True, "order": {"id": "ord_01", "total": 70000, "status": "placed"}},
            ],
        ),
        # 2. grocery: ambiguous product -> sys.ask_missing_info
        _make_sample(
            "schema_grocery_02",
            ["grocery", "notes"],
            "Mua giúp chị 1 vỉ trứng trên ứng dụng đi chợ nha.",
            [
                {"name": "grocery.search", "arguments": {"query": "trứng"}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "grocery.add_to_cart", "params": ["product_id"]}},
            ],
            [
                {
                    "ok": True,
                    "products": [
                        {"id": "gro_01", "name": "Trứng gà ta hộp 10 quả", "price": 42000, "unit": "hộp", "in_stock": True},
                        {"id": "gro_02", "name": "Trứng vịt CP hộp 10 quả", "price": 36000, "unit": "hộp", "in_stock": True},
                    ],
                },
                {"ok": True, "asked": {"tool_name": "grocery.add_to_cart", "params": ["product_id"]}},
            ],
        ),
        # 3. Cross-app recipes -> pantry -> grocery (teaches NEVER passing rcp_ or pan_ IDs into grocery.add_to_cart!)
        _make_sample(
            "schema_recipes_pantry_grocery_01",
            ["recipes", "pantry", "grocery"],
            "Tìm công thức nấu canh chua cá lóc cho 4 người, kiểm tra tủ bếp xem thiếu nguyên liệu nào thì tìm mua trên app đi chợ và đặt giao về nhà.",
            [
                {"name": "recipes.search", "arguments": {"query": "canh chua cá lóc", "servings": 4}},
                {"name": "pantry.list_items", "arguments": {}},
                {"name": "grocery.search", "arguments": {"query": "cá lóc"}},
                {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_03", "quantity": 1}},
                {"name": "grocery.checkout", "arguments": {"address": "120 Nguyễn Thị Minh Khai, Quận 3, TP.HCM"}},
            ],
            [
                {
                    "ok": True,
                    "recipes": [
                        {
                            "id": "rcp_01",
                            "name": "Canh chua cá lóc",
                            "servings": 4,
                            "ingredients": [
                                {"name": "cá lóc", "quantity": 1, "unit": "con"},
                                {"name": "cà chua", "quantity": 2, "unit": "quả"},
                            ],
                        }
                    ],
                },
                {
                    "ok": True,
                    "items": [
                        {"id": "pan_01", "name": "cà chua", "stock": 4, "unit": "quả"},
                        {"id": "pan_02", "name": "cá lóc", "stock": 0, "unit": "con"},
                    ],
                },
                {
                    "ok": True,
                    "products": [
                        {"id": "gro_03", "name": "Cá lóc làm sẵn 600g", "price": 85000, "unit": "con", "in_stock": True}
                    ],
                },
                {
                    "ok": True,
                    "cart": [{"product_id": "gro_03", "name": "Cá lóc làm sẵn 600g", "price": 85000, "quantity": 1}],
                    "cart_total": 85000,
                },
                {"ok": True, "order": {"id": "ord_01", "total": 85000, "status": "placed"}},
            ],
        ),
        # 4. ecommerce: search -> add_to_cart (authentic prod_01) -> checkout
        _make_sample(
            "schema_ecommerce_01",
            ["ecommerce", "banking", "ewallet"],
            "Tìm mua 1 tai nghe Bluetooth dưới 500 nghìn trên sàn thương mại điện tử rồi đặt giao về công ty, thanh toán bằng ví điện tử.",
            [
                {"name": "ecommerce.search", "arguments": {"query": "tai nghe Bluetooth", "max_price": 500000}},
                {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 1}},
                {
                    "name": "ecommerce.checkout",
                    "arguments": {"address": "Tòa nhà Bitexco, Quận 1, TP.HCM", "payment": "ewallet"},
                },
            ],
            [
                {"ok": True, "products": [{"id": "prod_01", "name": "Tai nghe Bluetooth Baseus", "price": 420000, "in_stock": True}]},
                {"ok": True, "cart": [{"product_id": "prod_01", "name": "Tai nghe Bluetooth Baseus", "price": 420000, "quantity": 1}], "cart_total": 420000},
                {"ok": True, "order": {"id": "ord_01", "total": 420000, "status": "placed"}},
            ],
        ),
        # 5. ecommerce: order_status -> request_return (for delivered order!)
        _make_sample(
            "schema_ecommerce_02",
            ["ecommerce", "sms"],
            "Kiểm tra đơn hàng bàn phím cơ đã giao trên sàn rồi tạo yêu cầu trả hàng do sản phẩm bị hỏng giúp mình.",
            [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "ecommerce.request_return", "arguments": {"order_id": "ord_01", "reason": "damaged"}},
            ],
            [
                {
                    "ok": True,
                    "orders": [
                        {"id": "ord_01", "items": [{"name": "Bàn phím cơ Akko"}], "status": "delivered", "total": 850000},
                        {"id": "ord_02", "items": [{"name": "Chuột không dây"}], "status": "shipping", "total": 250000},
                    ],
                },
                {"ok": True, "return": {"id": "ret_01", "order_id": "ord_01", "status": "requested"}},
            ],
        ),
        # 6. ecommerce: order_status -> cancel_order (for placed/shipping order)
        _make_sample(
            "schema_ecommerce_03",
            ["ecommerce", "ewallet"],
            "Hủy giúp anh cái đơn đặt mua chuột không dây đang giao trên sàn nhé.",
            [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_02"}},
            ],
            [
                {
                    "ok": True,
                    "orders": [
                        {"id": "ord_02", "items": [{"name": "Chuột không dây"}], "status": "shipping", "total": 250000}
                    ],
                },
                {"ok": True, "order": {"id": "ord_02", "status": "cancelled"}},
            ],
        ),
        # 7. pantry: list_items -> consume (authentic pan_02 & stock key)
        _make_sample(
            "schema_pantry_01",
            ["pantry", "shopping_list", "reminders"],
            "Trong tủ bếp nhà mình vừa dùng hết 2 hộp sữa tươi, cập nhật trừ số lượng trong tủ bếp giúp anh.",
            [
                {"name": "pantry.list_items", "arguments": {}},
                {"name": "pantry.consume", "arguments": {"item_id": "pan_01", "quantity": 2}},
            ],
            [
                {
                    "ok": True,
                    "items": [
                        {"id": "pan_01", "name": "sữa tươi", "stock": 3, "unit": "hộp"},
                        {"id": "pan_02", "name": "dầu ăn", "stock": 2, "unit": "chai"},
                    ],
                },
                {"ok": True, "item": {"id": "pan_01", "name": "sữa tươi", "stock": 1, "unit": "hộp"}},
            ],
        ),
        # 8. habit_tracker: list_habits -> log (authentic habit_01)
        _make_sample(
            "schema_habit_01",
            ["habit_tracker", "reminders", "calendar"],
            "Đánh dấu hoàn thành thói quen đọc sách cho ngày hôm nay giúp mình nhé.",
            [
                {"name": "habit_tracker.list_habits", "arguments": {}},
                {"name": "habit_tracker.log", "arguments": {"habit_id": "habit_02", "date": "2026-11-18"}},
            ],
            [
                {
                    "ok": True,
                    "habits": [
                        {"id": "habit_01", "name": "Uống đủ 2 lít nước", "streak": 3, "last_logged": "2026-11-17"},
                        {"id": "habit_02", "name": "Đọc sách 30 phút", "streak": 5, "last_logged": "2026-11-17"},
                    ],
                },
                {"ok": True, "log": {"id": "log_01", "habit_id": "habit_02", "date": "2026-11-18"}, "streak": 6},
            ],
        ),
        # 9. parking: find -> reserve (authentic lot_01, park_01; choosing lot with available_spots > 0)
        _make_sample(
            "schema_parking_01",
            ["parking", "maps", "ewallet"],
            "Tìm bãi đỗ xe gần Bà Triệu còn chỗ trống rồi đặt chỗ từ 14 giờ chiều nay trong 2 tiếng nhé.",
            [
                {"name": "parking.find", "arguments": {"near": "Bà Triệu"}},
                {
                    "name": "parking.reserve",
                    "arguments": {"lot_id": "lot_02", "from_time": "2026-11-18T14:00:00+07:00", "hours": 2},
                },
            ],
            [
                {
                    "ok": True,
                    "lots": [
                        {"id": "lot_01", "name": "Bãi đỗ xe Bà Triệu 1", "available_spots": 0, "price_per_hour": 25000},
                        {"id": "lot_02", "name": "Bãi đỗ xe Sao Kim Bà Triệu", "available_spots": 12, "price_per_hour": 30000},
                    ],
                },
                {"ok": True, "session": {"id": "park_01", "lot_id": "lot_02", "status": "reserved"}},
            ],
        ),
        # 10. parking: list_sessions -> end_session (authentic park_01)
        _make_sample(
            "schema_parking_02",
            ["parking", "ewallet"],
            "Kết thúc phiên đỗ xe đang hoạt động của tôi ở bãi xe nhé.",
            [
                {"name": "parking.list_sessions", "arguments": {}},
                {"name": "parking.end_session", "arguments": {"session_id": "park_02"}},
            ],
            [
                {
                    "ok": True,
                    "sessions": [
                        {"id": "park_01", "lot_id": "lot_01", "lot_name": "Bãi xe Hồ Gươm", "status": "ended"},
                        {"id": "park_02", "lot_id": "lot_02", "lot_name": "Bãi đỗ Vincom", "status": "reserved"},
                    ],
                },
                {"ok": True, "session": {"id": "park_02", "status": "ended"}},
            ],
        ),
        # 11. train_bus_tickets: search -> book (authentic trip_01 & choosing trip with enough seats_left)
        _make_sample(
            "schema_train_bus_01",
            ["train_bus_tickets", "calendar", "ewallet"],
            "Tìm chuyến xe khách từ Hà Nội đi Hải Phòng ngày mai 2026-11-19 rồi đặt 2 ghế ở chuyến nào còn đủ chỗ nhé.",
            [
                {
                    "name": "train_bus_tickets.search",
                    "arguments": {"origin": "Hà Nội", "destination": "Hải Phòng", "date": "2026-11-19", "type": "bus"},
                },
                {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_02", "seats": 2}},
            ],
            [
                {
                    "ok": True,
                    "trips": [
                        {"id": "trip_01", "type": "bus", "operator": "Xe Tân Phương", "depart": "06:00", "price": 250000, "seats_left": 1},
                        {"id": "trip_02", "type": "bus", "operator": "Xe Hoàng Long", "depart": "08:00", "price": 250000, "seats_left": 15},
                    ],
                },
                {"ok": True, "booking": {"id": "tk_01", "trip_id": "trip_02", "seats": 2, "status": "confirmed"}},
            ],
        ),
        # 12. train_bus_tickets: my_bookings -> cancel
        _make_sample(
            "schema_train_bus_02",
            ["train_bus_tickets", "calendar"],
            "Hủy giúp mình vé xe khách đi Đà Lạt đã đặt nhé.",
            [
                {"name": "train_bus_tickets.my_bookings", "arguments": {}},
                {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tk_01"}},
            ],
            [
                {
                    "ok": True,
                    "bookings": [{"id": "tk_01", "trip_id": "trip_03", "seats": 2, "status": "confirmed"}],
                },
                {"ok": True, "booking": {"id": "tk_01", "status": "cancelled"}},
            ],
        ),
        # 13. event_tickets: search -> book (authentic event_01 prefix!)
        _make_sample(
            "schema_event_tickets_01",
            ["event_tickets", "calendar", "ewallet"],
            "Tìm vé hài Sài Gòn Cười rồi đặt cho cả nhà 3 vé nhé.",
            [
                {"name": "event_tickets.search", "arguments": {"query": "Sài Gòn Cười"}},
                {"name": "event_tickets.book", "arguments": {"event_id": "event_01", "quantity": 3}},
            ],
            [
                {
                    "ok": True,
                    "events": [
                        {"id": "event_01", "name": "Đêm hài Sài Gòn Cười", "category": "hài kịch", "date": "2026-11-21", "price": 300000, "available": 20}
                    ],
                },
                {"ok": True, "booking": {"id": "tk_01", "event_id": "event_01", "quantity": 3, "status": "confirmed"}},
            ],
        ),
        # 14. recipes: search -> get -> save (authentic rcp_01)
        _make_sample(
            "schema_recipes_01",
            ["recipes", "notes", "pantry"],
            "Tìm công thức nấu món sườn xào chua ngọt cho 4 người ăn rồi lưu lại công thức đó giúp em.",
            [
                {"name": "recipes.search", "arguments": {"query": "sườn xào chua ngọt", "servings": 4}},
                {"name": "recipes.save", "arguments": {"recipe_id": "rcp_01"}},
            ],
            [
                {"ok": True, "recipes": [{"id": "rcp_01", "name": "Sườn xào chua ngọt", "servings": 4, "time_min": 30, "ingredients": []}]},
                {"ok": True, "saved": {"id": "sav_01", "recipe_id": "rcp_01"}},
            ],
        ),
        # 15. stocks: get_portfolio -> get_quote -> place_order (conditional satisfied)
        _make_sample(
            "schema_stocks_01",
            ["stocks", "banking", "reminders"],
            "Kiểm tra giá cổ phiếu ZMX hiện tại, nếu giá dưới 65000 thì đặt lệnh mua thêm 100 cổ phiếu nhé.",
            [
                {"name": "stocks.get_quote", "arguments": {"symbol": "ZMX"}},
                {"name": "stocks.place_order", "arguments": {"symbol": "ZMX", "side": "buy", "quantity": 100}},
            ],
            [
                {"ok": True, "quote": {"symbol": "ZMX", "name": "CTCP Sữa Mộc Xanh", "price": 64000, "change_pct": -0.6}},
                {"ok": True, "order": {"id": "ord_01", "symbol": "ZMX", "side": "buy", "quantity": 100, "status": "placed"}},
            ],
        ),
        # 16. stocks: NEGATIVE CONDITIONAL (get_quote -> condition NOT met -> STOP with 0 write calls!)
        _make_sample(
            "schema_stocks_neg_cond_01",
            ["stocks", "banking", "reminders"],
            "Xem giá mã ZMX giúp anh, nếu xuống dưới 60 nghìn thì mua 200 cổ phiếu, còn cao hơn thì thôi đừng mua.",
            [
                {"name": "stocks.get_quote", "arguments": {"symbol": "ZMX"}},
            ],
            [
                {"ok": True, "quote": {"symbol": "ZMX", "name": "CTCP Sữa Mộc Xanh", "price": 64000, "change_pct": -0.6}},
            ],
        ),
        # 17. weather + calendar: NEGATIVE CONDITIONAL (rain=False -> STOP immediately, do NOT update calendar!)
        _make_sample(
            "schema_weather_neg_cond_01",
            ["weather", "calendar", "sms", "contacts"],
            "Kiểm tra thời tiết ngày mai, nếu mai mưa thì dời lịch chạy bộ sáng mai sang 7 giờ sáng ngày kia, không mưa thì giữ nguyên nhé.",
            [
                {"name": "weather.get_forecast", "arguments": {"date": "2026-11-19"}},
            ],
            [
                {"ok": True, "forecast": [{"date": "2026-11-19", "condition": "nắng", "rain": False, "temp_min": 24, "temp_max": 32}]},
            ],
        ),
        # 18. bill_payment + ewallet: E-Wallet insufficient balance fallback (read both -> pay via bank directly!)
        _make_sample(
            "schema_bill_ewallet_fallback_01",
            ["bill_payment", "ewallet", "banking", "contacts", "sms"],
            "Thanh toán hóa đơn tiền nước tháng này bằng ví điện tử, nếu ví không đủ tiền thì trừ qua ngân hàng nhé.",
            [
                {"name": "bill_payment.list_bills", "arguments": {"type": "water"}},
                {"name": "ewallet.get_balance", "arguments": {}},
                {"name": "bill_payment.pay", "arguments": {"bill_id": "bill_02", "source": "bank"}},
            ],
            [
                {"ok": True, "bills": [{"id": "bill_02", "type": "water", "amount": 213000, "paid": False}]},
                {"ok": True, "balance": 127183},
                {"ok": True, "bill": {"id": "bill_02", "type": "water", "amount": 213000, "paid": True, "paid_with": "bank"}},
            ],
        ),
        # 19. Self-cancellation ("...À mà thôi, cứ giữ nguyên lịch nhé") -> 0 tool calls!
        _make_sample(
            "schema_self_cancel_01",
            ["weather", "calendar", "chat_ott"],
            "Nếu mai mưa thì lùi buổi dã ngoại sang Chủ Nhật tuần sau. À mà thôi, cô giáo bảo mưa vẫn tổ chức trong nhà, cứ giữ nguyên lịch nhé.",
            [],
            [],
        ),
        # 20. smart_home: selective fan_set (only type='fan' in rooms where climate is off, with speed=3)
        _make_sample(
            "schema_smarthome_fan_filter_01",
            ["smart_home"],
            "Bật hết quạt trong nhà lên số 3 giúp anh, phòng nào điều hòa đang chạy thì thôi không cần bật quạt.",
            [
                {"name": "smart_home.get_status", "arguments": {}},
                {"name": "smart_home.fan_set", "arguments": {"device_id": "fan_living", "power": "on", "speed": 3}},
            ],
            [
                {
                    "ok": True,
                    "devices": [
                        {"id": "light_living", "name": "Đèn phòng khách", "room": "phòng khách", "type": "light", "state": {"power": "on"}},
                        {"id": "fan_living", "name": "Quạt phòng khách", "room": "phòng khách", "type": "fan", "state": {"power": "off", "speed": 1}},
                        {"id": "ac_living", "name": "Điều hòa phòng khách", "room": "phòng khách", "type": "climate", "state": {"mode": "off", "temperature": 26}},
                        {"id": "fan_bedroom", "name": "Quạt phòng ngủ", "room": "phòng ngủ", "type": "fan", "state": {"power": "off", "speed": 1}},
                        {"id": "ac_bedroom", "name": "Điều hòa phòng ngủ", "room": "phòng ngủ", "type": "climate", "state": {"mode": "cool", "temperature": 25}},
                    ],
                },
                {
                    "ok": True,
                    "item": {"id": "fan_living", "name": "Quạt phòng khách", "room": "phòng khách", "type": "fan", "state": {"power": "on", "speed": 3}},
                },
            ],
        ),
        # 21. public_transit: find_routes & topup_card
        _make_sample(
            "schema_transit_01",
            ["public_transit", "maps", "ewallet"],
            "Tìm tuyến xe buýt hoặc metro từ Nhổn đi Ga Hà Nội rồi nạp thêm 100 nghìn vào thẻ đi xe nhé.",
            [
                {"name": "public_transit.find_routes", "arguments": {"origin": "Nhổn", "destination": "Ga Hà Nội"}},
                {"name": "public_transit.topup_card", "arguments": {"amount": 100000}},
            ],
            [
                {"ok": True, "routes": [{"route_id": "metro_3", "name": "Metro Nhổn - Ga Hà Nội", "type": "metro", "duration_min": 25, "fare": 15000}]},
                {"ok": True, "topup": {"id": "top_01", "amount": 100000, "status": "done"}, "card_balance": 115000},
            ],
        ),
        # 22. translator: translate -> chat_ott.send_message
        _make_sample(
            "schema_translator_01",
            ["translator", "chat_ott", "sms"],
            "Dịch câu 'Tôi bị dị ứng hải sản' sang tiếng Anh rồi gửi qua Zalo cho Lan giúp mình.",
            [
                {
                    "name": "translator.translate",
                    "arguments": {"text": "Tôi bị dị ứng hải sản", "to_lang": "en"},
                },
                {"name": "chat_ott.list_chats", "arguments": {}},
                {
                    "name": "chat_ott.send_message",
                    "arguments": {"chat_id": "chat_02", "content": "I am allergic to seafood"},
                },
            ],
            [
                {"ok": True, "text": "Tôi bị dị ứng hải sản", "to_lang": "en", "translation": "I am allergic to seafood", "translated": "I am allergic to seafood"},
                {"ok": True, "chats": [{"id": "chat_01", "name": "Tuấn", "type": "direct"}, {"id": "chat_02", "name": "Lan", "type": "direct"}]},
                {"ok": True, "message": {"id": "msg_01", "chat_id": "chat_02", "content": "I am allergic to seafood"}},
            ],
        ),
    ]
    return raw_specs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Tạo dữ liệu SFT ReAct có suy luận cấu trúc (Micro-Reasoning v3)")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out-dir", default=str(ROOT / "data/sft"))
    ap.add_argument("--val-size", type=int, default=145)
    ap.add_argument("--asr-noise-count", type=int, default=180, help="Số mẫu tăng cường nhiễu ASR thực tế")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    sys.path.insert(0, str((root / "train").resolve()))
    from zaic_tools.validate import catalog_tools  # noqa: E402

    catalog = json.loads((root / "train/train_tools_catalog.json").read_text(encoding="utf-8"))
    catalog_tools_map = catalog_tools(catalog)

    public_cat_path = root / "public_test/public_test_tools_catalog.json"
    public_catalog_tools_map = {}
    if public_cat_path.is_file():
        public_catalog = json.loads(public_cat_path.read_text(encoding="utf-8"))
        public_catalog_tools_map = catalog_tools(public_catalog)

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

    train_rows: list[dict[str, Any]] = []
    train_raw_samples: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    val_ids: list[str] = []

    for i, s in enumerate(samples):
        traj = {"id": s["id"], "messages": build_reasoning_trajectory(s, catalog_tools_map)}
        if i in val_idx:
            val_rows.append(traj)
            val_ids.append(s["id"])
        else:
            train_rows.append(traj)
            train_raw_samples.append(s)

    # 1. Add schema-transfer & reasoning archetype trajectories (oversampled 2x so unseen apps have strong gradient signal)
    if public_catalog_tools_map:
        transfer_samples = build_schema_transfer_samples(public_catalog_tools_map)
        for ts in transfer_samples:
            traj_row = {
                "id": ts["id"],
                "messages": build_reasoning_trajectory(ts, public_catalog_tools_map),
            }
            train_rows.append(traj_row)
            train_rows.append({"id": f"{ts['id']}_dup", "messages": traj_row["messages"]})

    # 2. Add in-context ASR-noise recovery trajectories
    if args.asr_noise_count > 0:
        noisy_rows = build_asr_noisy_variants(
            train_raw_samples,
            catalog_tools_map,
            max_variants=args.asr_noise_count,
            seed=args.seed,
        )
        train_rows.extend(noisy_rows)

    rng.shuffle(train_rows)

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
    print(f"Đã tạo {len(train_rows)} mẫu train có suy luận (Micro-Reasoning v3) -> {out_dir / 'sft_train.jsonl'}")
    print(f"Đã tạo {len(val_rows)} mẫu val có suy luận      -> {out_dir / 'sft_val.jsonl'}")


if __name__ == "__main__":
    main()
