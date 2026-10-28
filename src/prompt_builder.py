"""Prompt builder, compact tool schema serializer, and Vietnamese calendar helper for ZAC 2026 LifeMate."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

WEEKDAYS_VI = [
    "Thứ Hai",
    "Thứ Ba",
    "Thứ Tư",
    "Thứ Năm",
    "Thứ Sáu",
    "Thứ Bảy",
    "Chủ Nhật",
]


def build_calendar_helper(current_time_iso: str) -> str:
    """Build a deterministic relative-date lookup table from current_time ISO string."""
    if not current_time_iso:
        return ""
    try:
        dt = datetime.fromisoformat(current_time_iso)
    except ValueError:
        return f"Thời gian hiện tại: {current_time_iso}"

    tz_suffix = current_time_iso[19:] if len(current_time_iso) > 19 else "+07:00"
    wd_now = WEEKDAYS_VI[dt.weekday()]
    lines = [
        f"- Hiện tại: {wd_now}, ngày {dt.strftime('%Y-%m-%d')} lúc {dt.strftime('%H:%M:%S')} (múi giờ {tz_suffix})",
        f"- Hôm qua: {WEEKDAYS_VI[(dt - timedelta(days=1)).weekday()]}, {(dt - timedelta(days=1)).strftime('%Y-%m-%d')}",
        f"- Hôm nay: {wd_now}, {dt.strftime('%Y-%m-%d')}",
        f"- Ngày mai: {WEEKDAYS_VI[(dt + timedelta(days=1)).weekday()]}, {(dt + timedelta(days=1)).strftime('%Y-%m-%d')}",
        f"- Ngày kia (mốt): {WEEKDAYS_VI[(dt + timedelta(days=2)).weekday()]}, {(dt + timedelta(days=2)).strftime('%Y-%m-%d')}",
    ]
    upcoming = []
    for offset in range(1, 10):
        d = dt + timedelta(days=offset)
        upcoming.append(f"{WEEKDAYS_VI[d.weekday()]} ({d.strftime('%Y-%m-%d')})")
    lines.append("- Lịch 9 ngày tới: " + ", ".join(upcoming))
    return "\n".join(lines)


def compress_tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    """Compress a catalog tool definition into a token-efficient schema while keeping all validation rules."""
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])

    compact_props: dict[str, Any] = {}
    for k, v in props.items():
        p_info: dict[str, Any] = {"type": v.get("type", "string")}
        if k in required:
            p_info["required"] = True
        if "enum" in v:
            p_info["enum"] = v["enum"]
        if "format" in v:
            p_info["format"] = v["format"]
        for num_constraint in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if num_constraint in v:
                p_info[num_constraint] = v[num_constraint]
        if "items" in v:
            p_info["items"] = v["items"]
        if "x-ref" in v:
            p_info["x-ref"] = v["x-ref"]
        desc = v.get("description", "")
        if desc and not desc.startswith("Số nguyên, không dấu chấm") and not desc.startswith("Giờ có múi giờ"):
            p_info["desc"] = desc
        compact_props[k] = p_info

    out: dict[str, Any] = {
        "name": tool["name"],
        "kind": tool.get("kind", "write"),
        "desc": tool.get("description", ""),
        "params": compact_props,
    }
    if params.get("x-one-of"):
        out["x-one-of"] = params["x-one-of"]
    if params.get("required"):
        out["required"] = params["required"]
    return out


def get_allowed_tools_schema(sample: dict[str, Any], catalog_tools_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compressed schemas for all tools installed in `sample` plus `sys.ask_missing_info`."""
    installed = sample.get("context", {}).get("installed_apps", {})
    allowed_names: list[str] = []
    for app_info in installed.values():
        for t_name in app_info.get("tools", []):
            if t_name not in allowed_names:
                allowed_names.append(t_name)
    if "sys.ask_missing_info" not in allowed_names:
        allowed_names.append("sys.ask_missing_info")

    schemas = []
    for name in allowed_names:
        if name in catalog_tools_map:
            schemas.append(compress_tool_spec(catalog_tools_map[name]))
    return schemas


SYSTEM_INSTRUCTION = """Bạn là trợ lý ảo thông minh LifeMate điều khiển các ứng dụng trên thiết bị của người dùng.
Nhiệm vụ của bạn là xử lý YÊU CẦU HIỆN TẠI của người dùng bằng cách suy luận theo Schema và gọi chính xác các công cụ (tools) được cung cấp.

### QUY TẮC BẮT BUỘC:
1. **Đọc dữ liệu trước khi Ghi (Read-before-Write)**:
   - Khi một tool `write` cần ID tham chiếu (`x-ref` như `contact_id`, `device_id`, `event_id`, `alarm_id`, `reminder_id`, `bill_id`, `restaurant_id`, `booking_id`...) mà ID đó CHƯA có trong lịch sử, bạn PHẢI gọi tool `read` tương ứng (`contacts.search`, `smart_home.get_status`, `calendar.list_events`, `alarm.list`, `bill_payment.list_bills`...) trước để lấy đúng ID thực tế. Tuyệt đối không tự bịa ID hay địa chỉ.
   - Nếu thông tin và ID cần thiết ĐÃ CÓ SẴN trong phần LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ, hãy dùng trực tiếp ID đó.
2. **Xử lý câu điều kiện & Tự hủy lệnh ("...À mà thôi", "...khoan đã")**:
   - Nếu người dùng đặt điều kiện (VD: "Nếu mai mưa thì..."), hãy gọi tool kiểm tra điều kiện trước. Nếu điều kiện KHÔNG thỏa mãn, dừng lại và kết thúc (không gọi tool hành động).
   - Nếu người dùng đổi ý hoặc tự hủy yêu cầu (VD: "...À mà thôi, cứ để đó", "...thôi khỏi"), chỉ thực hiện ý định cuối cùng của người dùng. Nếu người dùng hủy hết, KHÔNG gọi tool nào cả.
3. **Khi thiếu thông tin hoặc thông tin mơ hồ (`sys.ask_missing_info`)**:
   - Nếu đã gọi tool `read` mà tìm thấy NHIỀU KẾT QUẢ trùng tên (VD: có 2 người tên Minh, 2 người tên Lan) mà lời nói/lịch sử không đủ để phân biệt ai, HOẶC người dùng chưa cung cấp tham số bắt buộc của tool `write`, hãy gọi `sys.ask_missing_info` với `tool_name` là tên tool `write` định gọi và `params` là danh sách tên tham số còn thiếu/mơ hồ (VD: `{"tool_name": "phone.call", "params": ["contact_id"]}`).
4. **Quy trình suy luận bắt buộc (<think>)**:
   - Ở mỗi bước, bạn PHẢI suy nghĩ ngắn gọn trong thẻ `<think>` theo cấu trúc:
<think>
- Ý định & Điều kiện: [Ý định chính; có câu điều kiện hay từ phủ định 'mà thôi' không]
- Tra cứu Schema: [Tool nào giải quyết được theo schema; tham số cần thiết là gì; đã có ID chưa]
- Quyết định: [Gọi tool read để tra cứu / Gọi tool write / Gọi sys.ask_missing_info / Kết thúc]
</think>
   - Sau đó xuất ĐÚNG 1 lệnh gọi tool trong thẻ `<tool_call>`:
<tool_call>
{"name": "<tên_tool>", "arguments": {<các_tham_số>}}
</tool_call>
   - Khi đã hoàn thành toàn bộ yêu cầu, xuất thẻ `<think>` xác nhận hoàn thành và câu trả lời ngắn gọn (KHÔNG xuất thêm thẻ `<tool_call>`)."""


def build_user_context_block(sample: dict[str, Any], catalog_tools_map: dict[str, dict[str, Any]]) -> str:
    """Build the context + tool schemas block for the sample."""
    ctx = sample.get("context", {})
    profile = ctx.get("user_profile", {})
    loc = ctx.get("location", {})
    cal_str = build_calendar_helper(ctx.get("current_time", ""))
    schemas = get_allowed_tools_schema(sample, catalog_tools_map)
    schemas_str = "\n".join(json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in schemas)

    return (
        f"### BỐI CẢNH NGƯỜI DÙNG (CONTEXT)\n"
        f"{cal_str}\n"
        f"- Người dùng: {profile.get('name', '')} | Thành phố: {profile.get('city', '')}\n"
        f"- Địa chỉ nhà: {profile.get('home_address', '')}\n"
        f"- Địa chỉ công ty/nơi làm việc: {profile.get('work_address', '')}\n"
        f"- Tọa độ hiện tại: lat={loc.get('lat')}, lng={loc.get('lng')}\n\n"
        f"### DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP GỌI (ALLOWED TOOLS)\n"
        f"{schemas_str}"
    )


def format_history_block(
    history: list[dict[str, Any]],
    history_transcripts: dict[str, str] | None = None,
) -> str:
    """Format `sample['history']` as a structured read-only log of already-executed turns."""
    if not history:
        return "Không có lịch sử trước đó."
    history_transcripts = history_transcripts or {}
    lines: list[str] = []
    turn_idx = 0
    for turn in history:
        role = turn.get("role")
        if role == "user":
            turn_idx += 1
            audio_path = turn.get("audio", "")
            text = turn.get("audio_text") or history_transcripts.get(audio_path) or "(Lượt nói trước)"
            lines.append(f"[Lượt {turn_idx} - Người dùng]: {text}")
        elif role == "assistant":
            t_calls = turn.get("tool_calls") or []
            t_results = turn.get("tool_results") or []
            for tc, tr in zip(t_calls, t_results):
                lines.append(
                    f"  -> Đã gọi tool: {tc['name']}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)}) => Kết quả: {json.dumps(tr, ensure_ascii=False)}"
                )
            if turn.get("text"):
                lines.append(f"[Lượt {turn_idx} - Trợ lý]: {turn['text']}")
    return "\n".join(lines)


def build_initial_messages(
    sample: dict[str, Any],
    final_user_text: str,
    catalog_tools_map: dict[str, dict[str, Any]],
    history_transcripts: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build the initial ChatML message list where all `assistant` messages belong strictly to the final request."""
    context_block = build_user_context_block(sample, catalog_tools_map)
    history_block = format_history_block(sample.get("history") or [], history_transcripts)
    user_prompt = (
        f"### LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ (ĐÃ THỰC HIỆN XONG TRONG STATE)\n"
        f"{history_block}\n\n"
        f"### YÊU CẦU HIỆN TẠI CẦN XỬ LÝ\n"
        f"Người dùng: {final_user_text}"
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION + "\n\n" + context_block},
        {"role": "user", "content": user_prompt},
    ]
