"""Prompt builder, two-tier semantic tool schema serializer, and 14-day calendar helper for ZAC 2026 LifeMate."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from .router import SemanticToolRouter

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
    """Build a deterministic 14-day relative-date lookup table from current_time ISO string."""
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
    for offset in range(1, 15):
        d = dt + timedelta(days=offset)
        upcoming.append(f"{WEEKDAYS_VI[d.weekday()]} ({d.strftime('%Y-%m-%d')})")
    lines.append("- Lịch 14 ngày tới: " + ", ".join(upcoming))
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


def get_two_tier_allowed_tools(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    user_query: str = "",
    router: SemanticToolRouter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Organize installed tools into Tier 1 (Recommended Candidates) and Tier 2 (Secondary Tools)
    to focus model attention while maintaining 100% tool recall.
    """
    router = router or SemanticToolRouter(top_k_recommended=4)
    rec_apps, sec_apps = router.route_sample_apps(user_query, sample, catalog_tools_map)

    installed = sample.get("context", {}).get("installed_apps", {})

    rec_schemas: list[dict[str, Any]] = []
    sec_schemas: list[dict[str, Any]] = []

    for app in rec_apps:
        for tname in installed.get(app, {}).get("tools", []):
            if tname in catalog_tools_map:
                rec_schemas.append(compress_tool_spec(catalog_tools_map[tname]))

    for app in sec_apps:
        for tname in installed.get(app, {}).get("tools", []):
            if tname in catalog_tools_map:
                sec_schemas.append(compress_tool_spec(catalog_tools_map[tname]))

    # Always ensure system tool is present
    if "sys.ask_missing_info" in catalog_tools_map:
        rec_schemas.append(compress_tool_spec(catalog_tools_map["sys.ask_missing_info"]))

    return rec_schemas, sec_schemas


SYSTEM_INSTRUCTION = """Bạn là trợ lý ảo thông minh LifeMate điều khiển các ứng dụng trên thiết bị của người dùng.
Nhiệm vụ của bạn là xử lý YÊU CẦU HIỆN TẠI của người dùng bằng cách suy luận theo Schema và gọi chính xác các công cụ (tools) được cung cấp trong ALLOWED TOOLS.

### QUY TẮC BẮT BUỘC:
1. **Tìm kiếm công cụ & Đọc trước khi Ghi (Tool Discovery & Read-before-Write)**:
   - Dựa vào ý định của người dùng, tìm kiếm công cụ phù hợp trong ALLOWED TOOLS.
   - Khi một tool `write` cần ID tham chiếu (`x-ref` hoặc tham số kết thúc bằng `_id`), bạn PHẢI gọi tool `read` tương ứng trước để lấy đúng ID thực tế trả về từ hệ thống. Tuyệt đối không tự bịa ID hay dùng nhầm ID giữa các ứng dụng khác nhau.
   - Nếu thông tin hoặc ID cần thiết ĐÃ CÓ SẴN trong phần LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ, hãy dùng trực tiếp ID đó mà KHÔNG gọi lại tool `read`.
2. **Xử lý câu điều kiện, So sánh số học & Tự hủy lệnh ("...À mà thôi")**:
   - Nếu người dùng đặt điều kiện (VD: "Nếu mai mưa thì...", "Nếu ví không đủ tiền thì trừ ngân hàng..."):
     + PHẢI gọi các tool `read` kiểm tra điều kiện trước (VD: tra cứu số dư và hóa đơn để so sánh).
     + Sau khi nhận `<tool_response>`, nếu điều kiện KHÔNG thỏa mãn, PHẢI DỪNG LẠI NGAY và KHÔNG gọi tool `write`.
   - Nếu người dùng đổi ý hoặc tự hủy ("...À mà thôi, cứ giữ nguyên nhé", "...thôi khỏi"), chỉ thực hiện ý định cuối cùng còn hiệu lực. Nếu hủy toàn bộ, KHÔNG gọi tool nào cả.
3. **Quy trình suy luận bắt buộc (<think>)**:
   - Ở mỗi bước, bạn PHẢI suy nghĩ trong thẻ `<think>` theo 3 phần:
<think>
- Phân tích Intent & Routing: [Xác định ứng dụng liên quan trong ALLOWED TOOLS; kiểm tra điều kiện/số học hoặc từ đổi ý; thông tin đã có trong lịch sử chưa]
- Tra cứu Schema & Dependency: [Đối chiếu ALLOWED TOOLS; tham số required/x-ref cần thiết; kết quả/ID thực tế lấy được]
- Quyết định: [Gọi tool read / Gọi tool write / Gọi sys.ask_missing_info / Kết thúc không gọi tool]
</think>
   - Sau đó xuất ĐÚNG 1 lệnh gọi tool trong thẻ `<tool_call>`:
<tool_call>
{"name": "<tên_tool>", "arguments": {<các_tham_số>}}
</tool_call>
   - Khi đã hoàn thành toàn bộ yêu cầu (hoặc điều kiện không thỏa mãn / không cần gọi thêm tool), xuất thẻ `<think>` xác nhận và câu trả lời ngắn gọn (KHÔNG xuất thêm thẻ `<tool_call>`)."""


def build_user_context_block(
    sample: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    user_query: str = "",
    router: SemanticToolRouter | None = None,
) -> str:
    """Build the context block with Two-Tier tool schemas."""
    ctx = sample.get("context", {})
    profile = ctx.get("user_profile", {})
    loc = ctx.get("location", {})
    cal_str = build_calendar_helper(ctx.get("current_time", ""))

    rec_schemas, sec_schemas = get_two_tier_allowed_tools(
        sample=sample,
        catalog_tools_map=catalog_tools_map,
        user_query=user_query,
        router=router,
    )

    rec_str = "\n".join(json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in rec_schemas)
    sec_str = "\n".join(json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in sec_schemas)

    tools_block = f"### DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP GỌI (ALLOWED TOOLS)\n"
    tools_block += f"[CÔNG CỤ LIÊN QUAN TRỰC TIẾP (RECOMMENDED)]:\n{rec_str}\n"
    if sec_str:
        tools_block += f"\n[CÁC CÔNG CỤ KHÁC ĐÃ CÀI ĐẶT (SECONDARY)]:\n{sec_str}\n"

    return (
        f"### BỐI CẢNH NGƯỜI DÙNG (CONTEXT)\n"
        f"{cal_str}\n"
        f"- Người dùng: {profile.get('name', '')} | Thành phố: {profile.get('city', '')}\n"
        f"- Địa chỉ nhà: {profile.get('home_address', '')}\n"
        f"- Địa chỉ công ty/nơi làm việc: {profile.get('work_address', '')}\n"
        f"- Tọa độ hiện tại: lat={loc.get('lat')}, lng={loc.get('lng')}\n\n"
        f"{tools_block}"
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
    router: SemanticToolRouter | None = None,
) -> list[dict[str, str]]:
    """Build the initial ChatML message list with Two-Tier tool schemas."""
    context_block = build_user_context_block(
        sample=sample,
        catalog_tools_map=catalog_tools_map,
        user_query=final_user_text,
        router=router,
    )
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
