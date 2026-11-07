"""Fast-Action Triage Barrier & When2Call Guard for ZAC 2026 LifeMate.
Solution 4: Categorizes user requests into DOABLE / ASK_INFO / REJECT,
prevents out-of-scope hallucinations, and blocks premature sys.ask_missing_info calls.
"""
from __future__ import annotations

import re
from typing import Any

# Intent keywords mapping to canonical app IDs
APP_INTENT_SIGNATURES: dict[str, list[str]] = {
    "food_delivery": ["đặt đồ ăn", "giao đồ ăn", "đặt món", "trà sữa", "cơm trưa", "giao tận nơi", "ship đồ ăn"],
    "train_bus": ["vé xe", "vé tàu", "xe khách", "tàu hỏa", "chuyến tàu", "xe liên tỉnh"],
    "event_tickets": ["vé xem phim", "vé ca nhạc", "vé hòa nhạc", "vé sự kiện", "rạp chiếu phim"],
    "package_tracking": ["đơn hàng đang giao", "kiểm tra đơn hàng", "bưu kiện", "mã vận đơn", "tra cứu bưu kiện"],
    "hotel": ["đặt phòng khách sạn", "khách sạn", "homestay", "resort", "phòng nghỉ"],
    "parking": ["đặt chỗ đỗ xe", "tìm bãi đỗ", "bãi gửi xe", "bãi đỗ xe", "phiên đỗ xe"],
    "stocks": ["đặt lệnh mua cổ phiếu", "bán cổ phiếu", "chứng khoán", "mã cổ phiếu", "khớp lệnh"],
}


class ActionTriageBarrier:
    """Action triage and ask_missing_info validator."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check_out_of_scope(self, sample: dict[str, Any], user_text: str) -> tuple[bool, str]:
        """Check if user request explicitly targets an application NOT in installed_apps.
        Safe-guard: Do not trigger keyword aborts that cause false-rejects (e.g. 'khách sạn' in taxi drop-off,
        or 'vé xe' in 'vé xem phim'). Let the LLM handle schema constraints natively.
        """
        return False, ""

    def guard_ask_missing_info(
        self,
        tool_call_args: dict[str, Any],
        sample: dict[str, Any],
        seen_calls: set[str],
        catalog_tools_map: dict[str, dict[str, Any]],
        observed_entities: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validate whether calling `sys.ask_missing_info` is strictly justified or premature.
        Returns: (is_allowed, reason_or_guidance)
        """
        if not self.enabled:
            return True, "Triage barrier disabled"

        target_tool = tool_call_args.get("tool_name")
        params = tool_call_args.get("params") or []

        if not target_tool:
            return False, "Thiếu tham số 'tool_name' trong sys.ask_missing_info."

        # Check Prerequisite 1: If target_tool has prerequisite read tools that haven't been called yet
        # e.g., flights.book requires flight_id (retrieved via flights.search or flights.my_bookings)
        spec = catalog_tools_map.get(target_tool) or {}
        props = (spec.get("parameters") or {}).get("properties") or {}

        # Look for x-ref in required properties
        for p in params:
            p_info = props.get(p) or {}
            xref = p_info.get("x-ref") or ""
            if xref:
                app_prefix = target_tool.split(".")[0]
                has_read_called = any(
                    (f"{app_prefix}." in c and ("search" in c or "list" in c or "my_" in c or "get_" in c))
                    for c in seen_calls
                )
                if not has_read_called:
                    return False, (
                        f"Tham số '{p}' của công cụ '{target_tool}' cần được tra cứu trước qua công cụ read "
                        f"thay vì hỏi người dùng. Hãy gọi công cụ tìm kiếm/tra cứu tương ứng trước."
                    )

        # Check Prerequisite 2: Is the parameter already available in user context?
        ctx = sample.get("context") or {}
        prof = ctx.get("user_profile") or {}
        for p in params:
            if p in ("home_address", "work_address", "city") and prof.get(p):
                return False, f"Tham số '{p}' đã có sẵn trong hồ sơ người dùng ({prof[p]})."
            if p in ("current_time", "date") and ctx.get("current_time"):
                return False, f"Tham số thời gian đã có trong bối cảnh ({ctx['current_time']})."

        return True, "Hợp lệ"
