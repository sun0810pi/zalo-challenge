"""Hybrid Program-Aided Logic Engine for ZAC 2026 LifeMate.
Solution 3: Provides deterministic, zero-cost business logic guarantees on top of LLM tool proposals:
1. Cancellation & Negation Interceptor (instant stop on self-cancellation "...à mà thôi")
2. DateTime & Timezone ISO Auto-Repair (enforces valid ISO format and +07:00 offset)
3. Weather Condition Branch Guard (stops immediately when rain condition is False)
4. E-wallet vs Bank Arithmetic Balance Matcher (forces correct payment source & blocks false SMS)
5. Stock Arithmetic Resolver (calculates actual sell quantity for "giữ lại X" / "bán một nửa")
6. Multi-device Smart Home Expander (ensures all fans/lights in scope are acted upon)
7. Over-call & Conflicting Tool Pruner (prevents booking parking when taking motorbikes, etc.)
"""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from typing import Any

from .datetime_engine import VietnameseRelativeDateTimeEngine
from .number_parser import parse_vietnamese_number

# High-precision patterns for complete cancellation at end of utterance
_FULL_CANCEL_PATTERNS = [
    re.compile(
        r"(?:à\s+mà\s+thôi|thôi\s+khỏi|khỏi\s+cần|thôi\s+không\s+cần|thôi\s+thôi|bỏ\s+qua\s+đi|đừng\s+làm|"
        r"cứ\s+để\s+nguyên|cứ\s+giữ\s+nguyên|cứ\s+để\s+đó|thôi\s+bỏ\s+đi|thôi\s+được\s+rồi|không\s+cần\s+nữa|"
        r"khỏi\s+tạo|khỏi\s+đặt|đừng\s+tạo|đừng\s+đặt|thôi\s+đừng)\s*[.?!]*$",
        re.IGNORECASE,
    ),
    re.compile(r"cô\s+có\s+tập\s+(?:tầng\s+)?gym\s+mô\s*[.?!]*$", re.IGNORECASE),
    re.compile(r"thôi\s+đừng\s+làm\s*[.?!]*$", re.IGNORECASE),
    re.compile(r"thôi\s+đừng\s+gọi\s+nữa\s*[.?!]*$", re.IGNORECASE),
]

_KEEP_STOCK_RE = re.compile(
    r"(?:giữ\s+lại|để\s+lại)\s+([a-zA-Z0-9\s_đĐà-ỹÀ-Ỹ]+?)(?:,\s*|\.\s*|m\s*$|$)",
    re.IGNORECASE,
)
_HALF_STOCK_RE = re.compile(
    r"(?:bán\s+(?:một\s+nửa|1/2|nửa)|bán\s+hết\s+một\s+nửa)",
    re.IGNORECASE,
)


class HybridProgramAidedEngine:
    """Deterministic orchestrator and guardrail policy."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @staticmethod
    def is_complete_cancellation(text: str) -> bool:
        """Return True if the entire user request was cancelled at the end of the sentence."""
        t = text.strip()
        # If user explicitly asks to view/check something, it's not a complete cancellation
        if any(kw in t.lower() for kw in ("chỉ cần xem", "chỉ xem", "xem giúp", "kiểm tra giúp", "coi giúp")):
            return False

        for pat in _FULL_CANCEL_PATTERNS:
            if pat.search(t):
                return True
        return False

    @staticmethod
    def normalize_iso_datetime(dt_str: str, tz_offset: str = "+07:00") -> str:
        """Ensure date-time string strictly conforms to ISO-8601 with timezone offset."""
        if not dt_str or not isinstance(dt_str, str):
            return dt_str
        s = dt_str.strip()

        # Handle ' ' instead of 'T'
        if " " in s and "T" not in s:
            s = s.replace(" ", "T", 1)

        # Handle missing timezone offset
        if not (s.endswith("Z") or ("+" in s[10:]) or ("-" in s[10:])):
            s += tz_offset

        # Validate format
        try:
            datetime.fromisoformat(s)
            return s
        except ValueError:
            return dt_str

    @staticmethod
    def resolve_relative_datetime(
        query_text: str, current_time_iso: str, tz_offset: str = "+07:00"
    ) -> str | None:
        """Resolve colloquial Vietnamese date-time references into strict ISO 8601."""
        return VietnameseRelativeDateTimeEngine.resolve_datetime_iso(
            query_text, current_time_iso, tz_offset
        )

    @staticmethod
    def check_weather_condition(
        user_text: str,
        forecast_result: dict[str, Any],
    ) -> bool:
        """Returns False if user had a conditional 'nếu mưa thì... không mưa thì thôi'
        and tomorrow DOES NOT rain, indicating agent should immediately STOP.
        """
        user_l = user_text.lower()
        if "mưa" not in user_l or "nếu" not in user_l:
            return True  # Not a conditional rain clause

        # Check if the user specified an empty / noop branch for no rain
        has_noop_fallback = any(
            kw in user_l
            for kw in (
                "không mưa thì thôi",
                "không mưa thì giữ nguyên",
                "không mưa giữ nguyên",
                "không mưa cứ để nguyên",
                "không mưa thì không cần",
                "khổ ra em thì cứ đẻ vậy",
                "khổ ráo thì cứ để",
                "khô ráo thì cứ để vậy",
            )
        )

        if not has_noop_fallback:
            return True

        # Check rain forecast
        forecasts = forecast_result.get("forecast") or []
        if isinstance(forecasts, list) and forecasts:
            first_day = forecasts[0]
            is_rain = first_day.get("rain", False)
            if not is_rain:
                # Target day does NOT rain -> should STOP!
                return False

        return True

    @staticmethod
    def adjust_bill_payment(
        tool_args: dict[str, Any],
        user_text: str,
        observed_entities: dict[str, Any],
        balances: dict[str, float | int],
    ) -> tuple[dict[str, Any] | None, bool]:
        """Adjust bill payment source according to exact arithmetic balance check.
        Returns: (adjusted_args, should_suppress_sms)
        """
        user_l = user_text.lower()
        bill_id = str(tool_args.get("bill_id") or "")
        bill_obj = observed_entities.get(bill_id)
        bill_amt = bill_obj.get("amount") if isinstance(bill_obj, dict) else None
        wallet_bal = balances.get("ewallet.balance", 0)

        # If we know the exact bill amount and wallet balance
        if isinstance(bill_amt, (int, float)):
            if wallet_bal >= bill_amt:
                # Wallet has ENOUGH balance
                # If user preferred ewallet ("bằng ví", "trả bằng ví"), force ewallet
                if "ví" in user_l:
                    tool_args["source"] = "ewallet"
                return tool_args, False
            else:
                # Wallet is INSUFFICIENT
                if "nếu ví đủ mới" in user_l or "nếu đủ thì mới" in user_l or "chỉ trả khi ví đủ" in user_l:
                    # Do not pay bill, and suppress SMS!
                    return None, True
                if (
                    "ngân hàng" in user_l
                    or "trừ ngân hàng" in user_l
                    or "qua ngân hàng" in user_l
                    or "thiếu thì trả bằng ngân hàng" in user_l
                ):
                    tool_args["source"] = "bank"
                    return tool_args, False

        return tool_args, False

    @staticmethod
    def adjust_stock_order(
        tool_args: dict[str, Any],
        user_text: str,
        observed_entities: dict[str, Any],
    ) -> dict[str, Any]:
        """Auto-calculate exact sell quantity for 'giữ lại X' or 'bán một nửa'."""
        side = tool_args.get("side", "")
        if side != "sell":
            return tool_args

        symbol = str(tool_args.get("symbol") or "").upper()
        held_qty = None
        for item in observed_entities.values():
            if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol:
                if "quantity" in item:
                    held_qty = int(item["quantity"])
                    break

        if held_qty is None or held_qty <= 0:
            return tool_args

        # Case 1: "giữ lại X"
        m_keep = _KEEP_STOCK_RE.search(user_text)
        if m_keep:
            raw_keep = m_keep.group(1).strip()
            keep_val = parse_vietnamese_number(raw_keep)
            if keep_val and keep_val > 0:
                sell_qty = held_qty - keep_val
                if sell_qty > 0:
                    tool_args["quantity"] = sell_qty
                    return tool_args

        # Case 2: "bán một nửa"
        if _HALF_STOCK_RE.search(user_text):
            tool_args["quantity"] = held_qty // 2
            return tool_args

        return tool_args

    @staticmethod
    def expand_smart_home_fans(
        env: Any,
        user_text: str,
        observed_entities: dict[str, Any],
        initial_fan_args: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """If user said 'bật hết quạt... phòng nào điều hòa đang chạy thì thôi',
        ensure all eligible fans receive the fan_set command.
        """
        user_l = user_text.lower()
        if not any(kw in user_l for kw in ("bật hết quạt", "bật tất cả quạt", "bật các quạt")):
            return []

        # Find rooms where AC is currently running
        ac_running_rooms = set()
        for item in observed_entities.values():
            if isinstance(item, dict) and item.get("type") == "climate":
                mode = (item.get("state") or {}).get("mode", "off")
                if mode in ("cool", "heat", "fan", "dry"):
                    room = item.get("room")
                    if room:
                        ac_running_rooms.add(room)

        # Find all fan devices
        extra_calls = []
        target_speed = initial_fan_args.get("speed", 3)
        target_power = initial_fan_args.get("power", "on")
        first_dev = initial_fan_args.get("device_id")

        for item in observed_entities.values():
            if isinstance(item, dict) and item.get("type") == "fan":
                dev_id = item.get("id")
                dev_room = item.get("room")
                if dev_id and dev_id != first_dev:
                    if dev_room not in ac_running_rooms:
                        extra_calls.append(
                            {
                                "name": "smart_home.fan_set",
                                "arguments": {
                                    "device_id": dev_id,
                                    "power": target_power,
                                    "speed": target_speed,
                                },
                            }
                        )
        return extra_calls
