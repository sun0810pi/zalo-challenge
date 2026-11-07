"""Vietnamese Relative DateTime Engine for ZAC 2026 LifeMate.
Resolves relative colloquial Vietnamese temporal references (hôm nay, ngày mai, ngày mốt,
thứ X tuần sau, X giờ chiều/sáng, v.v.) anchored to `current_time` from sample context.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

WEEKDAY_MAP = {
    "hai": 0, "thứ hai": 0, "thứ 2": 0, "t2": 0,
    "ba": 1, "thứ ba": 1, "thứ 3": 1, "t3": 1,
    "tư": 2, "thứ tư": 2, "thứ 4": 2, "t4": 2,
    "năm": 3, "thứ năm": 3, "thứ 5": 3, "t5": 3,
    "sáu": 4, "thứ sáu": 4, "thứ 6": 4, "t6": 4,
    "bảy": 5, "thứ bảy": 5, "thứ 7": 5, "t7": 5,
    "chủ nhật": 6, "cn": 6,
}


class VietnameseRelativeDateTimeEngine:
    """Deterministic Vietnamese relative date/time resolution."""

    @staticmethod
    def resolve_date_string(query_text: str, current_time_iso: str) -> str | None:
        """Extract and resolve relative date from text. Returns YYYY-MM-DD or None."""
        if not current_time_iso:
            return None
        try:
            base_dt = datetime.fromisoformat(current_time_iso)
        except Exception:
            return None

        t = query_text.lower()
        target_date = base_dt.date()
        matched = False

        if "ngày mai" in t or " sáng mai" in t or " chiều mai" in t or " tối mai" in t or t.endswith(" mai") or " mai " in t:
            target_date += timedelta(days=1)
            matched = True
        elif "ngày mốt" in t or "hôm kia" in t or "ngày kia" in t:
            target_date += timedelta(days=2)
            matched = True
        elif "hôm qua" in t:
            target_date -= timedelta(days=1)
            matched = True
        elif "hôm nay" in t or "chiều nay" in t or "sáng nay" in t or "tối nay" in t:
            matched = True
        else:
            for wk_str, wk_idx in WEEKDAY_MAP.items():
                pattern = rf"\b{wk_str}\b"
                if re.search(pattern, t):
                    current_wk = base_dt.weekday()
                    delta_days = (wk_idx - current_wk) % 7
                    if "tuần sau" in t or "tuần tới" in t:
                        delta_days += 7
                    elif delta_days == 0 and "tuần sau" not in t:
                        delta_days = 7
                    target_date += timedelta(days=delta_days)
                    matched = True
                    break

        if matched:
            return target_date.strftime("%Y-%m-%d")
        return None

    @classmethod
    def resolve_datetime_iso(cls, query_text: str, current_time_iso: str, tz_offset: str = "+07:00") -> str | None:
        """Resolve full ISO datetime with hours and minutes anchored to current_time."""
        if not current_time_iso:
            return None
        try:
            base_dt = datetime.fromisoformat(current_time_iso)
        except Exception:
            return None

        date_str = cls.resolve_date_string(query_text, current_time_iso)
        if not date_str:
            return None

        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        target_hour = 9  # Default morning 09:00 if not specified
        target_minute = 0

        t = query_text.lower()
        time_m = re.search(r'(\d{1,2})\s*(?:h|giờ|:)\s*(\d{0,2})', t)
        if time_m:
            h = int(time_m.group(1))
            m = int(time_m.group(2)) if time_m.group(2) else 0
            if any(w in t for w in ("chiều", "tối", "pm")) and h < 12:
                h += 12
            elif "sáng" in t and h == 12:
                h = 0
            target_hour = h
            target_minute = m

        dt = datetime(target_date.year, target_date.month, target_date.day, target_hour, target_minute, 0)
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_offset
