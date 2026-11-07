"""Deterministic Entity Normalizer & Database-Aware Fuzzy Fallback for ZAC 2026 LifeMate.
Solution 2: Bridges phonetic/colloquial speech to concrete database IDs.
Handles empty read recovery, relationship aliases, and query broadening.
"""
from __future__ import annotations

import copy
import re
from typing import Any

# Vietnamese kinship / nickname synonyms
RELATIONSHIP_ALIASES: dict[str, list[str]] = {
    "bố": ["bố", "ba", "cha", "bố yêu", "ba yêu", "bố đt", "ba đt"],
    "ba": ["ba", "bố", "cha", "ba yêu", "bố yêu"],
    "mẹ": ["mẹ", "má", "u", "mẹ yêu", "má yêu", "mẹ đt"],
    "má": ["má", "mẹ", "má yêu", "mẹ yêu"],
    "vợ": ["vợ", "bà xã", "vợ yêu", "bx", "nhà"],
    "chồng": ["chồng", "ông xã", "chồng yêu", "ox"],
    "anh": ["anh", "anh yêu", "anh trai"],
    "chị": ["chị", "chị gái", "chị yêu"],
    "em": ["em", "em gái", "em trai", "em yêu"],
    "con": ["con", "con gái", "con trai"],
    "sếp": ["sếp", "thủ trưởng", "lead"],
}


class EntityNormalizer:
    """Deterministic normalizer and recovery engine for entities in LifeMate."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def expand_query_variants(self, query: str) -> list[str]:
        """Generate search query variants for colloquial terms or compound names."""
        if not self.enabled or not query:
            return [query] if query else []

        q_clean = query.strip()
        variants = [q_clean]
        q_lower = q_clean.lower()

        # Check kinship dictionary
        for canonical, aliases in RELATIONSHIP_ALIASES.items():
            if q_lower == canonical or any(a == q_lower for a in aliases):
                for a in aliases:
                    if a not in variants:
                        variants.append(a)
                break

        # Check multi-word Vietnamese names: e.g. "Lê Minh" -> also try "Minh"
        parts = q_clean.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            if last_name not in variants and len(last_name) >= 2:
                variants.append(last_name)

        return variants

    def recover_empty_read(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
        sample: dict[str, Any],
        observed_entities: dict[str, Any],
    ) -> dict[str, Any]:
        """If a search/read tool returned an empty collection, attach grounded recovery hints."""
        if not self.enabled or not isinstance(tool_result, dict):
            return tool_result

        # Check if any list in result is empty
        is_empty = False
        for k, v in tool_result.items():
            if isinstance(v, list) and len(v) == 0:
                is_empty = True
                break

        if not is_empty:
            return tool_result

        result_copy = copy.deepcopy(tool_result)

        # 1. contacts.search returned []
        if tool_name == "contacts.search":
            orig_q = str(tool_args.get("query") or "")
            q_variants = self.expand_query_variants(orig_q)
            # Check if any contact in observed_entities matches variants
            found_candidate = None
            for ent_id, ent in observed_entities.items():
                if isinstance(ent, dict) and "name" in ent:
                    c_name = str(ent.get("name", "")).lower()
                    if any(v.lower() in c_name for v in q_variants):
                        found_candidate = (ent_id, ent.get("name"))
                        break

            if found_candidate:
                cid, cname = found_candidate
                result_copy["_hint"] = (
                    f"Không tìm thấy theo từ khóa '{orig_q}', nhưng tìm thấy liên hệ '{cname}' "
                    f"với ID: '{cid}'. Hãy sử dụng contact_id='{cid}'."
                )
            else:
                result_copy["_hint"] = (
                    f"Kết quả tìm kiếm '{orig_q}' không có dữ liệu. "
                    f"Hãy thử tìm kiếm với từ khóa ngắn gọn hơn (ví dụ tên riêng không dấu hoặc tên gọi) "
                    f"hoặc kiểm tra lại danh bạ."
                )
            return result_copy

        # 2. flights.search returned []
        if tool_name == "flights.search":
            result_copy["_hint"] = (
                "Không tìm thấy chuyến bay theo điều kiện tìm kiếm. "
                "Nếu người dùng hỏi về chuyến bay của họ, hãy gọi `flights.my_bookings` để kiểm tra vé đã đặt."
            )
            return result_copy

        # 3. calendar.list_events returned []
        if tool_name == "calendar.list_events":
            result_copy["_hint"] = (
                "Không có sự kiện nào trong khoảng thời gian tra cứu. "
                "Hãy kiểm tra lại ngày theo bảng bối cảnh 14 ngày hoặc gọi `calendar.list_events` không tham số."
            )
            return result_copy

        # 4. bill_payment.list_bills returned []
        if tool_name == "bill_payment.list_bills":
            result_copy["_hint"] = (
                "Không tìm thấy hóa đơn cần thanh toán. Hãy gọi lại `bill_payment.list_bills` để xem tất cả."
            )
            return result_copy

        # Generic fallback hint
        result_copy["_hint"] = (
            "Kết quả tra cứu rỗng. Hãy thử lại với từ khóa ngắn gọn hơn hoặc gọi công cụ phù hợp khác."
        )
        return result_copy
