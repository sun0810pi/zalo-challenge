"""Vietnamese text-to-number converter and normalizer for spoken voice numbers."""
from __future__ import annotations

import re

VI_DIGITS = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "bảy": 7, "tám": 8, "chín": 9,
}

def parse_vietnamese_number(text: str) -> int | None:
    """Parse Vietnamese number words like 'hai trăm', '200', 'năm trăm', 'ba mươi' into int."""
    t = text.strip().lower()
    # If direct digits
    m_num = re.search(r"\b(\d+)\b", t)
    if m_num:
        return int(m_num.group(1))

    # Match common patterns: X trăm, X mươi
    for w, val in VI_DIGITS.items():
        if f"{w} trăm" in t:
            return val * 100
        if f"{w} nghìn" in t or f"{w} ngàn" in t:
            return val * 1000
        if f"{w} triệu" in t:
            return val * 1000000

    if "một nửa" in t or "nửa" in t:
        return -1  # flag for half

    return None
