"""Vietnamese text-to-number converter and normalizer for spoken voice numbers."""
from __future__ import annotations

import re

VI_DIGITS = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "nhăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
}

_UNIT_SCALES = {
    "nghìn": 1_000,
    "ngàn": 1_000,
    "k": 1_000,
    "triệu": 1_000_000,
    "tỷ": 1_000_000_000,
}


def _parse_under_thousand(tokens: list[str]) -> int | None:
    """Parse Vietnamese integer < 1000 from word tokens (e.g. 'hai trăm năm mươi', 'mười lăm')."""
    if not tokens:
        return None
    total = 0
    i = 0
    matched = False
    n = len(tokens)

    # Hundreds place: X trăm ...
    if i + 1 < n and tokens[i + 1] == "trăm" and tokens[i] in VI_DIGITS:
        total += VI_DIGITS[tokens[i]] * 100
        i += 2
        matched = True
        if i < n and tokens[i] in ("linh", "lẻ"):
            i += 1
            if i < n and tokens[i] in VI_DIGITS:
                total += VI_DIGITS[tokens[i]]
                return total
            return total

    # Tens place
    if i < n:
        if tokens[i] == "mười":
            total += 10
            i += 1
            matched = True
            if i < n and tokens[i] in VI_DIGITS:
                total += VI_DIGITS[tokens[i]]
                i += 1
        elif i + 1 < n and tokens[i + 1] == "mươi" and tokens[i] in VI_DIGITS:
            total += VI_DIGITS[tokens[i]] * 10
            i += 2
            matched = True
            if i < n and tokens[i] in VI_DIGITS:
                total += VI_DIGITS[tokens[i]]
                i += 1
        elif tokens[i] in VI_DIGITS:
            # Handle shortened tens like "hai lăm" (25), "ba mốt" (31) if followed by digit
            if i + 1 < n and tokens[i + 1] in VI_DIGITS and VI_DIGITS[tokens[i]] >= 2:
                total += VI_DIGITS[tokens[i]] * 10 + VI_DIGITS[tokens[i + 1]]
                i += 2
            else:
                total += VI_DIGITS[tokens[i]]
                i += 1
            matched = True

    return total if matched else None


def parse_vietnamese_number(text: str) -> int | None:
    """Parse Vietnamese digits or compound number words into a non-negative integer."""
    if not text:
        return None
    t = text.strip().lower()

    # 1. Handle numeric literals with optional thousand separators / scale suffixes
    m_scaled = re.search(
        r"\b(\d+(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ|k)?\b",
        t,
    )
    if m_scaled:
        raw_num = m_scaled.group(1)
        scale_word = m_scaled.group(2)
        if re.match(r"^\d{1,3}(?:[.,]\d{3})+$", raw_num):
            val: float = float(raw_num.replace(".", "").replace(",", ""))
        else:
            val = float(raw_num.replace(",", "."))
        if scale_word in _UNIT_SCALES:
            val *= _UNIT_SCALES[scale_word]
        return int(round(val))

    # 2. Compound Vietnamese word numbers (with recursive scale decomposition)
    tokens = re.findall(r"[a-zà-ỹ]+", t)
    if not tokens:
        return None

    return _parse_compound_words(tokens)


def _parse_compound_words(tokens: list[str]) -> int | None:
    """Recursively parse compound Vietnamese number words using descending scale delimiters."""
    if not tokens:
        return None

    for scale_word in ("tỷ", "triệu", "nghìn", "ngàn"):
        if scale_word in tokens:
            idx = tokens.index(scale_word)
            scale = _UNIT_SCALES[scale_word]
            prefix_val = _parse_compound_words(tokens[:idx])
            if prefix_val is None:
                continue
            tail_tokens = tokens[idx + 1 :]
            rem_val = _parse_compound_words(tail_tokens) or 0
            # Handle shorthand like "một triệu hai" -> 1_200_000
            if rem_val < 10 and len(tail_tokens) == 1 and scale >= 1000:
                rem_val *= scale // 10
            return prefix_val * scale + rem_val

    return _parse_under_thousand(tokens)
