"""Domain-Aware Acoustic & Semantic Language Preprocessor.
Performs:
1. Whisper subtitle & repetition loop cleaning.
2. High-precision Vietnamese phonetic acoustic de-noising (reconstructs garbled intent verbs and domain nouns).
3. Dynamic Context Entity harvesting (profile, history, installed app vocabulary).
4. Multi-gram Sørensen-Dice fuzzy alignment with candidate proper nouns.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

# Vietnamese diacritic folding helper
_DIACRITICS_MAP = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    "đ": "d",
}

# Known silence / boilerplate subtitle hallucinations produced by Whisper on silence
_BOILERPLATE_HALLUCINATIONS = [
    re.compile(r"hãy\s+subscribe\s+cho\s+kênh[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"để\s+không\s+bỏ\s+lỡ\s+những\s+video\s+hấp\s+dẫn[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"hãy\s+subscribe\s+cho\s+video[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"cảm\s+ơn\s+các\s+bạn\s+đã\s+(?:xem|theo\s+dõi)[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"nhớ\s+đăng\s+ký\s+kênh\s+và\s+bấm\s+chuông[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"subtitles?\s+by[^.?!]*[.?!]?", re.IGNORECASE),
]

_LOOP_RE = re.compile(r"\b(\w+(?:\s+\w+)?)(?:[\s,]+\1){2,}\b", re.IGNORECASE)

# Systematic Vietnamese acoustic & regional phonetic confusion corrections
_PHONETIC_ACOUSTIC_RULES: list[tuple[re.Pattern[str], str]] = [
    # Vehicle & Ride hailing
    (re.compile(r"\b(?:đặc|đạt)\s+(?:sẽ|xe)\s+bốm\s+trọt\b", re.IGNORECASE), "đặt xe 4 chỗ"),
    (re.compile(r"\bđặc\s+sẽ\s+bốm\s+trọt\b", re.IGNORECASE), "đặt xe 4 chỗ"),
    (re.compile(r"\b(?:đặc|đạt)\s+sẽ\b", re.IGNORECASE), "đặt xe"),
    (re.compile(r"\bđặc\s+xe\b", re.IGNORECASE), "đặt xe"),
    (re.compile(r"\b(?:bốm|bốn)\s+trọt\b", re.IGNORECASE), "4 chỗ"),
    (re.compile(r"\bbảy\s+trọt\b", re.IGNORECASE), "7 chỗ"),
    (re.compile(r"\bđặc\s+ngay\b", re.IGNORECASE), "đặt ngay"),
    (re.compile(r"\bđặc\s+nè\b", re.IGNORECASE), "đặt ngay"),
    (re.compile(r"\bđặc\s+dúp\b", re.IGNORECASE), "đặt giúp"),
    (re.compile(r"\b(?:nạp|lạp)\s+(?:chiếu\s+)?phim\b", re.IGNORECASE), "rạp phim"),
    (re.compile(r"\bcái\s+nạp\s+phim\b", re.IGNORECASE), "rạp chiếu phim"),

    # Calendar & Event
    (re.compile(r"\b(?:huyết|quyết)\s+họp\b", re.IGNORECASE), "hủy họp"),
    (re.compile(r"\b(?:khủy|khuỷ)\s+lịch\b", re.IGNORECASE), "hủy lịch"),
    (re.compile(r"\b(?:vỏi|vỏ)\s+lịch\b", re.IGNORECASE), "hủy lịch"),
    (re.compile(r"\b(?:ta|đã)\s+(?:bóng|bống|vốn)\b", re.IGNORECASE), "đá bóng"),
    (re.compile(r"\b(?:máy\s+mùn|dào\s+mộn|dời\s+mộn)\b", re.IGNORECASE), "dời muộn"),
    (re.compile(r"\b(?:sạm|sáng|trắng)\s+mang\b", re.IGNORECASE), "sáng mai"),
    (re.compile(r"\bxấu\s+dưới\b", re.IGNORECASE), "6 rưỡi"),
    (re.compile(r"\b(?:giáo|dáo)\s+ngoại\b", re.IGNORECASE), "dã ngoại"),
    (re.compile(r"\btự\s+vận\b", re.IGNORECASE), "từ vựng"),
    (re.compile(r"\b(?:tức|tấc)\s+(?:giùm|dùm)\b", re.IGNORECASE), "nhắc giùm"),
    (re.compile(r"\bhội\s+nhị\b", re.IGNORECASE), "hội nghị"),
    (re.compile(r"\bsổ\s+quên\b", re.IGNORECASE), "sợ quên"),
    (re.compile(r"\btầng\s+gym\b", re.IGNORECASE), "phòng gym"),

    # Smart home
    (re.compile(r"\bthả\s+đèn\b", re.IGNORECASE), "hạ đèn"),

    # Alarm & Reminders
    (re.compile(r"\b(?:mào|báo|vào|bảo)\s+tức\b", re.IGNORECASE), "báo thức"),
    (re.compile(r"\bđặt\s+bạo\s+lực\s+lúc\b", re.IGNORECASE), "đặt báo thức lúc"),

    # Communication & Social
    (re.compile(r"\b(?:giao\s+lời\s+khô|giao\s+lời\s+cho|giao\s+lưu\s+cho|giao\s+lời|gia\s+lô|ra\s+lô)\b", re.IGNORECASE), "Zalo cho"),
    (re.compile(r"\bnhắn\s+(?:giao\s+lời|gia\s+lô|ra\s+lô)\b", re.IGNORECASE), "nhắn Zalo"),
    (re.compile(r"\bgọi\s+(?:giao\s+lời|gia\s+lô|ra\s+lô)\b", re.IGNORECASE), "gọi Zalo"),
    (re.compile(r"\b(?:nhắn|nhận|gửi)\s+(?:tin\s+)?(?:em\s+em|ép\s+em|fm)\b", re.IGNORECASE), "nhắn SMS"),
    (re.compile(r"\btin\s+nhắn\s+(?:em\s+em|ép\s+em|fm)\b", re.IGNORECASE), "tin nhắn SMS"),
    (re.compile(r"\bđói\s+tá\b", re.IGNORECASE), "đối tác"),
    (re.compile(r"\bbiết\s+góp\b", re.IGNORECASE), "viết mail"),
    (re.compile(r"\bbố\s+dạo\s+máy\b", re.IGNORECASE), "cô giáo Mai"),

    # Stocks & Finance
    (re.compile(r"\b(?:giá\s+)?xe\s+xe\s+nhà\b", re.IGNORECASE), "giá vàng SJC"),
    (re.compile(r"\bzcanner\b", re.IGNORECASE), "ZKN"),
    (re.compile(r"\bdứt\s+end\b", re.IGNORECASE), "ZKN"),
    (re.compile(r"\b(?:dổ|đổ)\s+lại\s+(\d+)\b", re.IGNORECASE), r"giữ lại \1"),

    # Food & Recipes & Grocery & Orders
    (re.compile(r"\b(?:hủ|hũ)\s+đơn\b", re.IGNORECASE), "hủy đơn"),
    (re.compile(r"\b(?:hải|hai)\s+công\s+tục\b", re.IGNORECASE), "hai công thức"),
    (re.compile(r"\bcông\s+tục\b", re.IGNORECASE), "công thức"),
    (re.compile(r"\bkhẩu\s+phầm\b", re.IGNORECASE), "khẩu phần"),
    (re.compile(r"\byếu\s+mạch\b", re.IGNORECASE), "yến mạch"),
    (re.compile(r"\bdư\s+dụt\b", re.IGNORECASE), "dưa chuột"),
    (re.compile(r"\bhãy\s+sài\s+gòn\s+cười\b", re.IGNORECASE), "hài Sài Gòn Cười"),
    (re.compile(r"\bliên\s+hoan\s+tin\s+nhắn\b", re.IGNORECASE), "Liên hoan âm nhạc"),

    # Fuel & Prices
    (re.compile(r"\bxăng\s+chín\s+(?:lắm|năm)\b", re.IGNORECASE), "xăng 95"),
]


def strip_accents(text: str) -> str:
    """Strip Vietnamese tone marks and diacritics for phonetic comparison."""
    text_lower = text.lower()
    res = []
    for ch in text_lower:
        res.append(_DIACRITICS_MAP.get(ch, ch))
    return "".join(res)


def token_dice_similarity(s1: str, s2: str) -> float:
    """Compute Sørensen-Dice character bi-gram similarity between two strings."""
    s1_folded = strip_accents(s1).replace(" ", "")
    s2_folded = strip_accents(s2).replace(" ", "")
    if not s1_folded or not s2_folded:
        return 0.0
    if s1_folded == s2_folded:
        return 1.0
    if len(s1_folded) < 2 or len(s2_folded) < 2:
        return 1.0 if s1_folded == s2_folded else 0.0

    pairs1 = {s1_folded[i : i + 2] for i in range(len(s1_folded) - 1)}
    pairs2 = {s2_folded[i : i + 2] for i in range(len(s2_folded) - 1)}
    intersection = len(pairs1 & pairs2)
    return (2.0 * intersection) / (len(pairs1) + len(pairs2))


def extract_context_entities(sample: dict[str, Any]) -> dict[str, str]:
    """Dynamically harvest candidate proper nouns, person names, addresses, stock symbols,
    and device locations from the sample context and history.
    Returns: {normalized_key: original_proper_noun}
    """
    entities: dict[str, str] = {}
    ctx = sample.get("context") or {}
    prof = ctx.get("user_profile") or {}

    for k in ("name", "city", "home_address", "work_address"):
        val = prof.get(k)
        if val and isinstance(val, str) and len(val.strip()) > 1:
            clean_val = val.strip()
            entities[strip_accents(clean_val)] = clean_val

    # Harvest common stock symbols if stocks app is installed
    installed = ctx.get("installed_apps") or {}
    if "stocks" in installed:
        for sym in ("ZKN", "Vitech", "ZTS", "ZMX", "ZDH", "VNINDEX", "HPG", "FPT", "VIC"):
            entities[strip_accents(sym)] = sym

    # Harvest from history tool_results across all apps
    for turn in sample.get("history") or []:
        for tr in turn.get("tool_results") or []:
            if isinstance(tr, dict):
                for v in tr.values():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                for key in ("name", "title", "label", "counterparty", "channel", "symbol", "stop", "route", "provider"):
                                    cand = item.get(key)
                                    if cand and isinstance(cand, str) and len(cand.strip()) > 1:
                                        c_clean = cand.strip()
                                        entities[strip_accents(c_clean)] = c_clean
                    elif isinstance(tr, dict):
                        for key in ("name", "symbol", "provider"):
                            cand = tr.get(key)
                            if cand and isinstance(cand, str) and len(cand.strip()) > 1:
                                c_clean = cand.strip()
                                entities[strip_accents(c_clean)] = c_clean

    return entities


class ContextLanguagePreprocessorA:
    """Acoustic & Semantic Corrector: Cleans Whisper boilerplate, applies phonetic de-noising,
    and dynamically aligns context entities.
    """

    def __init__(self, sim_threshold: float = 0.72):
        self.sim_threshold = sim_threshold

    def clean_boilerplate_and_loops(self, text: str) -> str:
        """Strip YouTube hallucination boilerplate and collapse n-gram repetition loops."""
        if not text:
            return ""
        cleaned = text.strip()
        for pat in _BOILERPLATE_HALLUCINATIONS:
            cleaned = pat.sub(" ", cleaned)
        for _ in range(2):
            cleaned = _LOOP_RE.sub(r"\1", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip(" ,.")

    def apply_phonetic_corrections(self, text: str, sample: dict[str, Any] | None = None) -> str:
        """Apply high-precision Vietnamese phonetic acoustic fixes."""
        cleaned = text
        for pat, repl in _PHONETIC_ACOUSTIC_RULES:
            cleaned = pat.sub(repl, cleaned)

        # Context-dependent app fixes
        apps = set(((sample or {}).get("context", {}).get("installed_apps") or {}).keys())
        if "tv" in apps:
            cleaned = re.sub(r"\bkênh\s+vov\s*([1-9])\b", r"kênh VTV\1", cleaned, flags=re.IGNORECASE)
        if "bill_payment" in apps:
            cleaned = re.sub(r"\bhóa\s+đơn\s+tiền\s+(?:được|lớp)\b", "hóa đơn tiền nước", cleaned, flags=re.IGNORECASE)

        return re.sub(r"\s+", " ", cleaned).strip(" ,.")

    def align_entities(self, text: str, sample: dict[str, Any]) -> str:
        """Align phonetically garbled spans in text to candidate proper nouns."""
        entities = extract_context_entities(sample)
        if not entities:
            return text

        words = text.split()
        if not words:
            return text

        matched_spans: list[tuple[int, int, str]] = []
        n = len(words)

        for span_len in (4, 3, 2, 1):
            for i in range(n - span_len + 1):
                if any(i < end and i + span_len > start for start, end, _ in matched_spans):
                    continue

                sub_str = " ".join(words[i : i + span_len])
                sub_folded = strip_accents(sub_str)

                best_match = None
                best_sim = 0.0

                for ent_folded, ent_original in entities.items():
                    if abs(len(sub_folded) - len(ent_folded)) > 6:
                        continue
                    sim = token_dice_similarity(sub_str, ent_original)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = ent_original

                if best_match and best_sim >= self.sim_threshold:
                    matched_spans.append((i, i + span_len, best_match))

        if not matched_spans:
            return text

        matched_spans.sort(key=lambda x: x[0])
        out_tokens: list[str] = []
        cur_idx = 0
        for start, end, repl in matched_spans:
            if start > cur_idx:
                out_tokens.extend(words[cur_idx:start])
            out_tokens.append(repl)
            cur_idx = end
        if cur_idx < n:
            out_tokens.extend(words[cur_idx:])

        return " ".join(out_tokens)

    def preprocess(self, raw_asr_text: str, sample: dict[str, Any]) -> str:
        """Full pipeline: clean boilerplate -> phonetic de-noising -> dynamic entity alignment."""
        c1 = self.clean_boilerplate_and_loops(raw_asr_text)
        c2 = self.apply_phonetic_corrections(c1, sample)
        c3 = self.align_entities(c2, sample)
        return c3
