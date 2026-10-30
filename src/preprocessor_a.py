"""Deterministic, context-grounded language preprocessor (Option A: Zero Extra Parameters).
Extracts candidate entities and vocabulary dynamically from the sample's user_profile,
installed_apps, and conversation history, then performs fuzzy and phonetic alignment
to repair noisy Whisper ASR output without hardcoding dataset-specific words.
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
    """Dynamically harvest proper nouns, person names, addresses, and entities
    from user_profile and history turns of the specific sample.
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

    # Harvest from history tool_results across any app
    for turn in sample.get("history") or []:
        for tr in turn.get("tool_results") or []:
            if isinstance(tr, dict):
                for v in tr.values():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                for key in ("name", "title", "label", "counterparty", "channel", "symbol", "stop", "route"):
                                    cand = item.get(key)
                                    if cand and isinstance(cand, str) and len(cand.strip()) > 1:
                                        c_clean = cand.strip()
                                        entities[strip_accents(c_clean)] = c_clean

    return entities


class ContextLanguagePreprocessorA:
    """Preprocessor A: Zero-parameter, context-grounded fuzzy entity & acoustic normalizer."""

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

    def align_entities(self, text: str, sample: dict[str, Any]) -> str:
        """Align phonetically garbled spans in text to candidate entities dynamically harvested
        from the sample's user_profile and history (no hardcoded names!).
        """
        entities = extract_context_entities(sample)
        if not entities:
            return text

        words = text.split()
        if not words:
            return text

        # Check multi-word spans of length 1, 2, 3, 4 against candidate proper nouns
        matched_spans: list[tuple[int, int, str]] = []  # (start_word, end_word, replacement)
        n = len(words)

        for span_len in (4, 3, 2, 1):
            for i in range(n - span_len + 1):
                # Avoid overlapping previously matched longer spans
                if any(i < end and i + span_len > start for start, end, _ in matched_spans):
                    continue

                sub_str = " ".join(words[i : i + span_len])
                sub_folded = strip_accents(sub_str)

                best_match = None
                best_sim = 0.0

                for ent_folded, ent_original in entities.items():
                    # Only compare if lengths are reasonably compatible
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

        # Reconstruct string with aligned entities
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
        """Full pipeline of Preprocessor A: clean boilerplate -> collapse loops -> align context entities."""
        cleaned = self.clean_boilerplate_and_loops(raw_asr_text)
        aligned = self.align_entities(cleaned, sample)
        return aligned
