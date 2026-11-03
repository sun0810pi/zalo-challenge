"""Optional Uncertainty-Based Dual-ASR Fallback & Safe Entity Aligner for LifeMate.

Architecture:
1. Primary ASR: Whisper (`faster-whisper-large-v3-turbo` / `src/asr.py`) with acoustic confidence
   tracking (`avg_logprob`, `no_speech_prob`, `compression_ratio`).
2. Secondary Experimental ASR: `Qwen/Qwen3-ASR-0.6B` invoked on-demand when acoustic uncertainty,
   phonetic/fuzzy repairs, or high-stakes critical slots (amounts, dates/times, proper nouns,
   negations/cancellations) are detected.
3. Safe Context Entity Aligner: Fixes the over-aggressive n-gram preposition/verb swallowing bug
   in `ContextLanguagePreprocessorA.align_entities()` by enforcing word-count parity, boundary
   stopword protection, and context-cue validation.
4. Conflict Detection & Ambiguity Preservation:
   - Compares structured critical slots (`numbers_amounts`, `dates_times`, `negation_cancellation`,
     `proper_nouns`) between Whisper and `Qwen3-ASR-0.6B`.
   - NEVER silently guesses between conflicting names, amounts, dates, or negations.
   - When an unresolvable conflict on a critical slot occurs, preserves explicit ambiguity metadata
     (`conflict_detected=True`, `ambiguous_slots`, `clarification_note`) so the downstream agent
     can ask for clarification (`sys.ask_missing_info`) instead of executing an unsafe write tool call.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.asr import build_asr_hotwords_prompt, clean_asr_transcript
from src.preprocessor_a import (
    _BOILERPLATE_HALLUCINATIONS,
    _LOOP_RE,
    _PHONETIC_ACOUSTIC_RULES,
    extract_context_entities,
    strip_accents,
    token_dice_similarity,
)

# Common Vietnamese function words, verbs, and prepositions that must NEVER be swallowed
# into or overwritten by fuzzy proper-noun alignment (e.g. "đang" -> "Đà Nẵng", "từ Hà Nội" -> "Hà Nội")
_PROTECTED_VIETNAMESE_STOPWORDS: set[str] = {
    "tu", "di", "den", "o", "tai", "vao", "ra", "ve", "cho", "voi", "cua", "bang",
    "dang", "da", "se", "khong", "chua", "dung", "thoi", "hay", "nhe", "nha", "giup",
    "dum", "ho", "toi", "minh", "em", "anh", "chi", "bac", "co", "chu", "sep",
    "hom", "nay", "mai", "sang", "trua", "chieu", "toi", "dem", "gio", "phut", "ruoi",
    "mot", "hai", "ba", "bon", "tu", "nam", "lam", "sau", "bay", "tam", "chin", "muoi",
    "tram", "nghin", "ngan", "trieu", "ty", "dong", "k", "nua", "het", "tat", "ca",
    "neu", "thi", "ma", "nhung", "hoac", "va", "la", "co", "duoc", "can", "muon",
    "xem", "kiem", "tra", "coi", "dat", "huy", "xoa", "bat", "tat", "mo", "dong",
    "chuyen", "nap", "tra", "thanh", "toan", "mua", "ban", "giu", "lai", "doi",
    "nhac", "hen", "goi", "nhan", "gui", "tin", "lich", "bao", "thuc", "xe", "ve",
    "phong", "nha", "cong", "ty", "van", "quan", "tp", "thanh", "pho", "duong",
    "bun", "cha", "pho", "com", "ca", "phe", "tra", "sua", "nuoc", "dien", "mang",
}

# Number word mapping for slot extraction
_VI_NUMBER_WORDS: dict[str, int] = {
    "không": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "bẩy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
}

_DATE_TIME_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("hom_nay", re.compile(r"\bhôm\s+nay\b", re.IGNORECASE)),
    ("ngay_mai", re.compile(r"\b(?:ngày\s+mai|sáng\s+mai|chiều\s+mai|tối\s+mai|mai)\b", re.IGNORECASE)),
    ("ngay_kia", re.compile(r"\b(?:ngày\s+kia|ngày\s+mốt)\b", re.IGNORECASE)),
    ("sang_nay", re.compile(r"\bsáng\s+nay\b", re.IGNORECASE)),
    ("chieu_nay", re.compile(r"\bchiều\s+nay\b", re.IGNORECASE)),
    ("toi_nay", re.compile(r"\btối\s+nay\b", re.IGNORECASE)),
    ("cuoi_tuan", re.compile(r"\bcuối\s+tuần\b", re.IGNORECASE)),
    ("thu_trong_tuan", re.compile(r"\bthứ\s+(?:hai|ba|tư|năm|sáu|bảy|2|3|4|5|6|7)\b", re.IGNORECASE)),
    ("chu_nhat", re.compile(r"\bchủ\s+nhật\b", re.IGNORECASE)),
    (
        "explicit_hour_half",
        re.compile(
            r"\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười\s+một|mười\s+hai)\s*(?:giờ\s*)?rưỡi\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_hour_min",
        re.compile(
            r"\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười\s+một|mười\s+hai)\s*giờ(?:\s*(\d+|mười|mười\s+lăm|hai\s+mươi|ba\s+mươi|bốn\s+lăm)\s*(?:phút)?)?\b",
            re.IGNORECASE,
        ),
    ),
    ("offset_nua_tieng", re.compile(r"\bnửa\s+tiếng\b", re.IGNORECASE)),
    ("offset_1_tieng", re.compile(r"\b(?:1|một)\s+tiếng\b", re.IGNORECASE)),
    ("offset_2_tieng", re.compile(r"\b(?:2|hai)\s+tiếng\b", re.IGNORECASE)),
]

_NEGATION_CANCEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "self_cancel_full",
        re.compile(
            r"(?:à\s+mà\s+thôi|thôi\s+khỏi|khỏi\s+cần|thôi\s+không\s+cần|bỏ\s+qua\s+đi|đừng\s+làm|thôi\s+bỏ\s+đi|thôi\s+đừng\s+làm)",
            re.IGNORECASE,
        ),
    ),
    (
        "keep_unchanged",
        re.compile(
            r"(?:cứ\s+để\s+nguyên|cứ\s+giữ\s+nguyên|không\s+mưa\s+giữ\s+nguyên|không\s+mưa\s+thì\s+thôi|cứ\s+để\s+đó|cứ\s+để\s+lịch\s+đó)",
            re.IGNORECASE,
        ),
    ),
    (
        "self_repair_a_khong",
        re.compile(r"\b(?:à\s+không|à\s+nhầm|ý\s+tôi\s+là|đổi\s+lại\s+thành)\b", re.IGNORECASE),
    ),
    (
        "only_view_override",
        re.compile(r"\b(?:chỉ\s+cần\s+xem|chỉ\s+xem|xem\s+giúp)\b", re.IGNORECASE),
    ),
    (
        "conditional_if_not",
        re.compile(r"\b(?:còn\s+không\s+thì|nếu\s+không|phòng\s+nào\s+.*?\s+thì\s+thôi)\b", re.IGNORECASE),
    ),
]


@dataclass
class ASRProvenanceRecord:
    """Complete audit record for a single audio transcription & fallback decision."""

    sample_id: str
    audio_path: str
    raw_whisper_transcript: str
    whisper_cleaned_transcript: str
    qwen3_asr_transcript: Optional[str]
    current_hybrid_transcript: str
    fallback_corrected_transcript: str
    secondary_asr_invoked: bool
    trigger_reasons: List[str] = field(default_factory=list)
    correction_reasons: List[str] = field(default_factory=list)
    confidence_evidence: Dict[str, Any] = field(default_factory=dict)
    conflict_detected: bool = False
    ambiguous_slots: Dict[str, Any] = field(default_factory=dict)
    clarification_prompt: Optional[str] = None
    whisper_latency_sec: float = 0.0
    qwen3_latency_sec: float = 0.0
    total_latency_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_critical_slots(text: str, sample: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract normalized critical slots (numbers/amounts, dates/times, negations, proper nouns)
    from a Vietnamese utterance for cross-ASR consistency verification and evaluation.
    """
    t = (text or "").strip()
    t_lower = t.lower()

    # 1. Extract numbers and amounts (handle self-repair like "5 giờ nhé, à không, 5 rưỡi")
    numbers: List[str] = []
    # Normalize common spoken amounts (exclude clock units 'giờ'/'phút' which belong to dates_times)
    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|tỷ|k|%|chỗ|tiếng|giờ|phút|cổ\s+phiếu)?\b", t_lower):
        num_str = m.group(1).replace(".", "").replace(",", "")
        unit = (m.group(2) or "").strip()
        if unit in ("giờ", "phút"):
            continue
        if unit in ("nghìn", "ngàn", "k"):
            try:
                numbers.append(str(int(num_str) * 1000))
            except ValueError:
                numbers.append(f"{num_str}_{unit}")
        elif unit == "triệu":
            try:
                numbers.append(str(int(num_str) * 1000000))
            except ValueError:
                numbers.append(f"{num_str}_{unit}")
        else:
            numbers.append(f"{num_str}{('_' + unit) if unit else ''}")

    # Spoken Vietnamese number phrases (e.g. "bốn chỗ", "một nửa", "năm mươi ngàn", "tám giờ")
    spoken_num_patterns = [
        (r"\b(?:bốn|4)\s+chỗ\b", "4_chỗ"),
        (r"\b(?:bảy|7)\s+chỗ\b", "7_chỗ"),
        (r"\b(?:một\s+nửa|1/2|nửa\s+tiếng)\b", "0.5_half"),
        (r"\b(?:năm\s+mươi|50)\s*(?:nghìn|ngàn)\b", "50000"),
        (r"\b(?:một\s+trăm|100)\s*(?:nghìn|ngàn)\b", "100000"),
        (r"\b(?:hai\s+trăm|200)\s*(?:nghìn|ngàn)\b", "200000"),
        (r"\b(?:năm\s+trăm|500)\s*(?:nghìn|ngàn)\b", "500000"),
    ]
    for pat, norm_val in spoken_num_patterns:
        if re.search(pat, t_lower) and norm_val not in numbers:
            numbers.append(norm_val)

    # Spoken digit sequences (e.g., phone numbers "không chín năm không...")
    words = [re.sub(r"[^\wà-ỹ]", "", w) for w in t_lower.split()]
    digit_run: List[str] = []
    for w in words:
        if w in _VI_NUMBER_WORDS and _VI_NUMBER_WORDS[w] < 10:
            digit_run.append(str(_VI_NUMBER_WORDS[w]))
        else:
            if len(digit_run) >= 6:
                phone_str = "".join(digit_run)
                if phone_str not in numbers:
                    numbers.append(f"phone:{phone_str}")
            digit_run = []
    if len(digit_run) >= 6:
        phone_str = "".join(digit_run)
        if phone_str not in numbers:
            numbers.append(f"phone:{phone_str}")

    # 2. Extract date/time expressions
    dates_times: List[str] = []
    for label, pat in _DATE_TIME_PATTERNS:
        for m in pat.finditer(t_lower):
            if label in ("explicit_hour_half", "explicit_hour_min"):
                raw_hr = m.group(1).strip()
                hr_val = _VI_NUMBER_WORDS.get(raw_hr, raw_hr)
                if label == "explicit_hour_half":
                    dates_times.append(f"time:{hr_val}:30")
                else:
                    raw_min = (m.group(2) or "00").strip()
                    min_val = _VI_NUMBER_WORDS.get(raw_min, raw_min)
                    dates_times.append(f"time:{hr_val}:{min_val}")
            else:
                if label not in dates_times:
                    dates_times.append(label)

    # 3. Extract negation / cancellation / conditional markers
    negations: List[str] = []
    for label, pat in _NEGATION_CANCEL_PATTERNS:
        if pat.search(t_lower):
            negations.append(label)

    # 4. Extract proper nouns & domain entities grounded in sample context
    entities_found: List[str] = []
    if sample:
        ctx_entities = extract_context_entities(sample)
        t_folded = strip_accents(t_lower)
        for ent_folded, ent_orig in ctx_entities.items():
            # Skip generic user_profile city/name unless explicitly mentioned in utterance
            if re.search(r"\b" + re.escape(ent_folded) + r"\b", t_folded):
                if ent_orig not in entities_found:
                    entities_found.append(ent_orig)

    # Also detect standard domain proper nouns / brands
    for brand in ("Zalo", "SMS", "SJC", "PNJ", "MoMo", "ZaloPay", "VTV1", "VTV3", "EVN", "Sawaco", "ZKN", "FPT", "HPG"):
        if re.search(r"\b" + re.escape(brand.lower()) + r"\b", t_lower):
            if brand not in entities_found:
                entities_found.append(brand)

    return {
        "numbers_amounts": sorted(set(numbers)),
        "dates_times": sorted(set(dates_times)),
        "negation_cancellation": sorted(set(negations)),
        "proper_nouns_entities": sorted(set(entities_found)),
    }


class SafeContextPreprocessor:
    """Hardened acoustic & context entity preprocessor that prevents false-positive
    overwrites of common Vietnamese verbs/prepositions while preserving high-precision
    phonetic repairs and logged correction provenance.
    """

    def __init__(self, sim_threshold: float = 0.80):
        self.sim_threshold = sim_threshold

    def preprocess_with_evidence(
        self,
        raw_text: str,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        if not raw_text:
            return "", reasons

        cleaned = raw_text.strip()

        # 1. Strip boilerplate hallucinations and Whisper initial_prompt echoes
        for pat in _BOILERPLATE_HALLUCINATIONS:
            if pat.search(cleaned):
                cleaned = pat.sub(" ", cleaned)
                reasons.append("removed_whisper_subtitle_hallucination")

        prompt_echo_re = re.compile(
            r"(?:trợ\s+lý\s+ảo\s+tiếng\s+việt\s+lifemate[^.?!]*[.?!]?|"
            r"từ\s+khóa[:\s]+[^.?!]*?(?:chứng\s+khoán|sức\s+khỏe|thời\s+tiết|lịch\s+hẹn)[^.?!]*[.?!]?|"
            r"từ\s+khóa\s+đơn,\s*tiền\s+điện\s+thoại,\s*hãy\s+subscribe)",
            re.IGNORECASE,
        )
        if prompt_echo_re.search(cleaned):
            cleaned = prompt_echo_re.sub(" ", cleaned)
            reasons.append("removed_whisper_prompt_echo")

        # 2. Collapse repetition loops
        for _ in range(2):
            if _LOOP_RE.search(cleaned):
                cleaned = _LOOP_RE.sub(r"\1", cleaned)
                reasons.append("collapsed_word_repetition_loop")

        # 3. Apply high-precision phonetic acoustic rules
        extra_safe_rules = list(_PHONETIC_ACOUSTIC_RULES) + [
            (re.compile(r"\bvốn\s+chỗ\b", re.IGNORECASE), "4 chỗ"),
            (re.compile(r"\bchuyển\s+ví\s+cho\s+con\s+mai\b", re.IGNORECASE), "chuyển ví cho cô Mai"),
            (re.compile(r"\btiền\s+địa\s+tháng\s+này\b", re.IGNORECASE), "tiền điện tháng này"),
            (re.compile(r"\bkhởi\s+khỏi,\s*không\s+cần\b", re.IGNORECASE), "thôi khỏi, không cần"),
        ]
        for pat, repl in extra_safe_rules:
            if pat.search(cleaned):
                cleaned = pat.sub(repl, cleaned)
                reasons.append(f"phonetic_rule:{repl}")

        apps = set(((sample or {}).get("context", {}).get("installed_apps") or {}).keys())
        if "tv" in apps and re.search(r"\bkênh\s+vov\s*([1-9])\b", cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(r"\bkênh\s+vov\s*([1-9])\b", r"kênh VTV\1", cleaned, flags=re.IGNORECASE)
            reasons.append("app_context_rule:VOV->VTV")
        if "bill_payment" in apps and re.search(
            r"\bhóa\s+đơn\s+tiền\s+(?:được|lớp)\b", cleaned, flags=re.IGNORECASE
        ):
            cleaned = re.sub(
                r"\bhóa\s+đơn\s+tiền\s+(?:được|lớp)\b",
                "hóa đơn tiền nước",
                cleaned,
                flags=re.IGNORECASE,
            )
            reasons.append("app_context_rule:tien_nuoc")

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")

        # 4. Safe Context Entity Alignment (fixes preposition/verb swallowing bug)
        if sample:
            aligned, align_reasons = self._safe_align_entities(cleaned, sample)
            cleaned = aligned
            reasons.extend(align_reasons)

        return cleaned, reasons

    def _safe_align_entities(
        self,
        text: str,
        sample: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        entities = extract_context_entities(sample)
        if not entities:
            return text, reasons

        # Exclude user_profile city/name from blind fuzzy replacement unless length >= 3 words
        ctx = sample.get("context") or {}
        prof = ctx.get("user_profile") or {}
        profile_cities = {
            strip_accents(str(prof.get("city", "")).strip())
            for _ in [0]
            if prof.get("city")
        }

        words = text.split()
        if not words:
            return text, reasons

        matched_spans: List[Tuple[int, int, str, str]] = []
        n = len(words)

        for span_len in (4, 3, 2):
            for i in range(n - span_len + 1):
                if any(i < end and i + span_len > start for start, end, _, _ in matched_spans):
                    continue

                span_words = words[i : i + span_len]
                # Strip punctuation for stopword check
                clean_span_words = [
                    strip_accents(re.sub(r"[^\wà-ỹ]", "", w)) for w in span_words
                ]
                if not all(clean_span_words):
                    continue

                # Rule 1: Never allow protected prepositions/verbs at the start or end of a fuzzy span
                if (
                    clean_span_words[0] in _PROTECTED_VIETNAMESE_STOPWORDS
                    or clean_span_words[-1] in _PROTECTED_VIETNAMESE_STOPWORDS
                ):
                    continue

                # Rule 2: If all words in the span are already common Vietnamese stopwords, skip
                if all(cw in _PROTECTED_VIETNAMESE_STOPWORDS for cw in clean_span_words):
                    continue

                sub_str = " ".join(span_words)
                sub_folded = strip_accents(sub_str)

                best_match = None
                best_sim = 0.0

                for ent_folded, ent_original in entities.items():
                    if ent_folded in profile_cities:
                        continue
                    ent_words = ent_original.split()
                    # Rule 3: Require exact word-count parity so adjacent words are never swallowed
                    if len(ent_words) != span_len:
                        continue
                    if abs(len(sub_folded) - len(ent_folded)) > 3:
                        continue
                    sim = token_dice_similarity(sub_str, ent_original)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = ent_original

                if best_match and best_sim >= self.sim_threshold:
                    if sub_str != best_match:
                        matched_spans.append(
                            (
                                i,
                                i + span_len,
                                best_match,
                                f"safe_entity_align:{sub_str}->{best_match}(sim={best_sim:.2f})",
                            )
                        )

        if not matched_spans:
            return text, reasons

        matched_spans.sort(key=lambda x: x[0])
        out_tokens: List[str] = []
        cur_idx = 0
        for start, end, repl, reason in matched_spans:
            if start > cur_idx:
                out_tokens.extend(words[cur_idx:start])
            out_tokens.append(repl)
            reasons.append(reason)
            cur_idx = end
        if cur_idx < n:
            out_tokens.extend(words[cur_idx:])

        return " ".join(out_tokens), reasons


class Qwen3ASRTranscriber:
    """Optional experimental secondary recognizer wrapping `Qwen/Qwen3-ASR-0.6B`."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        device: str = "cpu",
        cache_path: Optional[ str | Path ] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import sys
        import types
        import torch

        # Optional Japanese forced-aligner dependency (`nagisa`) has no prebuilt cp314 wheel
        # on Windows and is unused when `forced_aligner=None` for Vietnamese ASR.
        for opt_mod in ("nagisa", "soynlp"):
            if opt_mod not in sys.modules:
                try:
                    __import__(opt_mod)
                except ImportError:
                    sys.modules[opt_mod] = types.ModuleType(opt_mod)

        from qwen_asr import Qwen3ASRModel

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        dev_map = "cuda:0" if use_cuda else "cpu"
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=dev_map,
            max_new_tokens=256,
        )

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def transcribe_file(
        self,
        wav_path: Path,
        context_hint: str = "",
        cache_key: Optional[str] = None,
    ) -> Tuple[str, float]:
        key = cache_key or str(wav_path)
        if key in self.cache and isinstance(self.cache[key], dict) and self.cache[key].get("text"):
            return self.cache[key]["text"], float(self.cache[key].get("latency_sec", 0.0))

        self._load_model()
        import numpy as np
        import soundfile as sf

        audio_data, sr = sf.read(str(wav_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        t0 = time.perf_counter()
        results = self._model.transcribe(
            audio=(audio_data, int(sr)),
            context=context_hint or "",
            language="Vietnamese",
        )
        latency = time.perf_counter() - t0
        text = results[0].text.strip() if results else ""
        self.cache[key] = {"text": text, "latency_sec": round(latency, 4)}
        return text, latency


class UncertaintyAwareASRFallback:
    """Uncertainty-based dual-ASR fallback pipeline:
    - Uses Whisper (`faster-whisper-large-v3-turbo`) as primary recognizer.
    - Evaluates acoustic confidence + phonetic/slot risk triggers.
    - Invokes secondary recognizer (`Qwen/Qwen3-ASR-0.6B`) ONLY when uncertainty or
      slot risk affects a critical slot.
    - Compares structured critical slots (`numbers_amounts`, `dates_times`,
      `negation_cancellation`, `proper_nouns_entities`) between Whisper and Qwen3-ASR.
    - Never silently guesses between conflicting names, amounts, dates, or negations;
      preserves explicit ambiguity metadata for downstream agent clarification.
    """

    def __init__(
        self,
        whisper_model_path: str = "models/whisper-large-v3-turbo",
        qwen_model_name: str = "Qwen/Qwen3-ASR-0.6B",
        device: str = "cpu",
        whisper_cache_path: Optional[str | Path] = None,
        qwen_cache_path: Optional[str | Path] = None,
        avg_logprob_threshold: float = -0.35,
        always_verify_critical_writes: bool = True,
    ):
        self.whisper_model_path = whisper_model_path
        self.device = device
        self.avg_logprob_threshold = avg_logprob_threshold
        self.always_verify_critical_writes = always_verify_critical_writes
        self.safe_preprocessor = SafeContextPreprocessor(sim_threshold=0.80)
        self.qwen_asr = Qwen3ASRTranscriber(
            model_name=qwen_model_name,
            device=device,
            cache_path=qwen_cache_path,
        )
        self.whisper_cache_path = Path(whisper_cache_path) if whisper_cache_path else None
        self.whisper_cache: Dict[str, Dict[str, Any]] = {}
        if self.whisper_cache_path and self.whisper_cache_path.is_file():
            try:
                self.whisper_cache = json.loads(self.whisper_cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.whisper_cache = {}
        self._whisper_model = None

    def _load_whisper(self) -> None:
        if self._whisper_model is not None:
            return
        from faster_whisper import WhisperModel

        compute_type = "float16" if self.device == "cuda" else "int8"
        self._whisper_model = WhisperModel(
            self.whisper_model_path,
            device=self.device,
            compute_type=compute_type,
        )

    def save_caches(self) -> None:
        if self.whisper_cache_path:
            self.whisper_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.whisper_cache_path.write_text(
                json.dumps(self.whisper_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self.qwen_asr.save_cache()

    def transcribe_whisper_with_confidence(
        self,
        wav_path: Path,
        initial_prompt: str = "",
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = cache_key or str(wav_path)
        if key in self.whisper_cache and isinstance(self.whisper_cache[key], dict):
            return self.whisper_cache[key]

        self._load_whisper()
        import numpy as np
        import soundfile as sf

        audio_data, sr = sf.read(str(wav_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sr != 16000:
            import soxr

            audio_data = soxr.resample(audio_data, sr, 16000)

        t0 = time.perf_counter()
        segments_iter, info = self._whisper_model.transcribe(
            audio_data,
            language="vi",
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            vad_filter=True,
            initial_prompt=initial_prompt or None,
        )
        segments = list(segments_iter)
        latency = time.perf_counter() - t0

        raw_text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = (
            sum(s.avg_logprob for s in segments) / len(segments) if segments else -1.0
        )
        max_no_speech = max((s.no_speech_prob for s in segments), default=0.0)
        max_comp_ratio = max((s.compression_ratio for s in segments), default=1.0)

        entry = {
            "raw_text": raw_text,
            "avg_logprob": round(float(avg_logprob), 4),
            "no_speech_prob": round(float(max_no_speech), 4),
            "compression_ratio": round(float(max_comp_ratio), 4),
            "latency_sec": round(latency, 4),
        }
        self.whisper_cache[key] = entry
        return entry

    def should_trigger_secondary_asr(
        self,
        whisper_meta: Dict[str, Any],
        whisper_cleaned: str,
        correction_reasons: List[str],
        slots: Dict[str, Any],
        sample: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        triggers: List[str] = []
        avg_lp = float(whisper_meta.get("avg_logprob", -1.0))
        comp_ratio = float(whisper_meta.get("compression_ratio", 1.0))

        if avg_lp < self.avg_logprob_threshold:
            triggers.append(f"low_whisper_logprob({avg_lp:.3f}<{self.avg_logprob_threshold})")
        if not whisper_cleaned or len(whisper_cleaned.split()) < 3:
            triggers.append("empty_or_too_short_whisper_transcript")
        if comp_ratio > 1.85:
            triggers.append(f"high_compression_ratio({comp_ratio:.2f})")
        if correction_reasons:
            triggers.append("phonetic_or_entity_repair_triggered")
        if slots["negation_cancellation"]:
            triggers.append("critical_slot:negation_or_self_repair")
        if self.always_verify_critical_writes and (
            slots["numbers_amounts"] or slots["proper_nouns_entities"]
        ):
            installed = set(((sample.get("context") or {}).get("installed_apps") or {}).keys())
            write_sensitive_apps = {"banking", "ewallet", "bill_payment", "stocks", "sms", "chat_ott", "alarm", "calendar", "ride_hailing"}
            if installed & write_sensitive_apps and avg_lp < -0.18:
                triggers.append("critical_write_slot_verification")

        return (len(triggers) > 0), triggers

    def resolve_and_preserve_ambiguity(
        self,
        sample: Dict[str, Any],
        raw_whisper: str,
        whisper_cleaned: str,
        whisper_meta: Dict[str, Any],
        qwen_raw: Optional[str],
        correction_reasons: List[str],
        trigger_reasons: List[str],
        current_hybrid_text: str,
        qwen_latency: float,
    ) -> ASRProvenanceRecord:
        sample_id = sample.get("id", "unknown")
        audio_path = sample.get("audio", "")
        w_latency = float(whisper_meta.get("latency_sec", 0.0))

        w_slots = extract_critical_slots(whisper_cleaned, sample)

        if qwen_raw is None:
            return ASRProvenanceRecord(
                sample_id=sample_id,
                audio_path=audio_path,
                raw_whisper_transcript=raw_whisper,
                whisper_cleaned_transcript=whisper_cleaned,
                qwen3_asr_transcript=None,
                current_hybrid_transcript=current_hybrid_text,
                fallback_corrected_transcript=whisper_cleaned,
                secondary_asr_invoked=False,
                trigger_reasons=trigger_reasons,
                correction_reasons=correction_reasons,
                confidence_evidence={
                    "whisper_avg_logprob": whisper_meta.get("avg_logprob"),
                    "whisper_no_speech_prob": whisper_meta.get("no_speech_prob"),
                    "whisper_slots": w_slots,
                    "resolution": "primary_high_confidence",
                },
                conflict_detected=False,
                ambiguous_slots={},
                clarification_prompt=None,
                whisper_latency_sec=w_latency,
                qwen3_latency_sec=0.0,
                total_latency_sec=w_latency,
            )

        qwen_cleaned, qwen_reasons = self.safe_preprocessor.preprocess_with_evidence(qwen_raw, sample)
        q_slots = extract_critical_slots(qwen_cleaned, sample)

        ambiguous_slots: Dict[str, Any] = {}
        final_text = whisper_cleaned
        final_reasons = list(correction_reasons)

        # 0. Rescue empty Whisper transcript or Whisper subtitle/prompt-echo hallucination using Qwen3-ASR-0.6B
        whisper_had_hallucination = any(
            r in ("removed_whisper_subtitle_hallucination", "removed_whisper_prompt_echo")
            for r in correction_reasons
        )
        if (not whisper_cleaned or whisper_had_hallucination) and qwen_cleaned:
            final_text = qwen_cleaned
            final_reasons.append("qwen3_rescued_whisper_hallucination_or_empty")
            final_reasons.extend(qwen_reasons)
            w_slots = q_slots

        # 1. Check context-grounded proper nouns: if Qwen3-ASR recovered an exact context entity
        #    (e.g. contact name or brand in sample context) that Whisper missed due to phonetic distortion,
        #    verify whether Whisper has a conflicting valid entity or merely a garbled span.
        w_ents = set(w_slots["proper_nouns_entities"])
        q_ents = set(q_slots["proper_nouns_entities"])
        if q_ents != w_ents:
            only_in_q = q_ents - w_ents
            only_in_w = w_ents - q_ents
            if only_in_q and not only_in_w:
                # Qwen3-ASR recognized a context-grounded entity that Whisper missed completely
                # Check if we can safely align the span or adopt Qwen's context entity
                for ent in sorted(only_in_q):
                    final_reasons.append(f"qwen3_recovered_context_entity:{ent}")
            elif only_in_q and only_in_w:
                # Both recognizers found DIFFERENT valid context entities -> NEVER guess!
                ambiguous_slots["proper_nouns_entities"] = {
                    "whisper": sorted(w_ents),
                    "qwen3_asr": sorted(q_ents),
                }

        # 2. Check numbers / amounts conflict
        w_nums = set(w_slots["numbers_amounts"])
        q_nums = set(q_slots["numbers_amounts"])
        if w_nums != q_nums and (w_nums and q_nums):
            ambiguous_slots["numbers_amounts"] = {
                "whisper": sorted(w_nums),
                "qwen3_asr": sorted(q_nums),
            }

        # 3. Check dates / times conflict
        w_dt = set(w_slots["dates_times"])
        q_dt = set(q_slots["dates_times"])
        if w_dt != q_dt and (w_dt and q_dt):
            ambiguous_slots["dates_times"] = {
                "whisper": sorted(w_dt),
                "qwen3_asr": sorted(q_dt),
            }

        # 4. Check negation / cancellation conflict
        w_neg = set(w_slots["negation_cancellation"])
        q_neg = set(q_slots["negation_cancellation"])
        if w_neg != q_neg:
            ambiguous_slots["negation_cancellation"] = {
                "whisper": sorted(w_neg),
                "qwen3_asr": sorted(q_neg),
            }

        conflict_detected = len(ambiguous_slots) > 0
        clarification_prompt = None
        if conflict_detected:
            slot_names_vi = {
                "proper_nouns_entities": "tên liên hệ/đối tượng",
                "numbers_amounts": "số tiền/số lượng",
                "dates_times": "thời gian/ngày giờ",
                "negation_cancellation": "ý định hủy/xác nhận lệnh",
            }
            conflict_desc = ", ".join(slot_names_vi.get(k, k) for k in ambiguous_slots)
            clarification_prompt = (
                f"[ASR_AMBIGUITY_GUARD] Phát hiện xung đột nhận dạng giọng nói ở ({conflict_desc}): "
                f"Whisper='{whisper_cleaned}' vs Qwen3-ASR='{qwen_cleaned}'. "
                f"Không tự ý đoán giá trị khi gọi tool ghi (write); hãy ưu tiên kiểm tra ngữ cảnh hoặc hỏi lại người dùng (sys.ask_missing_info)."
            )
            final_reasons.append(f"preserved_ambiguity_on_conflict:{','.join(ambiguous_slots.keys())}")

        return ASRProvenanceRecord(
            sample_id=sample_id,
            audio_path=audio_path,
            raw_whisper_transcript=raw_whisper,
            whisper_cleaned_transcript=whisper_cleaned,
            qwen3_asr_transcript=qwen_raw,
            current_hybrid_transcript=current_hybrid_text,
            fallback_corrected_transcript=final_text,
            secondary_asr_invoked=True,
            trigger_reasons=trigger_reasons,
            correction_reasons=final_reasons,
            confidence_evidence={
                "whisper_avg_logprob": whisper_meta.get("avg_logprob"),
                "whisper_no_speech_prob": whisper_meta.get("no_speech_prob"),
                "whisper_compression_ratio": whisper_meta.get("compression_ratio"),
                "whisper_slots": w_slots,
                "qwen3_slots": q_slots,
                "resolution": "conflict_ambiguity_preserved" if conflict_detected else "dual_asr_verified",
            },
            conflict_detected=conflict_detected,
            ambiguous_slots=ambiguous_slots,
            clarification_prompt=clarification_prompt,
            whisper_latency_sec=w_latency,
            qwen3_latency_sec=round(qwen_latency, 4),
            total_latency_sec=round(w_latency + qwen_latency, 4),
        )
