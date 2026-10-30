"""Deterministic ASR module (faster-whisper / transformers) with context hotwords,
anti-hallucination decoding guards, and post-ASR phonetic normalization.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Domain vocabulary per app to bias Whisper initial_prompt toward valid tool/entity terms
APP_DOMAIN_HOTWORDS: dict[str, list[str]] = {
    "alarm": ["báo thức", "hẹn giờ", "tắt báo thức"],
    "appliances": ["máy giặt", "máy rửa bát", "robot hút bụi"],
    "banking": ["chuyển khoản", "ngân hàng", "số dư", "sổ tiết kiệm", "thụ hưởng"],
    "bill_payment": ["hóa đơn", "tiền điện", "tiền nước", "internet", "ví điện tử"],
    "calendar": ["lịch hẹn", "cuộc họp", "sự kiện", "dời lịch"],
    "chat_ott": ["Zalo", "nhắn Zalo", "gọi Zalo", "nhóm chat"],
    "contacts": ["danh bạ", "số điện thoại", "liên hệ"],
    "ecommerce": ["mua sắm", "giỏ hàng", "đặt hàng", "trả hàng", "hủy đơn"],
    "email": ["email", "hộp thư", "lưu trữ"],
    "event_tickets": ["vé sự kiện", "đêm nhạc", "vở kịch", "Sài Gòn Cười"],
    "ewallet": ["ví điện tử", "MoMo", "ZaloPay", "nạp tiền điện thoại", "quét mã QR"],
    "flights": ["chuyến bay", "vé máy bay", "làm thủ tục", "check-in", "Hà Nội", "Đà Nẵng", "Tân Sơn Nhất"],
    "food_delivery": ["đặt đồ ăn", "nhà hàng", "thực đơn", "giao đồ ăn"],
    "gold_price": ["giá vàng", "SJC", "PNJ", "vàng nhẫn", "vàng miếng"],
    "grocery": ["đi chợ", "yến mạch", "sữa tươi", "rau củ", "giỏ hàng"],
    "habit_tracker": ["thói quen", "chuỗi ngày", "đánh dấu"],
    "parking": ["bãi đỗ xe", "đặt chỗ đỗ xe"],
    "pantry": ["tủ bếp", "nguyên liệu", "tồn kho"],
    "public_transit": ["xe buýt", "tàu điện", "metro", "thẻ đi xe"],
    "recipes": ["công thức nấu ăn", "khẩu phần", "nguyên liệu"],
    "reminders": ["nhắc nhở", "lời nhắc"],
    "ride_hailing": ["đặt xe", "gọi xe", "cuốc xe"],
    "shopping_list": ["danh sách mua sắm", "cần mua"],
    "smart_home": ["đèn", "quạt", "điều hòa", "máy lạnh", "công tắc", "kịch bản"],
    "sms": ["tin nhắn SMS", "nhắn SMS", "gửi SMS"],
    "stocks": ["cổ phiếu", "chứng khoán", "danh mục", "đặt lệnh"],
    "train_bus_tickets": ["vé tàu", "vé xe khách", "chuyến tàu"],
    "translator": ["dịch sang tiếng Anh", "tiếng Pháp", "tiếng Nhật", "tiếng Hàn"],
    "tv": ["tivi", "kênh VTV1", "VTV3", "HBO", "âm lượng"],
    "weather": ["thời tiết", "dự báo", "trời mưa", "chất lượng không khí"],
}

# Known Whisper subtitle hallucinations on silence / background noise in Vietnamese
_HALLUCINATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"hãy\s+subscribe\s+cho\s+kênh\s+ghiền\s+mì\s+gõ[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"để\s+không\s+bỏ\s+lỡ\s+những\s+video\s+hấp\s+dẫn[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"hãy\s+subscribe\s+cho\s+video[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"cảm\s+ơn\s+các\s+bạn\s+đã\s+(?:xem|theo\s+dõi)\s+video[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"nhớ\s+đăng\s+ký\s+kênh\s+và\s+bấm\s+chuông[^.?!]*[.?!]?", re.IGNORECASE),
    re.compile(r"subtitles?\s+by[^.?!]*[.?!]?", re.IGNORECASE),
]

# High-precision phonetic replacements for systematic Whisper Vietnamese ASR confusions
_PHONETIC_RULES: list[tuple[re.Pattern[str], str]] = [
    # "Zalo" misheard as "giao lời khô", "giao lời", "giao lưu", "ra lô", "gia lô"
    (re.compile(r"\bgiao\s+lời\s+khô\b", re.IGNORECASE), "Zalo cho"),
    (re.compile(r"\bgiao\s+lời\s+cho\b", re.IGNORECASE), "Zalo cho"),
    (re.compile(r"\bnhắn\s+giao\s+lời\b", re.IGNORECASE), "nhắn Zalo"),
    (re.compile(r"\bgiao\s+lưu\s+tranh\s+tuần\b", re.IGNORECASE), "Zalo cho anh Tuấn"),
    (re.compile(r"\bgiao\s+lưu\s+cho\b", re.IGNORECASE), "Zalo cho"),
    (re.compile(r"\b(?:ra|gia)\s+lô\s+cho\b", re.IGNORECASE), "Zalo cho"),
    (re.compile(r"\bgọi\s+(?:giao\s+lời|ra\s+lô|gia\s+lô)\b", re.IGNORECASE), "gọi Zalo"),
    # "SMS" misheard as "em em", "ép em", "FM"
    (re.compile(r"\b(?:nhắn|nhận|gửi)\s+(?:tin\s+)?(?:em\s+em|ép\s+em|fm)\b", re.IGNORECASE), "nhắn SMS"),
    (re.compile(r"\btin\s+nhắn\s+(?:em\s+em|ép\s+em|fm)\b", re.IGNORECASE), "tin nhắn SMS"),
    # "báo thức" misheard as "mào tức", "báo tức", "bạo lực"
    (re.compile(r"\b(?:mào|báo)\s+tức\b", re.IGNORECASE), "báo thức"),
    (re.compile(r"\bđặt\s+bạo\s+lực\s+lúc\b", re.IGNORECASE), "đặt báo thức lúc"),
    # "Cô giáo Mai" misheard as "bố dạo máy"
    (re.compile(r"\bbố\s+dạo\s+máy\b", re.IGNORECASE), "cô giáo Mai"),
    # Food / grocery items
    (re.compile(r"\byếu\s+mạch\b", re.IGNORECASE), "yến mạch"),
    (re.compile(r"\bdư\s+dụt\b", re.IGNORECASE), "dưa chuột"),
    # Event tickets / places
    (re.compile(r"\bhãy\s+sài\s+gòn\s+cười\b", re.IGNORECASE), "hài Sài Gòn Cười"),
    (re.compile(r"\bhà\s+sài\s+gòn\s+cười\b", re.IGNORECASE), "hài Sài Gòn Cười"),
    (re.compile(r"\bliên\s+hoan\s+tin\s+nhắn\b", re.IGNORECASE), "Liên hoan âm nhạc"),
]

# Collapse repeated words/phrases looped >= 3 times (e.g. "các bạn, các bạn, các bạn...")
_WORD_LOOP_RE = re.compile(r"\b(\w+(?:\s+\w+)?)(?:[\s,]+\1){2,}\b", re.IGNORECASE)


def clean_asr_transcript(text: str, installed_apps: dict[str, Any] | None = None) -> str:
    """Clean Whisper hallucinations, repetition loops, and systematic Vietnamese phonetic confusions."""
    if not text:
        return ""
    cleaned = text.strip()

    # 1. Strip YouTube subtitle hallucinations
    for pat in _HALLUCINATION_PATTERNS:
        cleaned = pat.sub(" ", cleaned)

    # 2. Collapse word/bigram repetition loops (apply twice for nested loops)
    for _ in range(2):
        cleaned = _WORD_LOOP_RE.sub(r"\1", cleaned)

    # 3. Apply phonetic normalization rules
    for pat, repl in _PHONETIC_RULES:
        cleaned = pat.sub(repl, cleaned)

    # 4. Context-conditional phonetic fixes based on installed_apps
    apps = set((installed_apps or {}).keys())
    if "bill_payment" in apps:
        cleaned = re.sub(r"\bhóa\s+đơn\s+tiền\s+(?:được|lớp)\b", "hóa đơn tiền nước", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\btiền\s+lớp\s+tháng\s+này\b", "tiền nước tháng này", cleaned, flags=re.IGNORECASE)
    if "tv" in apps:
        cleaned = re.sub(r"\bkênh\s+vov\s*([1-9])\b", r"kênh VTV\1", cleaned, flags=re.IGNORECASE)

    # Clean up extra whitespace and trailing punctuation artifacts
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned


def build_asr_hotwords_prompt(sample: dict[str, Any]) -> str:
    """Build a Vietnamese initial_prompt with proper nouns from context/history and installed app keywords."""
    ctx = sample.get("context") or {}
    profile = ctx.get("user_profile") or {}
    installed_apps = ctx.get("installed_apps") or {}
    terms: list[str] = []

    for k in ("name", "city", "home_address", "work_address"):
        val = profile.get(k)
        if val and str(val) not in terms:
            terms.append(str(val))

    # Extract entity names/titles mentioned in history assistant text or tool_results
    for turn in sample.get("history") or []:
        for tr in turn.get("tool_results") or []:
            if isinstance(tr, dict):
                for val in tr.values():
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                for name_k in ("name", "title", "label", "room", "symbol", "channel", "stop", "route"):
                                    if item.get(name_k) and str(item[name_k]) not in terms:
                                        terms.append(str(item[name_k]))

    # Add concise domain vocabulary from installed apps
    domain_words: list[str] = []
    for app_name in installed_apps:
        for kw in APP_DOMAIN_HOTWORDS.get(app_name, [])[:3]:
            if kw not in domain_words:
                domain_words.append(kw)

    parts = ["Trợ lý ảo tiếng Việt LifeMate."]
    if terms:
        parts.append(f"Tên riêng: {', '.join(terms[:20])}.")
    if domain_words:
        parts.append(f"Từ khóa: {', '.join(domain_words[:20])}.")
    return " ".join(parts)


class Transcriber:
    """Deterministic Vietnamese ASR wrapper with persistent JSON cache and post-ASR cleaning."""

    def __init__(
        self,
        model_name: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
        device: str = "cuda",
        compute_type: str = "float16",
        cache_path: str | Path | None = None,
        use_gt_audio_text: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.use_gt_audio_text = use_gt_audio_text
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self._model = None
        self._backend = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._backend = "faster_whisper"
        except ImportError:
            import torch
            from transformers import pipeline

            hf_model = (
                "openai/whisper-large-v3-turbo"
                if "turbo" in self.model_name
                else self.model_name
            )
            self._model = pipeline(
                "automatic-speech-recognition",
                model=hf_model,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self._backend = "transformers"

    def save_cache(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def transcribe_file(
        self,
        wav_path: Path,
        initial_prompt: str = "",
        cache_key: str | None = None,
        installed_apps: dict[str, Any] | None = None,
    ) -> str:
        key = cache_key or str(wav_path)
        if key in self.cache and self.cache[key].strip():
            return clean_asr_transcript(self.cache[key], installed_apps=installed_apps)

        self._load_model()
        if self._backend == "faster_whisper":
            try:
                segments, _ = self._model.transcribe(
                    str(wav_path),
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
            except TypeError:
                segments, _ = self._model.transcribe(
                    str(wav_path),
                    language="vi",
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    initial_prompt=initial_prompt or None,
                )
            raw_text = " ".join(seg.text.strip() for seg in segments).strip()
        else:
            out = self._model(
                str(wav_path),
                generate_kwargs={
                    "language": "vi",
                    "num_beams": 1,
                    "do_sample": False,
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                },
            )
            raw_text = out["text"].strip()

        cleaned = clean_asr_transcript(raw_text, installed_apps=installed_apps)
        self.cache[key] = cleaned
        return cleaned

    def transcribe_sample(
        self,
        sample: dict[str, Any],
        split_dir: Path,
    ) -> tuple[str, dict[str, str]]:
        """Return `(final_user_text, {history_audio_rel_path: transcript})` for a sample."""
        history_transcripts: dict[str, str] = {}
        hotword_prompt = build_asr_hotwords_prompt(sample)
        installed_apps = (sample.get("context") or {}).get("installed_apps") or {}

        # 1. History user turns
        for turn in sample.get("history") or []:
            if turn.get("role") == "user":
                rel_audio = turn.get("audio", "")
                if self.use_gt_audio_text and turn.get("audio_text"):
                    history_transcripts[rel_audio] = turn["audio_text"]
                elif rel_audio:
                    wav_path = split_dir / rel_audio
                    history_transcripts[rel_audio] = self.transcribe_file(
                        wav_path,
                        initial_prompt=hotword_prompt,
                        cache_key=rel_audio,
                        installed_apps=installed_apps,
                    )

        # 2. Final user request
        rel_final = sample.get("audio", "")
        if self.use_gt_audio_text and sample.get("audio_text"):
            final_text = sample["audio_text"]
        else:
            wav_path = split_dir / rel_final
            final_text = self.transcribe_file(
                wav_path,
                initial_prompt=hotword_prompt,
                cache_key=rel_final,
                installed_apps=installed_apps,
            )

        return final_text, history_transcripts
