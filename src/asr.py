"""Deterministic ASR module (faster-whisper / transformers) with context hotwords and disk caching."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_asr_hotwords_prompt(sample: dict[str, Any]) -> str:
    """Build a short Vietnamese initial_prompt with proper nouns from context and history to reduce WER without biasing action intent."""
    ctx = sample.get("context") or {}
    profile = ctx.get("user_profile") or {}
    terms: list[str] = []
    for k in ("name", "city", "home_address", "work_address"):
        val = profile.get(k)
        if val and str(val) not in terms:
            terms.append(str(val))

    # Extract entity names/titles mentioned in history assistant text or tool_results
    for turn in sample.get("history") or []:
        for tr in turn.get("tool_results") or []:
            if isinstance(tr, dict):
                for list_key in ("contacts", "events", "devices", "alarms", "restaurants", "playlists", "tracks"):
                    for item in tr.get(list_key) or []:
                        if isinstance(item, dict):
                            for name_k in ("name", "title", "label", "room"):
                                if item.get(name_k) and str(item[name_k]) not in terms:
                                    terms.append(str(item[name_k]))

    joined = ", ".join(terms[:25])
    if joined:
        return f"Trợ lý ảo tiếng Việt LifeMate. Tên riêng và địa điểm: {joined}."
    return "Trợ lý ảo tiếng Việt LifeMate."


class Transcriber:
    """Deterministic Vietnamese ASR wrapper with persistent JSON cache."""

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

    def transcribe_file(self, wav_path: Path, initial_prompt: str = "", cache_key: str | None = None) -> str:
        key = cache_key or str(wav_path)
        if key in self.cache:
            return self.cache[key]

        self._load_model()
        if self._backend == "faster_whisper":
            segments, _ = self._model.transcribe(
                str(wav_path),
                language="vi",
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                initial_prompt=initial_prompt or None,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        else:
            out = self._model(
                str(wav_path),
                generate_kwargs={"language": "vi", "num_beams": 1, "do_sample": False},
            )
            text = out["text"].strip()

        self.cache[key] = text
        return text

    def transcribe_sample(
        self,
        sample: dict[str, Any],
        split_dir: Path,
    ) -> tuple[str, dict[str, str]]:
        """Return `(final_user_text, {history_audio_rel_path: transcript})` for a sample."""
        history_transcripts: dict[str, str] = {}
        hotword_prompt = build_asr_hotwords_prompt(sample)

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
            )

        return final_text, history_transcripts
