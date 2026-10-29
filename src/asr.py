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
    """Deterministic Vietnamese ASR wrapper with persistent JSON cache and LoRA support."""

    def __init__(
        self,
        model_name: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
        lora_path: str | Path | None = None,
        device: str = "cuda",
        compute_type: str = "float16",
        cache_path: str | Path | None = None,
        use_gt_audio_text: bool = False,
    ):
        if lora_path:
            lp = Path(lora_path)
            if not (lp / "adapter_config.json").is_file():
                sub_ckpts = sorted(
                    lp.glob("checkpoint-*/adapter_config.json"),
                    key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else 0,
                )
                if sub_ckpts:
                    resolved_sub = sub_ckpts[-1].parent.resolve()
                    print(f"[Transcriber] Tự động chọn checkpoint LoRA mới nhất: {resolved_sub}")
                    lora_path = str(resolved_sub)
                else:
                    print(f"[CẢNH BÁO] Không tìm thấy 'adapter_config.json' trong '{lora_path}'. Sẽ dùng model gốc '{model_name}'.")
                    lora_path = None
            else:
                lora_path = str(lp.resolve())

        self.model_name = model_name
        self.lora_path = lora_path
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
        self._processor = None
        self._backend = None

    def _load_model(self):
        if self._model is not None:
            return

        # Case 1: LoRA adapter provided -> load base Whisper, merge LoRA via PEFT and run HF pipeline
        if self.lora_path:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            base_model_name = self.model_name
            if "ct2" in self.model_name or "faster-whisper" in self.model_name:
                base_model_name = "openai/whisper-large-v3-turbo"

            cfg_path = Path(self.lora_path) / "adapter_config.json"
            if cfg_path.is_file():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if cfg.get("base_model_name_or_path"):
                        base_model_name = cfg["base_model_name_or_path"]
                except Exception:
                    pass

            print(f"[ASR] Khởi tạo Whisper Base '{base_model_name}' kết hợp LoRA '{self.lora_path}'...")
            processor_src = self.lora_path if (Path(self.lora_path) / "processor_config.json").is_file() else base_model_name
            try:
                self._processor = AutoProcessor.from_pretrained(processor_src)
            except Exception:
                self._processor = AutoProcessor.from_pretrained(base_model_name)

            torch_dtype = torch.float16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32
            base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base_model, self.lora_path)
            model = model.merge_and_unload()
            model.eval()

            device_id = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
            self._model = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=self._processor.tokenizer,
                feature_extractor=self._processor.feature_extractor,
                device=device_id,
                torch_dtype=torch_dtype,
            )
            self._backend = "transformers"
            print(f"[ASR] Đã nạp thành công mô hình Whisper fine-tuned (Transformers HF) trên device={device_id}!")
            return

        # Case 2: No LoRA adapter -> prefer faster-whisper CT2 for max speed
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
            gen_kwargs = {
                "language": "vi",
                "task": "transcribe",
                "num_beams": 1,
                "do_sample": False,
            }
            if initial_prompt and hasattr(self, "_processor") and self._processor is not None:
                try:
                    prompt_ids = self._processor.get_prompt_ids(initial_prompt, return_tensors="pt")
                    if hasattr(prompt_ids, "tolist"):
                        prompt_ids = prompt_ids[0].tolist() if prompt_ids.ndim > 1 else prompt_ids.tolist()
                    gen_kwargs["prompt_ids"] = prompt_ids
                except Exception:
                    pass
            out = self._model(
                str(wav_path),
                generate_kwargs=gen_kwargs,
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
