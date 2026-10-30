"""Neural Context-Grounded Language Preprocessor (Option B: Small Dedicated Model < 0.5B).
Uses a lightweight LLM (e.g. Qwen2.5-0.5B-Instruct or VietAI/vit5-base) conditioned
on the sample's live context entities and installed apps to reconstruct clean, coherent Vietnamese text
from phonetically distorted Whisper ASR outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .preprocessor_a import ContextLanguagePreprocessorA, extract_context_entities

_RESTORER_SYSTEM_PROMPT = """Bạn là chuyên gia chuẩn hóa nhận dạng giọng nói (ASR Post-Processor) cho trợ lý ảo LifeMate.
Nhiệm vụ: Câu đầu vào từ người dùng bị nhận diện sai/méo âm hoặc thiếu dấu do giọng địa phương. Dựa vào BỐI CẢNH (tên người, địa điểm, ứng dụng đã cài), hãy khôi phục lại câu nói chuẩn xác, tự nhiên của người dùng.
Quy tắc:
- Giữ nguyên ý định gốc của người dùng.
- Sửa các từ méo âm thành đúng tên riêng có trong bối cảnh hoặc từ ngữ phổ biến.
- Chỉ xuất ra ĐÚNG câu văn bản đã chuẩn hóa, không giải thích gì thêm."""


class ContextLanguagePreprocessorB:
    """Preprocessor B: Small neural model (< 0.5B parameters) with dynamic context conditioning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        fallback_preprocessor: ContextLanguagePreprocessorA | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.fallback = fallback_preprocessor or ContextLanguagePreprocessorA()
        self.model = None
        self.tokenizer = None
        self._load_failed = False

    def _lazy_load(self) -> bool:
        if self.model is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
            dev = "cuda" if (torch.cuda.is_available() and self.device in ("auto", "cuda")) else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=dev,
                trust_remote_code=True,
            )
            self.model.eval()
            return True
        except Exception as e:
            print(f"[PreprocessorB] Không thể nạp mô hình neural '{self.model_name}' ({e}). Tự động dùng Preprocessor A (Fuzzy/Rule-Free)!")
            self._load_failed = True
            return False

    def build_restoration_prompt(self, raw_asr_text: str, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Construct the context-conditioned restoration prompt for the 0.5B model."""
        entities = extract_context_entities(sample)
        ent_list = list(entities.values())[:15]
        ent_str = ", ".join(ent_list) if ent_list else "Không có"

        installed_apps = list((sample.get("context", {}).get("installed_apps") or {}).keys())
        apps_str = ", ".join(installed_apps[:10])

        user_content = (
            f"BỐI CẢNH THỰC TẾ:\n"
            f"- Các ứng dụng đã cài: {apps_str}\n"
            f"- Tên riêng/Địa điểm đã biết: {ent_str}\n\n"
            f"CÂU GỐC TỪ ASR (BỊ MÉO ÂM): \"{raw_asr_text}\"\n\n"
            f"CÂU CHUẨN HÓA:"
        )

        return [
            {"role": "system", "content": _RESTORER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def preprocess(self, raw_asr_text: str, sample: dict[str, Any]) -> str:
        """Clean boilerplate with Preprocessor A first, then pass through 0.5B neural model (or fallback)."""
        cleaned_raw = self.fallback.clean_boilerplate_and_loops(raw_asr_text)
        if not cleaned_raw:
            return ""

        if not self._lazy_load():
            return self.fallback.preprocess(cleaned_raw, sample)

        import torch

        messages = self.build_restoration_prompt(cleaned_raw, sample)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        restored = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Post-clean quote marks if present
        restored = restored.strip(' "\'\n\r')
        if not restored:
            return self.fallback.preprocess(cleaned_raw, sample)
        return restored
