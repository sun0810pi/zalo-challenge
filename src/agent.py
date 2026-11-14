"""Deterministic ReAct Tool-Calling Agent for ZAC 2026 LifeMate (< 5B LLM).
Features:
- Pure schema-driven state guardrails (no dataset-specific prefix dependencies)
- Dynamic two-tier semantic tool routing (recommended vs secondary installed tools)
- Solution 2: EntityNormalizer for robust fuzzy read recovery
- Solution 3: HybridProgramAidedEngine for cancellation & arithmetic decisions
- Solution 4: ActionTriageBarrier to eliminate hallucinated calls and premature sys.ask_missing_info
"""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Any

# Required for vLLM V1 on Linux/Colab to prevent "Cannot re-initialize CUDA in forked subprocess"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from .action_triage import ActionTriageBarrier
from .entity_normalizer import EntityNormalizer
from .guardrail import PureSchemaStateTracker, normalize_arguments, parse_tool_calls
from .hybrid_engine import HybridProgramAidedEngine
from .preprocessor_a import ContextLanguagePreprocessorA
from .prompt_builder import build_initial_messages
from .router import SemanticToolRouter

_THINK_OPEN_PLACEHOLDER = "__ZAC_THINK_OPEN__"
_THINK_CLOSE_PLACEHOLDER = "__ZAC_THINK_CLOSE__"


def render_chatml_preserving_think(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Render ChatML without Qwen3's Jinja template stripping `<think>...</think>` from intermediate assistant turns."""
    masked_msgs = []
    for m in messages:
        if m.get("role") == "assistant" and "</think>" in (m.get("content") or ""):
            c = m["content"].replace("<think>", _THINK_OPEN_PLACEHOLDER).replace("</think>", _THINK_CLOSE_PLACEHOLDER)
            masked_msgs.append({"role": "assistant", "content": c})
        else:
            masked_msgs.append(m)

    try:
        prompt = tokenizer.apply_chat_template(
            masked_msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            masked_msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Remove empty <think> block injected by Qwen3 when enable_thinking=False so assistant turn starts cleanly
    prompt = prompt.replace("<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n")
    prompt = prompt.replace(_THINK_OPEN_PLACEHOLDER, "<think>").replace(_THINK_CLOSE_PLACEHOLDER, "</think>")
    return prompt


def fix_all_seeds(seed: int = 42) -> None:
    """Fix all random seeds for 100% deterministic inference per competition rules."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


_VLLM_LORA_RANK_BUCKETS = (64, 128, 256, 512)


def _resolve_max_lora_rank(lora_path: str | None, override: int | None = None) -> int:
    """Resolve vLLM max_lora_rank from explicit override or adapter_config.json."""
    if override is not None:
        rank = int(override)
        return max(64, rank)

    detected_rank = 64
    if lora_path:
        cfg_path = Path(lora_path) / "adapter_config.json"
        if cfg_path.is_file():
            try:
                import json

                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                raw_r = cfg.get("r")
                if isinstance(raw_r, int) and raw_r > 0:
                    detected_rank = raw_r
            except Exception:
                detected_rank = 64

    target = max(64, detected_rank)
    for bucket in _VLLM_LORA_RANK_BUCKETS:
        if target <= bucket:
            return bucket
    return target


class LLMBackend:
    """Unified deterministic inference wrapper for vLLM or HuggingFace Transformers (+ optional LoRA)."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B",
        lora_path: str | None = None,
        backend: str = "auto",
        max_model_len: int = 16384,
        max_new_tokens: int = 512,
        gpu_memory_utilization: float = 0.68,
        load_in_4bit: bool = False,
        max_lora_rank: int | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.lora_path = lora_path
        self.backend = backend
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.load_in_4bit = load_in_4bit
        self.max_lora_rank = max_lora_rank

        self.tokenizer = None
        self.model = None
        self._vllm_sampling_params = None
        self._vllm_lora_request = None

        self._init_backend()

    def _init_backend(self) -> None:
        from transformers import AutoTokenizer

        tokenizer_path = self.lora_path if self.lora_path and Path(self.lora_path).is_dir() else self.model_name_or_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        if self.backend == "auto":
            if self.load_in_4bit:
                self._init_transformers()
            else:
                try:
                    import torch
                    import vllm  # noqa: F401

                    if torch.cuda.is_available():
                        self._init_vllm()
                    else:
                        self._init_transformers()
                except Exception as e:
                    print(f"[LLMBackend] vLLM không khả dụng ({e}). Tự động fallback sang Transformers.")
                    self._init_transformers()
        elif self.backend == "vllm":
            self._init_vllm()
        else:
            self._init_transformers()

    def _init_vllm(self) -> None:
        import torch
        from vllm import LLM as VLLMEngine
        from vllm import SamplingParams

        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        vllm_kwargs = {
            "model": self.model_name_or_path,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": True,
            "seed": 42,
            "enforce_eager": False,
        }
        if self.lora_path and Path(self.lora_path).is_dir():
            from vllm.lora.request import LoRARequest

            resolved_rank = _resolve_max_lora_rank(self.lora_path, override=self.max_lora_rank)
            print(f"[LLMBackend] Enabling LoRA from {self.lora_path} (max_lora_rank={resolved_rank})")
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = resolved_rank
            self.model = VLLMEngine(**vllm_kwargs)
            self._vllm_lora_request = LoRARequest("zac_lora", 1, self.lora_path)
        else:
            self.model = VLLMEngine(**vllm_kwargs)
            self._vllm_lora_request = None

        self._vllm_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            seed=42,
            stop=["</tool_call>", "<|im_end|>", "<|endoftext|>"],
            include_stop_str_in_output=True,
        )
        self.backend = "vllm"

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        if not torch.cuda.is_available():
            torch_dtype = torch.float32

        load_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
            "attn_implementation": "sdpa" if torch.cuda.is_available() else None,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["torch_dtype"] = torch.float16
            print("[LLMBackend] Khởi động model 4-bit (BitsAndBytes NF4) tiết kiệm VRAM.")
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **load_kwargs,
        )
        if self.lora_path and Path(self.lora_path).is_dir():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            if not self.load_in_4bit:
                self.model = self.model.merge_and_unload()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model.eval()
        self.backend = "transformers"

    @staticmethod
    def _postprocess_generation(text: str) -> str:
        """Clean EOS markers and trim at the first </tool_call> closing tag."""
        for eos_tok in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(eos_tok, "")
        text = text.strip()
        if "</tool_call>" in text:
            text = text[: text.find("</tool_call>") + len("</tool_call>")]
        elif "<tool_call>" in text and not text.endswith("</tool_call>"):
            text += "\n</tool_call>"
        return text

    def _prepare_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render ChatML prompt and truncate overflowing intermediate turns without slicing off system/user context."""
        working_msgs = list(messages)
        prompt = render_chatml_preserving_think(
            self.tokenizer,
            working_msgs,
            add_generation_prompt=True,
        )
        max_input_tokens = max(256, self.max_model_len - getattr(self, "max_new_tokens", 512) - 32)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Iteratively drop oldest intermediate (assistant, user) pairs while keeping system + initial user turns
        while len(input_ids) > max_input_tokens and len(working_msgs) > 4:
            working_msgs = working_msgs[:2] + working_msgs[4:]
            prompt = render_chatml_preserving_think(
                self.tokenizer,
                working_msgs,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Drop single remaining intermediate turn if still over budget
        while len(input_ids) > max_input_tokens and len(working_msgs) > 2:
            working_msgs = working_msgs[:2] + working_msgs[3:]
            prompt = render_chatml_preserving_think(
                self.tokenizer,
                working_msgs,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Last resort: preserve head of system prompt + tail of user prompt
        if len(input_ids) > max_input_tokens:
            head_len = max_input_tokens // 2
            tail_len = max_input_tokens - head_len
            input_ids = input_ids[:head_len] + input_ids[-tail_len:]
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        return prompt

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Render ChatML prompt and generate with specified temperature."""
        prompt = self._prepare_prompt(messages)

        if self.backend == "vllm":
            sampling_params = self._vllm_sampling_params
            if temperature > 0.0:
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=0.9,
                    max_tokens=self.max_new_tokens,
                    stop=["</tool_call>", "<|im_end|>", "<|endoftext|>"],
                    include_stop_str_in_output=True,
                )
            outputs = self.model.generate(
                [prompt],
                sampling_params=sampling_params,
                lora_request=self._vllm_lora_request,
                use_tqdm=False,
            )
            return self._postprocess_generation(outputs[0].outputs[0].text)

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        stop_token_ids = [
            self.tokenizer.eos_token_id,
            151645,  # <|im_end|>
            151658,  # </tool_call>
            151643,  # <|endoftext|>
        ]
        stop_token_ids = list(set([x for x in stop_token_ids if x is not None]))

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": stop_token_ids,
            }
            if temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9
            else:
                gen_kwargs["do_sample"] = False
                gen_kwargs["temperature"] = None
                gen_kwargs["top_p"] = None
                gen_kwargs["top_k"] = None

            out_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return self._postprocess_generation(text)

    def generate_batch(self, list_of_messages: list[list[dict[str, str]]]) -> list[str]:
        """Batch generation for multiple samples simultaneously via vLLM continuous batching."""
        if not list_of_messages:
            return []

        prompts = [self._prepare_prompt(messages) for messages in list_of_messages]

        if self.backend == "vllm":
            outputs = self.model.generate(
                prompts,
                sampling_params=self._vllm_sampling_params,
                lora_request=self._vllm_lora_request,
                use_tqdm=False,
            )
            return [self._postprocess_generation(out.outputs[0].text) for out in outputs]

        # Fallback to serial generation if transformers backend
        return [self.generate(msgs) for msgs in list_of_messages]


def build_preprocessor(mode: str = "safe", sim_threshold: float = 0.80) -> Any:
    """Create standardized language preprocessor (`safe`, `a`, or `b`) shared across all runners."""
    if mode == "b":
        from .preprocessor_b import ContextLanguagePreprocessorB

        return ContextLanguagePreprocessorB(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    if mode == "safe":
        from .asr_fallback import SafeContextPreprocessor

        safe_prep = SafeContextPreprocessor(sim_threshold=sim_threshold)

        class _SafePreprocessorAdapter:
            def preprocess(self, raw_asr_text: str, sample: dict[str, Any]) -> str:
                cleaned, _ = safe_prep.preprocess_with_evidence(raw_asr_text, sample)
                return cleaned

        return _SafePreprocessorAdapter()
    return ContextLanguagePreprocessorA()


class ReActSessionState:
    """Unified mutable state for one ReAct trajectory, shared by serial and batched runners."""

    def __init__(
        self,
        sample: dict[str, Any],
        env: Any,
        cleaned_user_text: str,
        cleaned_hist_transcripts: dict[str, str],
        catalog_tools_map: dict[str, dict[str, Any]],
        catalog_apps_map: dict[str, dict[str, Any]] | None = None,
        router: SemanticToolRouter | None = None,
        max_steps: int = 8,
        enable_fuzzy_normalizer: bool = True,
        enable_hybrid_engine: bool = True,
        enable_action_triage: bool = True,
        index: int = 0,
        max_apps: int | None = None,
    ):
        import time

        self.index = index
        self.sample = sample
        self.sample_id = sample["id"]
        self.env = env
        self.cleaned_user_text = cleaned_user_text
        self.catalog_tools_map = catalog_tools_map
        self.max_steps = max_steps
        self.step = 0
        self.finished = False
        self.guardrail_retries = 0
        self.seen_calls: set[str] = set()
        self.suppress_sms = False
        self.last_tool_name: str | None = None
        self.consecutive_same_read = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.state_hash: str = ""
        self.start_time = time.perf_counter()
        self.elapsed = 0.0
        self.messages: list[dict[str, str]] = []

        self.enable_fuzzy_normalizer = enable_fuzzy_normalizer
        self.enable_hybrid_engine = enable_hybrid_engine
        self.enable_action_triage = enable_action_triage

        self.normalizer = EntityNormalizer(enabled=enable_fuzzy_normalizer)
        self.hybrid_engine = HybridProgramAidedEngine(enabled=enable_hybrid_engine)
        self.action_triage = ActionTriageBarrier(enabled=enable_action_triage)

        cur_time = sample.get("context", {}).get("current_time", "")
        self.tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"

        if not cleaned_user_text:
            self._finish()
            return

        if self.enable_action_triage:
            is_oos, _ = self.action_triage.check_out_of_scope(sample, cleaned_user_text)
            if is_oos:
                self._finish()
                return

        if self.enable_hybrid_engine and self.hybrid_engine.is_complete_cancellation(cleaned_user_text):
            self._finish()
            return

        self.state_tracker = PureSchemaStateTracker(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=catalog_tools_map,
        )

        self.messages = build_initial_messages(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=catalog_tools_map,
            history_transcripts=cleaned_hist_transcripts,
            router=router,
            catalog_apps_map=catalog_apps_map,
            max_apps=max_apps,
        )

    def _finish(self) -> None:
        import time

        if self.finished:
            return
        self.finished = True
        self.tool_calls = self.env.get_tool_calls()
        self.state_hash = self.env.get_state_hash()
        self.elapsed = time.perf_counter() - self.start_time
        self.messages = []


def _append_tool_feedback_and_maybe_finish(
    state: ReActSessionState,
    reply: str,
    payload: dict[str, Any],
    count_as_retry: bool = True,
) -> None:
    """Append assistant turn + synthetic feedback and always honor max_steps / retry caps."""
    if not state.messages or state.messages[-1].get("content") != reply:
        state.messages.append({"role": "assistant", "content": reply})
    if count_as_retry:
        state.guardrail_retries += 1
    state.messages.append(
        {
            "role": "user",
            "content": f"<tool_response>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_response>",
        }
    )
    if state.guardrail_retries >= 3 or state.step >= state.max_steps:
        state._finish()


def step_react_state(state: ReActSessionState, reply: str) -> None:
    """Advance `ReActSessionState` by one LLM reply, guaranteed to progress or terminate cleanly."""
    if state.finished:
        return

    state.step += 1
    raw_calls = parse_tool_calls(reply)

    if not raw_calls:
        state._finish()
        return

    call = raw_calls[0]
    t_name = call["name"]
    t_args = normalize_arguments(
        t_name,
        call.get("arguments") or {},
        state.catalog_tools_map,
        tz_offset=state.tz_offset,
    )

    # DateTime Auto-Repair
    if state.enable_hybrid_engine:
        for k in ("time", "start", "end", "schedule_time"):
            if k in t_args and isinstance(t_args[k], str):
                t_args[k] = state.hybrid_engine.normalize_iso_datetime(t_args[k], state.tz_offset)

    # Guard sys.ask_missing_info
    if t_name == "sys.ask_missing_info" and state.enable_action_triage:
        allowed, triage_reason = state.action_triage.guard_ask_missing_info(
            t_args,
            state.sample,
            state.seen_calls,
            state.catalog_tools_map,
            state.state_tracker.observed_entities,
        )
        if not allowed:
            _append_tool_feedback_and_maybe_finish(
                state,
                reply,
                {
                    "ok": False,
                    "validation_error": triage_reason,
                    "hint": "Hãy gọi công cụ tra cứu chính xác hoặc chọn đối tượng phù hợp từ kết quả trước đó.",
                },
                count_as_retry=True,
            )
            return

    # Hybrid Stock Math & Bill Payment adjustments + deterministic suppressions
    if state.enable_hybrid_engine:
        if t_name == "stocks.place_order":
            t_args = state.hybrid_engine.adjust_stock_order(
                t_args, state.cleaned_user_text, state.state_tracker.observed_entities
            )
        elif t_name == "bill_payment.pay":
            t_args, supp_sms = state.hybrid_engine.adjust_bill_payment(
                t_args,
                state.cleaned_user_text,
                state.state_tracker.observed_entities,
                state.state_tracker.balances,
            )
            if t_args is None:
                state.suppress_sms = True
                _append_tool_feedback_and_maybe_finish(
                    state,
                    reply,
                    {
                        "ok": True,
                        "skipped": True,
                        "reason": "Điều kiện số dư ví điện tử không đủ theo yêu cầu người dùng nên bỏ qua bước thanh toán.",
                    },
                    count_as_retry=True,
                )
                return
            if supp_sms:
                state.suppress_sms = True

        if t_name == "sms.send" and state.suppress_sms:
            _append_tool_feedback_and_maybe_finish(
                state,
                reply,
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "Thanh toán đã bị hủy do điều kiện không thỏa mãn nên không gửi SMS xác nhận.",
                },
                count_as_retry=True,
            )
            return

        if t_name == "parking.reserve" and any("ride_hailing.book" in s and "bike" in s for s in state.seen_calls):
            _append_tool_feedback_and_maybe_finish(
                state,
                reply,
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "Đã đặt xe máy (bike), không cần đặt chỗ đỗ ô tô (parking.reserve).",
                },
                count_as_retry=True,
            )
            return

    # Entity normalization
    if state.enable_fuzzy_normalizer:
        t_args = state.normalizer.resolve_fuzzy_arguments(
            t_name, t_args, state.state_tracker.observed_entities, state.sample
        )

    state.messages.append({"role": "assistant", "content": reply})

    # Duplicate exact call check
    call_sig = f"{t_name}:{json.dumps(t_args, sort_keys=True, ensure_ascii=False)}"
    if call_sig in state.seen_calls:
        _append_tool_feedback_and_maybe_finish(
            state,
            reply,
            {
                "ok": False,
                "error": "Bạn vừa gọi đúng lệnh này rồi, không gọi lặp lại. Hãy chuyển sang bước tiếp theo hoặc kết thúc.",
            },
            count_as_retry=True,
        )
        return

    # Circuit breaker for runaway consecutive read-tool retry loops (e.g., 8x search/get_quote)
    tool_kind = (state.catalog_tools_map.get(t_name) or {}).get("kind", "write")
    if tool_kind == "read" and state.last_tool_name == t_name and state.consecutive_same_read >= 3:
        _append_tool_feedback_and_maybe_finish(
            state,
            reply,
            {
                "ok": False,
                "error": f"Đã gọi '{t_name}' liên tiếp {state.consecutive_same_read} lần không tiến triển. Hãy dùng dữ liệu đã có hoặc kết thúc.",
            },
            count_as_retry=True,
        )
        return

    # Zero-cost schema validation
    val_res = state.env.validate(t_name, t_args)
    if not val_res.get("ok"):
        err_msg = val_res.get("error", "Sai định dạng schema")
        _append_tool_feedback_and_maybe_finish(
            state,
            reply,
            {
                "ok": False,
                "validation_error": err_msg,
                "hint": "Hãy sửa lại đúng tên tham số/kiểu dữ liệu.",
            },
            count_as_retry=True,
        )
        return

    # Pure schema state precheck
    pre_err = state.state_tracker.check_pre_execution(t_name, t_args)
    if pre_err is not None:
        _append_tool_feedback_and_maybe_finish(
            state,
            reply,
            {"ok": False, "precheck_error": pre_err},
            count_as_retry=True,
        )
        return

    # Execute real simulator call
    state.seen_calls.add(call_sig)
    if tool_kind == "read":
        state.consecutive_same_read = (state.consecutive_same_read + 1) if state.last_tool_name == t_name else 1
    else:
        state.consecutive_same_read = 0
    state.last_tool_name = t_name

    exec_res = state.env.call(t_name, t_args)
    state.state_tracker.record_observation(t_name, t_args, exec_res)

    # Post-execution hybrid checks
    if state.enable_hybrid_engine:
        if t_name in ("weather.get_forecast", "weather.get_current"):
            if not state.hybrid_engine.check_weather_condition(state.cleaned_user_text, exec_res):
                state._finish()
                return

        if t_name == "smart_home.fan_set":
            extra_fans = state.hybrid_engine.expand_smart_home_fans(
                state.env, state.cleaned_user_text, state.state_tracker.observed_entities, t_args
            )
            for ef in extra_fans:
                ef_sig = f"{ef['name']}:{json.dumps(ef['arguments'], sort_keys=True, ensure_ascii=False)}"
                if ef_sig not in state.seen_calls:
                    state.seen_calls.add(ef_sig)
                    ef_res = state.env.call(ef["name"], ef["arguments"])
                    state.state_tracker.record_observation(ef["name"], ef["arguments"], ef_res)

    # Fuzzy Read Recovery
    if state.enable_fuzzy_normalizer:
        prompt_res = state.normalizer.recover_empty_read(
            t_name, t_args, exec_res, state.sample, state.state_tracker.observed_entities
        )
    else:
        prompt_res = exec_res

    state.messages.append(
        {
            "role": "user",
            "content": f"<tool_response>\n{json.dumps(prompt_res, ensure_ascii=False)}\n</tool_response>",
        }
    )

    if t_name == "sys.ask_missing_info" or state.step >= state.max_steps or state.guardrail_retries >= 3:
        state._finish()


class LifeMateAgent:
    """ReAct agent with single-tool execution per step, pure schema-driven state guardrail,
    and switchable solutions 2, 3, and 4.
    """

    def __init__(
        self,
        llm: LLMBackend,
        catalog_tools_map: dict[str, dict[str, Any]],
        max_steps: int = 8,
        preprocessor: Any | None = None,
        router: SemanticToolRouter | None = None,
        catalog_apps_map: dict[str, dict[str, Any]] | None = None,
        enable_fuzzy_normalizer: bool = True,
        enable_hybrid_engine: bool = True,
        enable_action_triage: bool = True,
        max_apps: int | None = None,
    ):
        self.llm = llm
        self.catalog_tools_map = catalog_tools_map
        self.max_steps = max_steps
        self.preprocessor = preprocessor or ContextLanguagePreprocessorA()
        self.router = router or SemanticToolRouter(top_k_recommended=4)
        self.catalog_apps_map = catalog_apps_map

        self.enable_fuzzy_normalizer = enable_fuzzy_normalizer
        self.enable_hybrid_engine = enable_hybrid_engine
        self.enable_action_triage = enable_action_triage
        self.max_apps = max_apps

        self.entity_normalizer = EntityNormalizer(enabled=enable_fuzzy_normalizer)
        self.hybrid_engine = HybridProgramAidedEngine(enabled=enable_hybrid_engine)
        self.action_triage = ActionTriageBarrier(enabled=enable_action_triage)

    def run_sample(
        self,
        env: Any,
        sample: dict[str, Any],
        final_user_text: str,
        history_transcripts: dict[str, str] | None = None,
        temperature: float = 0.0,
    ) -> tuple[list[dict[str, Any]], str]:
        """Run the agent on a single `ToolEnv` instance and return `(tool_calls, state_hash)`."""
        cleaned_user_text = self.preprocessor.preprocess(final_user_text, sample)
        cleaned_hist_transcripts = {
            k: self.preprocessor.preprocess(v, sample) for k, v in (history_transcripts or {}).items()
        }

        if cleaned_hist_transcripts and sample.get("history"):
            for turn in sample["history"]:
                if not turn.get("audio_text"):
                    aud = turn.get("audio", "")
                    if aud in cleaned_hist_transcripts:
                        turn["audio_text"] = cleaned_hist_transcripts[aud]

        state = ReActSessionState(
            sample=sample,
            env=env,
            cleaned_user_text=cleaned_user_text,
            cleaned_hist_transcripts=cleaned_hist_transcripts,
            catalog_tools_map=self.catalog_tools_map,
            catalog_apps_map=self.catalog_apps_map,
            router=self.router,
            max_steps=self.max_steps,
            enable_fuzzy_normalizer=self.enable_fuzzy_normalizer,
            enable_hybrid_engine=self.enable_hybrid_engine,
            enable_action_triage=self.enable_action_triage,
            max_apps=self.max_apps,
        )

        while not state.finished:
            reply = self.llm.generate(state.messages, temperature=temperature)
            step_react_state(state, reply)

        return state.tool_calls, state.state_hash
