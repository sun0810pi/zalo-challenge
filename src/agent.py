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
    ):
        self.model_name_or_path = model_name_or_path
        self.lora_path = lora_path
        self.backend = backend
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

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

            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if self.lora_path and Path(self.lora_path).is_dir():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model.eval()
        self.backend = "transformers"

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Render ChatML prompt and generate with specified temperature."""
        prompt = render_chatml_preserving_think(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
        )

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        max_input_tokens = self.max_model_len - self.max_new_tokens - 32
        if len(input_ids) > max_input_tokens:
            input_ids = input_ids[-max_input_tokens:]
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        if self.backend == "vllm":
            sampling_params = self._vllm_sampling_params
            if temperature > 0.0:
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=0.9,
                    max_tokens=self.max_new_tokens,
                    stop=["<|im_end|>", "<|endoftext|>"],
                )
            outputs = self.model.generate(
                [prompt],
                sampling_params=sampling_params,
                lora_request=self._vllm_lora_request,
                use_tqdm=False,
            )
            text = outputs[0].outputs[0].text
            for eos_tok in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(eos_tok, "")
            text = text.strip()
            if "<tool_call>" in text and not text.endswith("</tool_call>"):
                text += "\n</tool_call>"
            return text

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "stop_strings": ["</tool_call>", "<|im_end|>", "<|endoftext|>"],
                "tokenizer": self.tokenizer,
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
        for eos_tok in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(eos_tok, "")
        text = text.strip()
        if "</tool_call>" in text:
            text = text[: text.find("</tool_call>") + len("</tool_call>")]
        elif "<tool_call>" in text and not text.endswith("</tool_call>"):
            text += "\n</tool_call>"
        return text

    def generate_batch(self, list_of_messages: list[list[dict[str, str]]]) -> list[str]:
        """Batch generation for multiple samples simultaneously via vLLM continuous batching."""
        if not list_of_messages:
            return []

        prompts = []
        max_input_tokens = self.max_model_len - self.max_new_tokens - 32
        for messages in list_of_messages:
            prompt = render_chatml_preserving_think(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > max_input_tokens:
                input_ids = input_ids[-max_input_tokens:]
                prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            prompts.append(prompt)

        if self.backend == "vllm":
            outputs = self.model.generate(
                prompts,
                sampling_params=self._vllm_sampling_params,
                lora_request=self._vllm_lora_request,
                use_tqdm=False,
            )
            results = []
            for out in outputs:
                text = out.outputs[0].text
                for eos_tok in ("<|im_end|>", "<|endoftext|>"):
                    text = text.replace(eos_tok, "")
                text = text.strip()
                if "</tool_call>" in text:
                    text = text[: text.find("</tool_call>") + len("</tool_call>")]
                elif "<tool_call>" in text and not text.endswith("</tool_call>"):
                    text += "\n</tool_call>"
                results.append(text)
            return results

        # Fallback to serial generation if transformers backend
        return [self.generate(msgs) for msgs in list_of_messages]


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
    ):
        self.llm = llm
        self.catalog_tools_map = catalog_tools_map
        self.max_steps = max_steps
        self.preprocessor = preprocessor or ContextLanguagePreprocessorA()
        self.router = router or SemanticToolRouter(top_k_recommended=4)
        self.catalog_apps_map = catalog_apps_map

        # Switches for the 5 Solutions
        self.enable_fuzzy_normalizer = enable_fuzzy_normalizer
        self.enable_hybrid_engine = enable_hybrid_engine
        self.enable_action_triage = enable_action_triage

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
        # 1. Language Preprocessing
        cleaned_user_text = self.preprocessor.preprocess(final_user_text, sample)
        cleaned_hist_transcripts = {}
        for k, v in (history_transcripts or {}).items():
            cleaned_hist_transcripts[k] = self.preprocessor.preprocess(v, sample)

        if not cleaned_user_text:
            return env.get_tool_calls(), env.get_state_hash()

        # Solution 4 Check: Out of scope request -> immediate 0-call return
        if self.enable_action_triage:
            is_oos, _ = self.action_triage.check_out_of_scope(sample, cleaned_user_text)
            if is_oos:
                return env.get_tool_calls(), env.get_state_hash()

        # Solution 3 Check: Instant stop on complete cancellation
        if self.enable_hybrid_engine:
            if self.hybrid_engine.is_complete_cancellation(cleaned_user_text):
                return env.get_tool_calls(), env.get_state_hash()

        # 2. Build initial ChatML messages
        messages = build_initial_messages(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=self.catalog_tools_map,
            history_transcripts=cleaned_hist_transcripts,
            router=self.router,
            catalog_apps_map=self.catalog_apps_map,
        )

        cur_time = sample.get("context", {}).get("current_time", "")
        tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"
        seen_calls: set[str] = set()
        guardrail_retries = 0
        suppress_sms = False

        # 3. Initialize pure schema-driven state tracker
        state_tracker = PureSchemaStateTracker(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=self.catalog_tools_map,
        )

        for _step in range(self.max_steps):
            reply = self.llm.generate(messages, temperature=temperature)
            raw_calls = parse_tool_calls(reply)

            if not raw_calls:
                break

            call = raw_calls[0]
            t_name = call["name"]
            t_args = normalize_arguments(
                t_name,
                call.get("arguments") or {},
                self.catalog_tools_map,
                tz_offset=tz_offset,
            )

            # Solution 3: DateTime Auto-Repair
            if self.enable_hybrid_engine:
                for k in ("time", "start", "end", "schedule_time"):
                    if k in t_args and isinstance(t_args[k], str):
                        t_args[k] = self.hybrid_engine.normalize_iso_datetime(t_args[k], tz_offset)

            # Solution 4: Guard sys.ask_missing_info
            if t_name == "sys.ask_missing_info" and self.enable_action_triage:
                allowed, triage_reason = self.action_triage.guard_ask_missing_info(
                    t_args, sample, seen_calls, self.catalog_tools_map, state_tracker.observed_entities
                )
                if not allowed:
                    guardrail_retries += 1
                    messages.append({"role": "assistant", "content": reply})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"<tool_response>\n"
                                f"{json.dumps({'ok': False, 'validation_error': triage_reason, 'hint': 'Hãy chọn đối tượng phù hợp nhất từ kết quả tra cứu để thực thi công cụ tiếp theo.'}, ensure_ascii=False)}\n"
                                f"</tool_response>"
                            ),
                        }
                    )
                    if guardrail_retries >= 3:
                        break
                    continue

            # Solution 3: Stock Math & Bill Payment adjustments
            if self.enable_hybrid_engine:
                if t_name == "stocks.place_order":
                    t_args = self.hybrid_engine.adjust_stock_order(
                        t_args, cleaned_user_text, state_tracker.observed_entities
                    )
                elif t_name == "bill_payment.pay":
                    t_args, supp_sms = self.hybrid_engine.adjust_bill_payment(
                        t_args, cleaned_user_text, state_tracker.observed_entities, state_tracker.balances
                    )
                    if t_args is None:
                        suppress_sms = True
                        continue
                    if supp_sms:
                        suppress_sms = True

                if t_name == "sms.send" and suppress_sms:
                    continue

                if t_name == "parking.reserve" and any("ride_hailing.book" in s and "bike" in s for s in seen_calls):
                    continue

            messages.append({"role": "assistant", "content": reply})

            call_sig = f"{t_name}:{json.dumps(t_args, sort_keys=True, ensure_ascii=False)}"
            if call_sig in seen_calls:
                guardrail_retries += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response>\n"
                            f"{json.dumps({'ok': False, 'error': 'Bạn vừa gọi đúng lệnh này rồi, không gọi lặp lại. Hãy chuyển sang bước tiếp theo hoặc kết thúc.'}, ensure_ascii=False)}\n"
                            f"</tool_response>"
                        ),
                    }
                )
                if guardrail_retries >= 3:
                    break
                continue

            # Zero-cost schema validation
            val_res = env.validate(t_name, t_args)
            if not val_res.get("ok"):
                guardrail_retries += 1
                err_msg = val_res.get("error", "Sai định dạng schema")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response>\n"
                            f"{json.dumps({'ok': False, 'validation_error': err_msg, 'hint': 'Hãy sửa lại đúng tên tham số/kiểu dữ liệu.'}, ensure_ascii=False)}\n"
                            f"</tool_response>"
                        ),
                    }
                )
                if guardrail_retries >= 3:
                    break
                continue

            # Pure schema state precheck
            pre_err = state_tracker.check_pre_execution(t_name, t_args)
            if pre_err is not None:
                guardrail_retries += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response>\n"
                            f"{json.dumps({'ok': False, 'precheck_error': pre_err}, ensure_ascii=False)}\n"
                            f"</tool_response>"
                        ),
                    }
                )
                if guardrail_retries >= 3:
                    break
                continue

            seen_calls.add(call_sig)
            exec_res = env.call(t_name, t_args)
            state_tracker.record_observation(t_name, t_args, exec_res)

            # Solution 3: Weather condition check
            if self.enable_hybrid_engine:
                if t_name in ("weather.get_forecast", "weather.get_current"):
                    if not self.hybrid_engine.check_weather_condition(cleaned_user_text, exec_res):
                        break

                if t_name == "smart_home.fan_set":
                    extra_fans = self.hybrid_engine.expand_smart_home_fans(
                        env, cleaned_user_text, state_tracker.observed_entities, t_args
                    )
                    for ef in extra_fans:
                        seen_calls.add(f"{ef['name']}:{json.dumps(ef['arguments'], sort_keys=True, ensure_ascii=False)}")
                        ef_res = env.call(ef["name"], ef["arguments"])
                        state_tracker.record_observation(ef["name"], ef["arguments"], ef_res)

            # Solution 2: Fuzzy Read Recovery
            if self.enable_fuzzy_normalizer:
                prompt_res = self.entity_normalizer.recover_empty_read(
                    t_name, t_args, exec_res, sample, state_tracker.observed_entities
                )
            else:
                prompt_res = exec_res

            messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{json.dumps(prompt_res, ensure_ascii=False)}\n</tool_response>",
                }
            )

            if t_name == "sys.ask_missing_info" or guardrail_retries >= 3:
                break

        return env.get_tool_calls(), env.get_state_hash()
