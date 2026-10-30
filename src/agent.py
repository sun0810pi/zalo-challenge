"""Deterministic ReAct Tool-Calling Agent for ZAC 2026 LifeMate (< 5B LLM).
Features:
- Pure schema-driven state guardrails (no dataset-specific prefix dependencies)
- Dynamic two-tier semantic tool routing (recommended vs secondary installed tools)
- Robust context-grounded language preprocessing
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

from .guardrail import PureSchemaStateTracker, normalize_arguments, parse_tool_calls
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
        enable_thinking: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        if lora_path:
            lp = Path(lora_path)
            if not (lp / "adapter_config.json").is_file():
                sub_ckpts = sorted(
                    lp.glob("checkpoint-*/adapter_config.json"),
                    key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else 0,
                )
                if sub_ckpts:
                    resolved_sub = sub_ckpts[-1].parent.resolve()
                    print(f"[LLMBackend] Tự động dùng LoRA checkpoint mới nhất: {resolved_sub}")
                    lora_path = str(resolved_sub)
                else:
                    print(
                        f"[CẢNH BÁO] Không tìm thấy file 'adapter_config.json' trong '{lora_path}'. "
                        f"Sẽ chạy model gốc '{model_name_or_path}' (Zero-shot) thay vì crash!"
                    )
                    lora_path = None
            else:
                lora_path = str(lp.resolve())
        self.lora_path = lora_path
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_thinking = enable_thinking
        self.backend = backend
        self.tokenizer = None
        self.model = None
        self._vllm_sampling_params = None
        self._vllm_lora_request = None
        self._load()

    def _load(self) -> None:
        from transformers import AutoTokenizer

        tokenizer_src = (
            self.lora_path
            if (self.lora_path and (Path(self.lora_path) / "tokenizer_config.json").is_file())
            else self.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)

        if self.backend in ("auto", "vllm"):
            try:
                from vllm import LLM, SamplingParams
                from vllm.lora.request import LoRARequest

                self.model = LLM(
                    model=self.model_name_or_path,
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    dtype="bfloat16",
                    trust_remote_code=True,
                    enable_lora=bool(self.lora_path),
                    max_lora_rank=64,
                    enforce_eager=True,
                    seed=42,
                )
                self._vllm_sampling_params = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.max_new_tokens,
                    skip_special_tokens=False,
                    stop=["</tool_call>", "<|im_end|>", "<|endoftext|>"],
                    include_stop_str_in_output=True,
                    seed=42,
                )
                if self.lora_path:
                    self._vllm_lora_request = LoRARequest("zac_adapter", 1, self.lora_path)
                self.backend = "vllm"
                return
            except Exception as e:
                if self.backend == "vllm":
                    raise
                print(f"[LLMBackend] vLLM không khả dụng ({e}), chuyển sang transformers...")

        import torch
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=dtype if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if self.lora_path:
            try:
                import peft.import_utils
                import peft.tuners.lora.torchao

                peft.import_utils.is_torchao_available = lambda: False
                peft.tuners.lora.torchao.is_torchao_available = lambda: False
            except Exception:
                pass
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.backend = "transformers"

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Render ChatML prompt (respecting 16,384 token limit) and generate deterministically."""
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
            outputs = self.model.generate(
                [prompt],
                sampling_params=self._vllm_sampling_params,
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
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=["</tool_call>", "<|im_end|>", "<|endoftext|>"],
                tokenizer=self.tokenizer,
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


class LifeMateAgent:
    """ReAct agent with single-tool execution per step, pure schema-driven state guardrail,
    and two-tier semantic tool routing.
    """

    def __init__(
        self,
        llm: LLMBackend,
        catalog_tools_map: dict[str, dict[str, Any]],
        max_steps: int = 8,
        preprocessor: Any | None = None,
        router: SemanticToolRouter | None = None,
    ):
        self.llm = llm
        self.catalog_tools_map = catalog_tools_map
        self.max_steps = max_steps
        self.preprocessor = preprocessor or ContextLanguagePreprocessorA()
        self.router = router or SemanticToolRouter(top_k_recommended=4)

    @staticmethod
    def _augment_empty_read_hint(
        tool_name: str,
        tool_args: dict[str, Any],
        exec_res: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach a non-destructive `_hint` if a parameterized read/search call returned an empty list."""
        if not isinstance(exec_res, dict) or not tool_args:
            return exec_res

        for k, v in exec_res.items():
            if isinstance(v, list) and len(v) == 0:
                res_copy = copy.deepcopy(exec_res)
                res_copy["_hint"] = (
                    "Kết quả tra cứu rỗng (có thể do từ khóa tìm kiếm quá dài hoặc sai lệch âm ASR). "
                    "Hãy thử tra cứu lại với từ khóa ngắn gọn hơn hoặc gọi công cụ phù hợp khác."
                )
                return res_copy
        return exec_res

    def run_sample(
        self,
        env: Any,
        sample: dict[str, Any],
        final_user_text: str,
        history_transcripts: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Run the agent on a single `ToolEnv` instance and return `(tool_calls, state_hash)`."""
        # 1. Language Preprocessing
        cleaned_user_text = self.preprocessor.preprocess(final_user_text, sample)
        cleaned_hist_transcripts = {}
        for k, v in (history_transcripts or {}).items():
            cleaned_hist_transcripts[k] = self.preprocessor.preprocess(v, sample)

        # If Whisper output was purely stripped hallucination boilerplate, do not call tools
        if not cleaned_user_text:
            return env.get_tool_calls(), env.get_state_hash()

        # 2. Build initial ChatML messages with Two-Tier tool schemas
        messages = build_initial_messages(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=self.catalog_tools_map,
            history_transcripts=cleaned_hist_transcripts,
            router=self.router,
        )

        cur_time = sample.get("context", {}).get("current_time", "")
        tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"
        seen_calls: set[str] = set()
        guardrail_retries = 0

        # 3. Initialize pure schema-driven state tracker
        state_tracker = PureSchemaStateTracker(
            sample=sample,
            final_user_text=cleaned_user_text,
            catalog_tools_map=self.catalog_tools_map,
        )

        for _step in range(self.max_steps):
            reply = self.llm.generate(messages)
            raw_calls = parse_tool_calls(reply)

            # No tool call emitted -> agent completed all tasks
            if not raw_calls:
                break

            # STRICT REACT: Only execute the FIRST tool call emitted in this step
            call = raw_calls[0]
            t_name = call["name"]
            t_args = normalize_arguments(
                t_name,
                call.get("arguments") or {},
                self.catalog_tools_map,
                tz_offset=tz_offset,
            )

            messages.append({"role": "assistant", "content": reply})

            # Prevent infinite loops on identical calls
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

            # Zero-cost schema validation via env.validate()
            val_res = env.validate(t_name, t_args)
            if not val_res.get("ok"):
                guardrail_retries += 1
                err_msg = val_res.get("error", "Sai định dạng schema")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_response>\n"
                            f"{json.dumps({'ok': False, 'validation_error': err_msg, 'hint': 'Hãy sửa lại đúng tên tham số/kiểu dữ liệu hoặc gọi sys.ask_missing_info nếu người dùng chưa cung cấp đủ thông tin.'}, ensure_ascii=False)}\n"
                            f"</tool_response>"
                        ),
                    }
                )
                if guardrail_retries >= 3:
                    break
                continue

            # Zero-cost pure schema state check (prevents done: False soft-failures)
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

            # Validated & checked! Execute on the official simulator
            seen_calls.add(call_sig)
            exec_res = env.call(t_name, t_args)
            state_tracker.record_observation(t_name, t_args, exec_res)
            prompt_res = self._augment_empty_read_hint(t_name, t_args, exec_res)
            messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{json.dumps(prompt_res, ensure_ascii=False)}\n</tool_response>",
                }
            )

            if t_name == "sys.ask_missing_info" or guardrail_retries >= 3:
                break

        return env.get_tool_calls(), env.get_state_hash()
