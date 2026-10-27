"""Deterministic ReAct Tool-Calling Agent for ZAC 2026 LifeMate (< 5B LLM)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Any

# Bắt buộc cho vLLM V1 trên Linux/Colab để tránh lỗi "Cannot re-initialize CUDA in forked subprocess"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from .guardrail import normalize_arguments, parse_tool_calls
from .prompt_builder import build_initial_messages


def fix_all_seeds(seed: int = 42):
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
        # Không gọi torch.cuda.is_available() trước khi vLLM spawn subprocess nếu chưa khởi tạo CUDA
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
        enable_thinking: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        if lora_path:
            lp = Path(lora_path)
            if not (lp / "adapter_config.json").is_file():
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
        self.enable_thinking = enable_thinking
        self.backend = backend
        self.tokenizer = None
        self.model = None
        self._vllm_sampling_params = None
        self._vllm_lora_request = None
        self._load()

    def _load(self):
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
                    gpu_memory_utilization=0.72,
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
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Enforce 16,384 reading budget
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
            return outputs[0].outputs[0].text.strip()

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
            )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


class LifeMateAgent:
    """ReAct agent with `env.validate` sandbox guardrail and single-simulator enforcement."""

    def __init__(
        self,
        llm: LLMBackend,
        catalog_tools_map: dict[str, dict[str, Any]],
        max_steps: int = 6,
    ):
        self.llm = llm
        self.catalog_tools_map = catalog_tools_map
        self.max_steps = max_steps

    def run_sample(
        self,
        env: Any,
        sample: dict[str, Any],
        final_user_text: str,
        history_transcripts: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Run the agent on a single `ToolEnv` instance and return `(tool_calls, state_hash)`."""
        messages = build_initial_messages(
            sample=sample,
            final_user_text=final_user_text,
            catalog_tools_map=self.catalog_tools_map,
            history_transcripts=history_transcripts,
        )

        cur_time = sample.get("context", {}).get("current_time", "")
        tz_offset = cur_time[19:] if len(cur_time) > 19 else "+07:00"
        seen_calls: set[str] = set()
        validation_retries = 0

        for _step in range(self.max_steps):
            reply = self.llm.generate(messages)
            raw_calls = parse_tool_calls(reply)

            # No tool call emitted -> agent is done
            if not raw_calls:
                break

            messages.append({"role": "assistant", "content": reply})
            tool_response_blocks: list[str] = []
            asked_missing_info = False

            for call in raw_calls:
                t_name = call["name"]
                t_args = normalize_arguments(
                    t_name,
                    call.get("arguments") or {},
                    self.catalog_tools_map,
                    tz_offset=tz_offset,
                )

                # Prevent duplicate identical tool calls in the same turn
                call_sig = f"{t_name}:{json.dumps(t_args, sort_keys=True, ensure_ascii=False)}"
                if call_sig in seen_calls:
                    tool_response_blocks.append(
                        f"<tool_response>\n{json.dumps({'ok': False, 'error': 'Bạn vừa gọi đúng lệnh này rồi, không gọi lặp lại.'}, ensure_ascii=False)}\n</tool_response>"
                    )
                    continue

                # Zero-cost schema check via env.validate() (NOT recorded in env.get_tool_calls())
                val_res = env.validate(t_name, t_args)
                if not val_res.get("ok"):
                    validation_retries += 1
                    err_msg = val_res.get("error", "Sai định dạng schema")
                    tool_response_blocks.append(
                        f"<tool_response>\n{json.dumps({'ok': False, 'validation_error': err_msg, 'hint': 'Hãy sửa lại đúng tên tham số/kiểu dữ liệu hoặc gọi sys.ask_missing_info nếu người dùng chưa cung cấp đủ thông tin.'}, ensure_ascii=False)}\n</tool_response>"
                    )
                    if validation_retries >= 3:
                        break
                    continue

                # Validated! Execute on the single official simulator
                seen_calls.add(call_sig)
                exec_res = env.call(t_name, t_args)
                tool_response_blocks.append(
                    f"<tool_response>\n{json.dumps(exec_res, ensure_ascii=False)}\n</tool_response>"
                )
                if t_name == "sys.ask_missing_info":
                    asked_missing_info = True
                    break

            if asked_missing_info or validation_retries >= 3:
                break

            if tool_response_blocks:
                messages.append({"role": "user", "content": "\n".join(tool_response_blocks)})
            else:
                break

        return env.get_tool_calls(), env.get_state_hash()
