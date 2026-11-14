"""Regression tests for agent prompt preparation, ReActSessionState, and loop guards."""
from __future__ import annotations

import json
import unittest
from typing import Any

from src.agent import LLMBackend, ReActSessionState, build_preprocessor, step_react_state


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Deterministic lightweight token count (~1 token per 2 chars)
        return list(range(max(1, len(text) // 2)))


class _FakeEnv:
    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses = list(responses or [])

    def validate(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    def call(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, tool_args))
        if self._responses:
            return self._responses.pop(0)
        return {"ok": True, "quotes": [{"symbol": "FPT", "price": 120000}]}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return [{"name": n, "arguments": a} for n, a in self.calls]

    def get_state_hash(self) -> str:
        return "mock_state_hash"


class TestAgentAndBatchedLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {
            "stocks.get_quote": {
                "name": "stocks.get_quote",
                "kind": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            },
            "grocery.add_to_cart": {
                "name": "grocery.add_to_cart",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "x-ref": "grocery.products"},
                        "quantity": {"type": "integer"},
                    },
                    "required": ["product_id", "quantity"],
                },
            },
            "sys.ask_missing_info": {
                "name": "sys.ask_missing_info",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "params": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["tool_name", "params"],
                },
            },
        }
        self.sample = {
            "id": "unit_001",
            "audio_text": "Xem giá cổ phiếu FPT giúp tôi",
            "context": {
                "current_time": "2026-05-01T09:00:00+07:00",
                "installed_apps": {
                    "stocks": {"tools": ["stocks.get_quote"]},
                    "grocery": {"tools": ["grocery.add_to_cart"]},
                },
            },
        }

    def test_prepare_prompt_preserves_system_and_user_headers(self) -> None:
        backend = object.__new__(LLMBackend)
        backend.tokenizer = _FakeTokenizer()
        backend.max_model_len = 250  # forces dropping old middle turns

        messages = [
            {"role": "system", "content": "SYSTEM_HEADER_MUST_STAY"},
            {"role": "user", "content": "ORIGINAL_USER_REQUEST"},
            {"role": "assistant", "content": "OLD_ASST_" + ("a" * 300)},
            {"role": "user", "content": "OLD_TOOL_RESP_" + ("b" * 300)},
            {"role": "assistant", "content": "RECENT_ASST"},
            {"role": "user", "content": "RECENT_TOOL_RESP"},
        ]
        prompt = backend._prepare_prompt(messages)
        self.assertIn("SYSTEM_HEADER_MUST_STAY", prompt)
        self.assertIn("ORIGINAL_USER_REQUEST", prompt)
        self.assertNotIn("OLD_ASST_", prompt)
        self.assertIn("RECENT_TOOL_RESP", prompt)

    def test_postprocess_generation(self) -> None:
        raw = '<think>plan</think>\n<tool_call>\n{"name": "stocks.get_quote", "arguments": {"symbol": "FPT"}}</tool_call><|im_end|>'
        cleaned = LLMBackend._postprocess_generation(raw)
        self.assertNotIn("<|im_end|>", cleaned)
        self.assertTrue(cleaned.endswith("</tool_call>"))

    def test_build_preprocessor_safe_mode(self) -> None:
        prep = build_preprocessor(mode="safe", sim_threshold=0.8)
        self.assertIsNotNone(prep)

    def test_consecutive_same_read_tool_circuit_breaker(self) -> None:
        """Reproduces public_0443/0643 runaway read loop: >= 3 identical consecutive read tool calls finish cleanly."""
        env = _FakeEnv()
        state = ReActSessionState(
            sample=self.sample,
            env=env,
            cleaned_user_text="Xem giá cổ phiếu FPT",
            cleaned_hist_transcripts={},
            catalog_tools_map=self.catalog,
            max_steps=8,
        )
        for idx in range(1, 7):
            if state.finished:
                break
            reply = (
                f"<think>try {idx}</think>\n"
                f'<tool_call>\n{{"name": "stocks.get_quote", "arguments": {{"symbol": "SYM{idx}"}}}}\n</tool_call>'
            )
            step_react_state(state, reply)

        self.assertTrue(state.finished)
        self.assertEqual(len(env.calls), 3)

    def test_suppressed_guardrail_retry_terminates_instead_of_looping(self) -> None:
        env = _FakeEnv()
        state = ReActSessionState(
            sample=self.sample,
            env=env,
            cleaned_user_text="Mua trà hoa cúc",
            cleaned_hist_transcripts={},
            catalog_tools_map=self.catalog,
            max_steps=8,
        )
        bad_reply = (
            "<think>add</think>\n"
            '<tool_call>\n{"name": "grocery.add_to_cart", "arguments": {"product_id": "trà hoa cúc", "quantity": 1}}\n</tool_call>'
        )
        for _ in range(4):
            if not state.finished:
                step_react_state(state, bad_reply)

        self.assertTrue(state.finished)
        self.assertGreaterEqual(state.guardrail_retries, 3)
        self.assertEqual(len(env.calls), 0)

    def test_resolve_max_lora_rank_autodetects_adapter_config_and_buckets(self):
        import json
        import tempfile
        from pathlib import Path
        from src.agent import _resolve_max_lora_rank

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertEqual(_resolve_max_lora_rank(str(root)), 64)

            (root / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
            self.assertEqual(_resolve_max_lora_rank(str(root)), 64)

            (root / "adapter_config.json").write_text(json.dumps({"r": 96}), encoding="utf-8")
            self.assertEqual(_resolve_max_lora_rank(str(root)), 128)

            (root / "adapter_config.json").write_text(json.dumps({"r": 128}), encoding="utf-8")
            self.assertEqual(_resolve_max_lora_rank(str(root)), 128)

            (root / "adapter_config.json").write_text("{invalid json", encoding="utf-8")
            self.assertEqual(_resolve_max_lora_rank(str(root)), 64)

            self.assertEqual(_resolve_max_lora_rank(str(root), override=256), 256)
            self.assertEqual(_resolve_max_lora_rank(None), 64)


if __name__ == "__main__":
    unittest.main()
