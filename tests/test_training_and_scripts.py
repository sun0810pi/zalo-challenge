"""Regression tests for SFT token weighting, GRPO KL/reward helpers, prompt builder, and Hammer masking."""
from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompt_builder import build_allowed_tools_block


class _SubwordTokenizer:
    """Minimal deterministic tokenizer splitting strings into 3-char chunks to test multi-token spans."""

    def __init__(self) -> None:
        self._id_to_tok: dict[int, str] = {}
        self._tok_to_id: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for i in range(0, len(text), 3):
            tok = text[i : i + 3]
            if tok not in self._tok_to_id:
                tid = len(self._tok_to_id) + 1
                self._tok_to_id[tok] = tid
                self._id_to_tok[tid] = tok
            ids.append(self._tok_to_id[tok])
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self._id_to_tok[tid] for tid in token_ids)


class _SpyRouter:
    def __init__(self) -> None:
        self.received_descriptions: dict[str, str] | None = None

    def route_sample_apps(
        self,
        query: str,
        sample: dict,
        catalog_tools_map: dict,
        installed_app_descriptions: dict[str, str] | None = None,
    ) -> tuple[list[str], list[str]]:
        self.received_descriptions = installed_app_descriptions
        apps = list(sample.get("context", {}).get("installed_apps", {}).keys())
        return apps, []


class TestTrainingAndScripts(unittest.TestCase):
    def test_build_allowed_tools_block_passes_app_descriptions(self) -> None:
        catalog_tools_map = {
            "contacts.search": {
                "name": "contacts.search",
                "kind": "read",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        }
        catalog_apps_map = {
            "contacts": {"id": "contacts", "name_vi": "Danh bạ", "group": "Social", "description": "Quản lý danh bạ"}
        }
        sample = {"context": {"installed_apps": {"contacts": {"tools": ["contacts.search"]}}}}
        spy = _SpyRouter()
        block = build_allowed_tools_block(
            sample=sample,
            catalog_tools_map=catalog_tools_map,
            user_query="tìm số điện thoại",
            router=spy,  # type: ignore[arg-type]
            catalog_apps_map=catalog_apps_map,
        )
        self.assertIn("Quản lý danh bạ", block)
        self.assertIsNotNone(spy.received_descriptions)
        self.assertIn("contacts", spy.received_descriptions or {})

    def test_build_allowed_tools_block_keeps_all_apps_by_default_and_prunes_when_set(self) -> None:
        apps = ["contacts", "alarm", "calendar", "weather", "stocks", "parking"]
        catalog_tools_map = {
            f"{app}.run": {
                "name": f"{app}.run",
                "kind": "read",
                "parameters": {"type": "object", "properties": {}},
            }
            for app in apps
        }
        sample = {
            "context": {
                "installed_apps": {app: {"tools": [f"{app}.run"]} for app in apps}
            }
        }
        spy = _SpyRouter()
        full_block = build_allowed_tools_block(
            sample=sample,
            catalog_tools_map=catalog_tools_map,
            user_query="kiểm tra lịch",
            router=spy,  # type: ignore[arg-type]
        )
        for app in apps:
            self.assertIn(f"({app})", full_block)

        pruned_block = build_allowed_tools_block(
            sample=sample,
            catalog_tools_map=catalog_tools_map,
            user_query="kiểm tra lịch",
            router=spy,  # type: ignore[arg-type]
            max_apps=4,
        )
        included = [app for app in apps if f"({app})" in pruned_block]
        self.assertEqual(len(included), 4)

    def test_sft_multitoken_span_weighting(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("local_train_sft", ROOT / "train_sft.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        AssistantOnlyDataCollator = mod.AssistantOnlyDataCollator

        tok = _SubwordTokenizer()
        collator = AssistantOnlyDataCollator(tokenizer=tok, max_seq_len=512, weighted_loss=True)
        seg = '<tool_call>\n{"name": "sys.ask_missing_info", "arguments": {"id": "ord_991"}}\n</tool_call>'
        ids = tok.encode(seg)
        labels = list(ids)  # all active assistant tokens
        weights = collator.compute_weights_for_ids(ids, labels)

        self.assertEqual(len(weights), len(ids))
        # Multi-token spans (<tool_call>, sys.ask_missing_info, "name") should hit 2.5
        self.assertIn(2.5, weights)
        # ord_991 should hit at least 2.0
        self.assertTrue(any(w >= 2.0 for w in weights))
        # Outside labels (-100) must get 0.0
        prefixed_weights = collator.compute_weights_for_ids(ids, [-100] + labels[1:])
        self.assertEqual(prefixed_weights[0], 0.0)

    def test_grpo_k3_kl_is_non_negative(self) -> None:
        import importlib.util
        import torch

        # Stub zaic_tools / evaluate dependency if train/ isn't present in this test env
        fake_zaic = types.ModuleType("zaic_tools")
        fake_zaic.ToolEnv = object
        fake_zaic_val = types.ModuleType("zaic_tools.validate")
        fake_zaic_val.catalog_tools = lambda x: {}
        with patch.dict(sys.modules, {"zaic_tools": fake_zaic, "zaic_tools.validate": fake_zaic_val}):
            spec = importlib.util.spec_from_file_location("local_train_grpo_rlvr", ROOT / "train_grpo_rlvr.py")
            assert spec and spec.loader
            train_grpo_rlvr = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_grpo_rlvr)

        curr_lp = torch.tensor([[-0.2, -1.5, -0.8]])
        ref_lp = torch.tensor([[-0.5, -0.4, -0.8]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        kl = train_grpo_rlvr.compute_kl_k3(curr_lp, ref_lp, mask)
        self.assertGreaterEqual(float(kl.item()), 0.0)

    def test_hammer_mask_catalog_tool_spec_masks_xref_and_one_of(self) -> None:
        from scripts.data.build_hammer_when2call_dataset import mask_catalog_tool_spec

        spec = {
            "name": "ecommerce.request_return",
            "kind": "write",
            "params": {
                "order_id": {
                    "type": "string",
                    "x-ref": "ecommerce.orders (cần gọi ecommerce.order_status trước để lấy)",
                },
                "reason": {"type": "string"},
            },
            "required": ["order_id"],
            "x-one-of": [["order_id", "reason"]],
        }
        t_map = {
            "ecommerce.request_return": "svc_12_ecom.fn_01",
            "ecommerce.order_status": "svc_12_ecom.fn_02",
        }
        p_map = {
            "ecommerce.request_return": {
                "order_id": "p_12_01",
                "reason": "p_12_02",
            }
        }
        masked = mask_catalog_tool_spec(spec, t_map, p_map)
        self.assertEqual(masked["name"], "svc_12_ecom.fn_01")
        self.assertEqual(masked["required"], ["p_12_01"])
        self.assertEqual(masked["x-one-of"], [["p_12_01", "p_12_02"]])
        self.assertIn("svc_12_ecom.fn_02", masked["params"]["p_12_01"]["x-ref"])
        self.assertNotIn("ecommerce.order_status", masked["params"]["p_12_01"]["x-ref"])


if __name__ == "__main__":
    unittest.main()
