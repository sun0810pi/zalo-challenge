"""Regression tests for PureSchemaStateTracker & schema/argument guardrails."""
from __future__ import annotations

import unittest

from src.guardrail import PureSchemaStateTracker, normalize_arguments, parse_tool_calls


class TestGuardrail(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {
            "grocery.search": {
                "name": "grocery.search",
                "kind": "read",
                "returns": "grocery.products[]",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            "grocery.add_to_cart": {
                "name": "grocery.add_to_cart",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "x-ref": "grocery.products"},
                        "quantity": {"type": "integer", "minimum": 1},
                    },
                    "required": ["product_id", "quantity"],
                },
            },
            "pantry.get_items": {
                "name": "pantry.get_items",
                "kind": "read",
                "returns": "pantry.items[]",
                "parameters": {"type": "object", "properties": {}},
            },
            "ecommerce.request_return": {
                "name": "ecommerce.request_return",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "x-ref": "ecommerce.orders"},
                        "reason": {"type": "string"},
                    },
                    "required": ["order_id"],
                },
            },
            "ecommerce.order_status": {
                "name": "ecommerce.order_status",
                "kind": "read",
                "returns": "ecommerce.orders[]",
                "parameters": {"type": "object", "properties": {}},
            },
            "banking.transfer": {
                "name": "banking.transfer",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_account": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                    "required": ["to_account", "amount"],
                },
            },
            "stocks.get_quote": {
                "name": "stocks.get_quote",
                "kind": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "shares": {"type": "number"},
                    },
                },
            },
        }

        self.sample = {
            "id": "guard_001",
            "context": {
                "current_time": "2026-05-01T09:00:00+07:00",
                "installed_apps": {
                    "grocery": {"tools": ["grocery.search", "grocery.add_to_cart"]},
                    "pantry": {"tools": ["pantry.get_items"]},
                    "ecommerce": {"tools": ["ecommerce.order_status", "ecommerce.request_return"]},
                    "banking": {"tools": ["banking.transfer"]},
                    "stocks": {"tools": ["stocks.get_quote"]},
                },
            },
        }

    def test_blocks_non_ascii_phrase_id(self) -> None:
        """Reproduces public_0208: grocery.add_to_cart(product_id='trà hoa cúc')."""
        tracker = PureSchemaStateTracker(self.sample, "Mua giúp tôi trà hoa cúc", self.catalog)
        err = tracker.check_pre_execution(
            "grocery.add_to_cart", {"product_id": "trà hoa cúc", "quantity": 1}
        )
        self.assertIsNotNone(err)

    def test_blocks_cross_collection_id_leakage(self) -> None:
        """Reproduces public_0248: pantry ID 'pan_06' leaked into grocery.add_to_cart."""
        tracker = PureSchemaStateTracker(self.sample, "Thêm trà vào giỏ hàng", self.catalog)
        tracker.record_observation(
            "pantry.get_items",
            {},
            {"ok": True, "items": [{"id": "pan_06", "name": "Trà xanh"}]},
        )
        err = tracker.check_pre_execution(
            "grocery.add_to_cart", {"product_id": "pan_06", "quantity": 1}
        )
        self.assertIsNotNone(err)

    def test_allows_matching_collection_id(self) -> None:
        tracker = PureSchemaStateTracker(self.sample, "Thêm trà hoa cúc", self.catalog)
        tracker.record_observation(
            "grocery.search",
            {"query": "trà hoa cúc"},
            {"ok": True, "products": [{"id": "prod_101", "name": "Trà hoa cúc"}]},
        )
        err = tracker.check_pre_execution(
            "grocery.add_to_cart", {"product_id": "prod_101", "quantity": 2}
        )
        self.assertIsNone(err)

    def test_blocks_fabricated_numeric_order_id(self) -> None:
        """Reproduces public_0261: fabricated order_id='123456' without reading ecommerce.orders."""
        tracker = PureSchemaStateTracker(self.sample, "Trả hàng đơn vừa mua giúp tôi", self.catalog)
        err = tracker.check_pre_execution(
            "ecommerce.request_return", {"order_id": "123456", "reason": "lỗi"}
        )
        self.assertIsNotNone(err)

    def test_normalize_arguments_decimal_vs_thousand_int(self) -> None:
        norm_dec = normalize_arguments(
            "stocks.get_quote", {"symbol": "fpt", "shares": "1.5"}, self.catalog
        )
        self.assertEqual(norm_dec["shares"], 1.5)

        norm_int = normalize_arguments(
            "grocery.add_to_cart", {"product_id": "prod_1", "quantity": "250.000"}, self.catalog
        )
        self.assertEqual(norm_int["quantity"], 250000)

    def test_parse_tool_calls(self) -> None:
        reply = '<think>ok</think>\n<tool_call>\n{"name": "grocery.search", "arguments": {"query": "trà"}}\n</tool_call>'
        calls = parse_tool_calls(reply)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "grocery.search")


if __name__ == "__main__":
    unittest.main()
