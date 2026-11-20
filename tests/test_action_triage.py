"""Regression tests for ActionTriageBarrier & x-ref prerequisite resolution."""
from __future__ import annotations

import unittest

from src.action_triage import ActionTriageBarrier, _resolve_xref_read_tools


class TestActionTriage(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {
            "ecommerce.search": {
                "name": "ecommerce.search",
                "kind": "read",
                "returns": "ecommerce.products[]",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            "ecommerce.order_status": {
                "name": "ecommerce.order_status",
                "kind": "read",
                "returns": "ecommerce.orders[]",
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
            "contacts.search": {
                "name": "contacts.search",
                "kind": "read",
                "returns": "contacts.contacts[]",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            "phone.call": {
                "name": "phone.call",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                    },
                    "required": ["contact_id"],
                },
            },
        }
        self.sample = {
            "id": "test_001",
            "context": {
                "installed_apps": {
                    "ecommerce": {
                        "tools": [
                            "ecommerce.search",
                            "ecommerce.order_status",
                            "ecommerce.request_return",
                        ]
                    },
                    "contacts": {"tools": ["contacts.search"]},
                    "phone": {"tools": ["phone.call"]},
                }
            },
        }

    def test_resolve_exact_xref_read_tools(self) -> None:
        reads = _resolve_xref_read_tools("ecommerce.orders", "ecommerce.request_return", self.catalog)
        self.assertEqual(reads, ["ecommerce.order_status"])
        self.assertNotIn("ecommerce.search", reads)

    def test_blocks_ask_order_id_after_product_search_only(self) -> None:
        """Reproduces public_0279/0288: calling ecommerce.search (products) must NOT unlock asking order_id."""
        barrier = ActionTriageBarrier()
        ok, reason = barrier.guard_ask_missing_info(
            tool_call_args={"tool_name": "ecommerce.request_return", "params": ["order_id"]},
            sample=self.sample,
            seen_calls={'ecommerce.search:{"query": "tai nghe"}'},
            catalog_tools_map=self.catalog,
            observed_entities={},
        )
        self.assertFalse(ok)
        self.assertIn("ecommerce.order_status", reason)

    def test_allows_ask_order_id_after_order_status_called(self) -> None:
        barrier = ActionTriageBarrier()
        ok, _ = barrier.guard_ask_missing_info(
            tool_call_args={"tool_name": "ecommerce.request_return", "params": ["order_id"]},
            sample=self.sample,
            seen_calls={"ecommerce.order_status:{}"},
            catalog_tools_map=self.catalog,
            observed_entities={},
        )
        self.assertTrue(ok)

    def test_allows_ask_contact_id_after_contacts_search(self) -> None:
        """GT allows asking contact_id after contacts.search returns empty or ambiguous."""
        barrier = ActionTriageBarrier()
        ok, _ = barrier.guard_ask_missing_info(
            tool_call_args={"tool_name": "phone.call", "params": ["contact_id"]},
            sample=self.sample,
            seen_calls={'contacts.search:{"query": "Minh"}'},
            catalog_tools_map=self.catalog,
            observed_entities={},
        )
        self.assertTrue(ok)

    def test_allows_ask_booking_id_when_only_restaurant_search_exists(self) -> None:
        ext_catalog = dict(self.catalog)
        ext_catalog.update(
            {
                "restaurant_booking.search": {
                    "name": "restaurant_booking.search",
                    "kind": "read",
                    "returns": "restaurants[]",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
                "restaurant_booking.cancel": {
                    "name": "restaurant_booking.cancel",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"booking_id": {"type": "string", "x-ref": "restaurant_booking.bookings"}},
                        "required": ["booking_id"],
                    },
                },
                "public_services.list_services": {
                    "name": "public_services.list_services",
                    "kind": "read",
                    "returns": "services[]",
                    "description": "Danh sách thủ tục, các khung giờ còn trống (slots) và hồ sơ đã nộp (submitted_applications)",
                    "parameters": {"type": "object", "properties": {}},
                },
                "public_services.check_status": {
                    "name": "public_services.check_status",
                    "kind": "read",
                    "returns": "status",
                    "parameters": {
                        "type": "object",
                        "properties": {"application_id": {"type": "string", "x-ref": "public_services.applications"}},
                        "required": ["application_id"],
                    },
                },
            }
        )
        self.assertEqual(
            _resolve_xref_read_tools("restaurant_booking.bookings", "restaurant_booking.cancel", ext_catalog),
            [],
        )
        self.assertEqual(
            _resolve_xref_read_tools("public_services.applications", "public_services.check_status", ext_catalog),
            ["public_services.list_services"],
        )

        barrier = ActionTriageBarrier()
        ok_rb, _ = barrier.guard_ask_missing_info(
            tool_call_args={"tool_name": "restaurant_booking.cancel", "params": ["booking_id"]},
            sample=self.sample,
            seen_calls=set(),
            catalog_tools_map=ext_catalog,
            observed_entities={},
        )
        self.assertTrue(ok_rb)

        ok_ps, reason_ps = barrier.guard_ask_missing_info(
            tool_call_args={"tool_name": "public_services.check_status", "params": ["application_id"]},
            sample=self.sample,
            seen_calls=set(),
            catalog_tools_map=ext_catalog,
            observed_entities={},
        )
        self.assertFalse(ok_ps)
        self.assertIn("public_services.list_services", reason_ps)


if __name__ == "__main__":
    unittest.main()
