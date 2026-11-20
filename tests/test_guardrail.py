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

    def test_records_extended_entity_id_keys(self) -> None:
        extended_catalog = dict(self.catalog)
        extended_catalog.update(
            {
                "public_transit.search_routes": {
                    "name": "public_transit.search_routes",
                    "kind": "read",
                    "returns": "public_transit.routes[]",
                    "parameters": {"type": "object", "properties": {}},
                },
                "public_transit.get_arrival_time": {
                    "name": "public_transit.get_arrival_time",
                    "kind": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"route_id": {"type": "string", "x-ref": "public_transit.routes"}},
                        "required": ["route_id"],
                    },
                },
                "package_tracking.track": {
                    "name": "package_tracking.track",
                    "kind": "read",
                    "returns": "package_tracking.shipments[]",
                    "parameters": {"type": "object", "properties": {}},
                },
                "package_tracking.reschedule_delivery": {
                    "name": "package_tracking.reschedule_delivery",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"tracking_id": {"type": "string", "x-ref": "package_tracking.shipments"}},
                        "required": ["tracking_id"],
                    },
                },
            }
        )
        tracker = PureSchemaStateTracker(self.sample, "Kiểm tra tuyến xe và đơn hàng", extended_catalog)
        tracker.record_observation(
            "public_transit.search_routes",
            {},
            {"ok": True, "routes": [{"route_id": "rt_08", "name": "Tuyến 08"}]},
        )
        tracker.record_observation(
            "package_tracking.track",
            {},
            {"ok": True, "shipments": [{"id": "shp_1", "tracking_id": "VN123456"}]},
        )
        self.assertIn("rt_08", tracker.all_known_ids)
        self.assertIn("VN123456", tracker.all_known_ids)
        self.assertIsNone(tracker.check_pre_execution("public_transit.get_arrival_time", {"route_id": "rt_08"}))
        self.assertIsNone(tracker.check_pre_execution("package_tracking.reschedule_delivery", {"tracking_id": "VN123456"}))

    def test_public_services_nested_submitted_applications_unlocks_check_status(self) -> None:
        ps_catalog = dict(self.catalog)
        ps_catalog.update(
            {
                "public_services.list_services": {
                    "name": "public_services.list_services",
                    "kind": "read",
                    "returns": "services[]",
                    "parameters": {"type": "object", "properties": {}},
                },
                "public_services.check_status": {
                    "name": "public_services.check_status",
                    "kind": "read",
                    "returns": "status",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "application_id": {"type": "string", "x-ref": "public_services.applications"}
                        },
                        "required": ["application_id"],
                    },
                },
            }
        )
        tracker = PureSchemaStateTracker(self.sample, "Kiểm tra trạng thái hồ sơ", ps_catalog)
        tracker.record_observation(
            "public_services.list_services",
            {},
            {
                "ok": True,
                "services": [
                    {
                        "id": "svc_1",
                        "name": "Cấp lại căn cước",
                        "submitted_applications": [{"id": "hs_235", "submitted": "2026-11-11"}],
                    }
                ],
            },
        )
        self.assertIsNone(
            tracker.check_pre_execution("public_services.check_status", {"application_id": "hs_235"})
        )

    def test_singular_write_result_unlocks_cancel_and_blocks_foreign_key(self) -> None:
        rb_catalog = dict(self.catalog)
        rb_catalog.update(
            {
                "restaurant_booking.search": {
                    "name": "restaurant_booking.search",
                    "kind": "read",
                    "returns": "restaurants[]",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
                "restaurant_booking.book": {
                    "name": "restaurant_booking.book",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant_id": {"type": "string", "x-ref": "restaurant_booking.restaurants"},
                        },
                        "required": ["restaurant_id"],
                    },
                },
                "restaurant_booking.cancel": {
                    "name": "restaurant_booking.cancel",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "booking_id": {"type": "string", "x-ref": "restaurant_booking.bookings"},
                        },
                        "required": ["booking_id"],
                    },
                },
            }
        )
        history_sample = {
            "id": "rb_hist",
            "context": self.sample["context"],
            "history": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "restaurant_booking.book", "arguments": {"restaurant_id": "res_5821"}}
                    ],
                    "tool_results": [
                        {"ok": True, "booking": {"id": "rb_01", "restaurant_id": "res_5821", "status": "confirmed"}}
                    ],
                }
            ],
        }
        tracker = PureSchemaStateTracker(history_sample, "Hủy bàn vừa đặt", rb_catalog)
        self.assertIsNone(
            tracker.check_pre_execution("restaurant_booking.cancel", {"booking_id": "rb_01"})
        )
        self.assertIsNotNone(
            tracker.check_pre_execution("restaurant_booking.cancel", {"booking_id": "res_5821"})
        )

    def test_item_wrapper_resolves_to_tool_or_sole_app_collection(self) -> None:
        ext_catalog = dict(self.catalog)
        ext_catalog.update(
            {
                "alarm.create": {
                    "name": "alarm.create",
                    "kind": "write",
                    "parameters": {"type": "object", "properties": {"time": {"type": "string"}}},
                },
                "alarm.delete": {
                    "name": "alarm.delete",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}},
                        "required": ["alarm_id"],
                    },
                },
                "smart_home.light_set": {
                    "name": "smart_home.light_set",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"device_id": {"type": "string", "x-ref": "smart_home.devices"}},
                        "required": ["device_id"],
                    },
                },
                "smart_home.fan_set": {
                    "name": "smart_home.fan_set",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"device_id": {"type": "string", "x-ref": "smart_home.devices"}},
                        "required": ["device_id"],
                    },
                },
                "smart_home.switch_set": {
                    "name": "smart_home.switch_set",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"device_id": {"type": "string", "x-ref": "smart_home.devices"}},
                        "required": ["device_id"],
                    },
                },
                "smart_home.scene_run": {
                    "name": "smart_home.scene_run",
                    "kind": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"scene_id": {"type": "string", "x-ref": "smart_home.scenes"}},
                        "required": ["scene_id"],
                    },
                },
                "video_streaming.search": {
                    "name": "video_streaming.search",
                    "kind": "read",
                    "returns": "titles[]",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        )
        tracker = PureSchemaStateTracker(self.sample, "Xóa báo thức và chỉnh đèn", ext_catalog)
        tracker.record_observation(
            "alarm.create",
            {"time": "06:30"},
            {"ok": True, "item": {"id": "alm_01", "time": "06:30"}},
        )
        tracker.record_observation(
            "smart_home.light_set",
            {"device_id": "dev_light_1"},
            {"ok": True, "item": {"id": "dev_light_1", "type": "light"}},
        )
        tracker.record_observation(
            "smart_home.switch_set",
            {"device_id": "plug_heater"},
            {"ok": True, "item": {"id": "plug_heater", "type": "plug"}},
        )
        self.assertIsNone(tracker.check_pre_execution("alarm.delete", {"alarm_id": "alm_01"}))
        self.assertIsNone(tracker.check_pre_execution("smart_home.light_set", {"device_id": "dev_light_1"}))
        self.assertIsNone(tracker.check_pre_execution("smart_home.switch_set", {"device_id": "plug_heater"}))
        self.assertIsNotNone(tracker.check_pre_execution("smart_home.scene_run", {"scene_id": "dev_light_1"}))
        self.assertIsNotNone(tracker.check_pre_execution("smart_home.fan_set", {"device_id": "dev_light_1"}))
        self.assertIsNotNone(tracker.check_pre_execution("smart_home.fan_set", {"device_id": "plug_heater"}))

        norm_vid = normalize_arguments("video_streaming.search", {"query": "phim tài liệu"}, ext_catalog)
        self.assertEqual(norm_vid["query"], "tài liệu")


if __name__ == "__main__":
    unittest.main()
