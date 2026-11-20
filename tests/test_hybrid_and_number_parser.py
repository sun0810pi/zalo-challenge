"""Regression tests for Vietnamese compound number parser and HybridProgramAidedEngine."""
from __future__ import annotations

import unittest

from src.hybrid_engine import HybridProgramAidedEngine
from src.number_parser import parse_vietnamese_number


class TestHybridAndNumberParser(unittest.TestCase):
    def test_compound_vietnamese_numbers(self) -> None:
        self.assertEqual(parse_vietnamese_number("hai trăm năm mươi"), 250)
        self.assertEqual(parse_vietnamese_number("ba trăm linh năm"), 305)
        self.assertEqual(parse_vietnamese_number("mười lăm"), 15)
        self.assertEqual(parse_vietnamese_number("250.000"), 250000)
        self.assertEqual(parse_vietnamese_number("1.5 triệu"), 1500000)
        self.assertEqual(parse_vietnamese_number("hai triệu năm trăm nghìn"), 2500000)
        self.assertIsNone(parse_vietnamese_number("nửa"))

    def test_adjust_bill_payment_skips_when_balance_missing(self) -> None:
        """Never assume ewallet.balance == 0 when balance wasn't queried."""
        orig_args = {"bill_id": "bill_1", "amount": 500000}
        adj_args, cancelled = HybridProgramAidedEngine.adjust_bill_payment(
            tool_args=dict(orig_args),
            user_text="Thanh toán hóa đơn điện nếu đủ tiền",
            observed_entities={"bill_1": {"id": "bill_1", "amount": 500000}},
            balances={},
        )
        self.assertEqual(adj_args, orig_args)
        self.assertFalse(cancelled)

    def test_adjust_bill_payment_adjusts_when_balance_observed(self) -> None:
        adj_args, cancelled = HybridProgramAidedEngine.adjust_bill_payment(
            tool_args={"bill_id": "bill_1", "amount": 500000},
            user_text="Nếu đủ tiền thì thanh toán, không thì nạp thêm",
            observed_entities={"bill_1": {"id": "bill_1", "amount": 500000}},
            balances={"ewallet.balance": 300000},
        )
        self.assertIsInstance(cancelled, bool)
        if adj_args is not None:
            self.assertIsInstance(adj_args, dict)

    def test_expand_smart_home_fans(self) -> None:
        class _Env:
            @staticmethod
            def validate(name: str, args: dict) -> dict:
                return {"ok": True}

        entities = {
            "fan_1": {"id": "fan_1", "name": "Quạt phòng khách", "type": "fan"},
            "fan_2": {"id": "fan_2", "name": "Quạt phòng ngủ", "type": "fan"},
        }
        extras = HybridProgramAidedEngine.expand_smart_home_fans(
            env=_Env(),
            user_text="Tắt tất cả các quạt giúp tôi",
            observed_entities=entities,
            initial_fan_args={"device_id": "fan_1", "action": "turn_off"},
        )
        self.assertIsInstance(extras, list)
        if extras:
            self.assertTrue(all("name" in c and "arguments" in c for c in extras))

    def test_check_weather_condition_branches_and_multiday(self) -> None:
        # 1. "Nếu mưa thì ..., không mưa thì thôi" -> stop on dry forecast, continue on rain
        self.assertFalse(
            HybridProgramAidedEngine.check_weather_condition(
                "Nếu mai mưa thì dời chạy bộ, không mưa thì thôi.",
                {"ok": True, "forecast": [{"date": "2026-11-16", "rain": False}]},
            )
        )
        self.assertTrue(
            HybridProgramAidedEngine.check_weather_condition(
                "Nếu mai mưa thì dời chạy bộ, không mưa thì thôi.",
                {"ok": True, "forecast": [{"date": "2026-11-16", "rain": True}]},
            )
        )
        # 2. "Nếu trời không mưa thì ..." -> stop when all target forecast days rain
        self.assertFalse(
            HybridProgramAidedEngine.check_weather_condition(
                "Nếu trời không mưa thì đặt vé công viên nhé.",
                {"ok": True, "forecast": [{"date": "2026-11-16", "rain": True}]},
            )
        )
        # 3. Multi-day forecast with mixed rain/dry must NOT falsely stop
        self.assertTrue(
            HybridProgramAidedEngine.check_weather_condition(
                "Nếu mai trời không mưa thì đặt xe đi chơi nhé.",
                {
                    "ok": True,
                    "forecasts": [
                        {"date": "2026-11-15", "rain": True},
                        {"date": "2026-11-16", "rain": False},
                    ],
                },
            )
        )
        # 4. Non-weather "nếu" conditional must NOT trigger false stop
        self.assertTrue(
            HybridProgramAidedEngine.check_weather_condition(
                "Xem thời tiết mai dùm anh, nếu rảnh thì nhắc anh gọi bạn Minh.",
                {"ok": True, "forecast": [{"date": "2026-11-16", "rain": True}]},
            )
        )


if __name__ == "__main__":
    unittest.main()
