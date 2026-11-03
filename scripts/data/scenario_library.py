"""Scenario library for Simulator-Grounded SFT Data Generation (ZAC 2026 LifeMate).

Contains:
- Phase 2 Pilot Batch: 24 high-value scenarios covering the 11 unseen apps + critical reasoning archetypes
  (read-before-write, multi-result selection, multi-step dependencies, arithmetic/conditional checks,
   missing info clarification, negation/self-cancellation).
- Phase 3 Full Expansion: 140 diverse, non-duplicated scenarios with strict template & seed separation
  between train (~80%) and val (~20%) splits.

All scenarios execute against the REAL `zaic_tools.executor.Session` simulator and extract IDs dynamically
from actual simulator responses.
"""
from __future__ import annotations

import random
from typing import Any

from scripts.data.simulator_scenario_engine import ScenarioSpec, SimulatorRunner

USER_PROFILES = [
    {
        "name": "Nguyễn Minh Quân",
        "city": "Hà Nội",
        "home_address": "Số 45 Nguyễn Chí Thanh, Đống Đa, Hà Nội",
        "work_address": "Tòa nhà Keangnam, Phạm Hùng, Nam Từ Liêm, Hà Nội",
        "lat": 21.0245,
        "lng": 105.8091,
    },
    {
        "name": "Trần Thu Hà",
        "city": "TP.HCM",
        "home_address": "128 Nguyễn Thị Minh Khai, Quận 3, TP.HCM",
        "work_address": "Tòa nhà Bitexco, số 2 Hải Triều, Quận 1, TP.HCM",
        "lat": 10.7769,
        "lng": 106.7009,
    },
    {
        "name": "Lê Hoàng Nam",
        "city": "Đà Nẵng",
        "home_address": "72 Nguyễn Văn Linh, Hải Châu, Đà Nẵng",
        "work_address": "Khu Công nghệ thông tin tập trung, Liên Chiểu, Đà Nẵng",
        "lat": 16.0601,
        "lng": 108.2122,
    },
    {
        "name": "Phạm Thanh Thảo",
        "city": "Hà Nội",
        "home_address": "Số 18 Hoàng Đạo Thúy, Cầu Giấy, Hà Nội",
        "work_address": "Tòa nhà Lotte Center, 54 Liễu Giai, Ba Đình, Hà Nội",
        "lat": 21.0069,
        "lng": 105.8027,
    },
    {
        "name": "Võ Quốc Bảo",
        "city": "TP.HCM",
        "home_address": "215 Điện Biên Phủ, Bình Thạnh, TP.HCM",
        "work_address": "Khu đô thị Sala, Thủ Đức, TP.HCM",
        "lat": 10.8015,
        "lng": 106.7114,
    },
]


def _make_profile(idx: int) -> tuple[str, dict[str, str], dict[str, float]]:
    p = USER_PROFILES[idx % len(USER_PROFILES)]
    return (
        p["city"],
        {
            "name": p["name"],
            "city": p["city"],
            "home_address": p["home_address"],
            "work_address": p["work_address"],
        },
        {"lat": p["lat"], "lng": p["lng"]},
    )


def build_pilot_24_scenarios() -> list[ScenarioSpec]:
    """Build the Phase 2 pilot batch of 24 high-value scenarios covering all critical audit issues."""
    scenarios: list[ScenarioSpec] = []

    # 1. parking.find -> select lot with available_spots > 0 -> parking.reserve
    city, prof, loc = _make_profile(0)
    def plan_p01(r: SimulatorRunner):
        res = r.step("parking.find", {"near": "Bà Triệu"})
        chosen = next(l for l in res["lots"] if l.get("available_spots", 0) > 0)
        r.step("parking.reserve", {"lot_id": chosen["id"], "from_time": "2026-11-20T14:00:00+07:00", "hours": 3})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_001_parking_reserve_select",
        template_id="tpl_parking_reserve_select",
        split="train",
        scenario_type="multi_result_selection",
        primary_apps=["parking"],
        installed_apps=["parking", "maps", "ewallet", "calendar"],
        user_request="Tìm bãi đỗ xe gần Bà Triệu xem bãi nào còn chỗ trống thì đặt giữ chỗ từ 14 giờ chiều nay trong 3 tiếng giúp anh.",
        current_time="2026-11-20T10:15:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "parking": {
                "lots": [
                    {"id": "lot_bt_01", "name": "Bãi đỗ xe Bà Triệu A", "address": "110 Bà Triệu", "area": "Hai Bà Trưng", "price_per_hour": 25000, "available_spots": 0},
                    {"id": "lot_bt_02", "name": "Bãi đỗ xe Vincom Bà Triệu", "address": "191 Bà Triệu", "area": "Hai Bà Trưng", "price_per_hour": 30000, "available_spots": 9},
                ],
                "sessions": [],
            }
        },
        objective="Select the parking lot near Bà Triệu with available_spots > 0 and reserve for 3 hours.",
        expect_mutation=True,
        plan_fn=plan_p01,
        verify_fn=lambda final_state, **_: (
            len(final_state["parking"]["sessions"]) == 1
            and final_state["parking"]["sessions"][0]["lot_id"] == "lot_bt_02"
            and final_state["parking"]["sessions"][0]["status"] == "reserved",
            "Expected reserved session on lot_bt_02",
        ),
    ))

    # 2. parking.list_sessions -> select active/reserved session -> parking.end_session
    city, prof, loc = _make_profile(1)
    def plan_p02(r: SimulatorRunner):
        res = r.step("parking.list_sessions", {})
        active = next(s for s in res["sessions"] if s.get("status") != "ended")
        r.step("parking.end_session", {"session_id": active["id"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_002_parking_end_active",
        template_id="tpl_parking_end_active",
        split="train",
        scenario_type="multi_result_selection",
        primary_apps=["parking"],
        installed_apps=["parking", "ewallet", "ride_hailing"],
        user_request="Kiểm tra các phiên đỗ xe của chị rồi kết thúc phiên đang đỗ ở bãi xe Nguyễn Huệ nhé.",
        current_time="2026-11-20T16:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "parking": {
                "lots": [
                    {"id": "lot_nh_01", "name": "Bãi xe Lê Lợi", "address": "45 Lê Lợi", "area": "Quận 1", "price_per_hour": 30000, "available_spots": 5},
                    {"id": "lot_nh_02", "name": "Bãi xe Nguyễn Huệ", "address": "88 Nguyễn Huệ", "area": "Quận 1", "price_per_hour": 35000, "available_spots": 2},
                ],
                "sessions": [
                    {"id": "park_11", "lot_id": "lot_nh_01", "from_time": "2026-11-19T09:00:00+07:00", "hours": 2, "end_time": "2026-11-19T11:00:00+07:00", "total": 60000, "status": "ended"},
                    {"id": "park_12", "lot_id": "lot_nh_02", "from_time": "2026-11-20T14:00:00+07:00", "hours": 3, "end_time": "2026-11-20T17:00:00+07:00", "total": 105000, "status": "reserved"},
                ],
            }
        },
        objective="Find the non-ended parking session at Nguyễn Huệ and end it.",
        expect_mutation=True,
        plan_fn=plan_p02,
        verify_fn=lambda final_state, **_: (
            final_state["parking"]["sessions"][1]["status"] == "ended"
            and final_state["parking"]["lots"][1]["available_spots"] == 3,
            "Expected session park_12 to be ended and spots incremented",
        ),
    ))

    # 3. stocks.get_quote -> conditional buy satisfied -> stocks.place_order
    city, prof, loc = _make_profile(0)
    def plan_p03(r: SimulatorRunner):
        q_res = r.step("stocks.get_quote", {"symbol": "FPT"})
        price = q_res["quote"]["price"]
        if price < 135000:
            r.step("stocks.place_order", {"symbol": "FPT", "side": "buy", "quantity": 200, "price": price})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_003_stocks_cond_buy_met",
        template_id="tpl_stocks_cond_buy_met",
        split="train",
        scenario_type="arithmetic_condition",
        primary_apps=["stocks"],
        installed_apps=["stocks", "banking", "news", "reminders"],
        user_request="Xem giá cổ phiếu FPT hiện tại giúp anh, nếu giá đang dưới 135 nghìn thì đặt mua 200 cổ phiếu theo giá hiện tại luôn.",
        current_time="2026-11-20T10:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "stocks": {
                "quotes": [
                    {"symbol": "FPT", "name": "CTCP FPT", "price": 131500, "change_pct": -0.8},
                    {"symbol": "HPG", "name": "Tập đoàn Hòa Phát", "price": 28400, "change_pct": 1.1},
                ],
                "holdings": [],
                "orders": [],
                "alerts": [],
            }
        },
        objective="Check FPT quote (131500 < 135000) and place buy order for 200 shares.",
        expect_mutation=True,
        plan_fn=plan_p03,
        verify_fn=lambda final_state, **_: (
            len(final_state["stocks"]["orders"]) == 1
            and final_state["stocks"]["orders"][0]["symbol"] == "FPT"
            and final_state["stocks"]["orders"][0]["quantity"] == 200,
            "Expected buy order for 200 FPT",
        ),
    ))

    # 4. stocks.get_portfolio -> sell half ("bán một nửa") -> stocks.place_order
    city, prof, loc = _make_profile(2)
    def plan_p04(r: SimulatorRunner):
        p_res = r.step("stocks.get_portfolio", {})
        holding = next(h for h in p_res["holdings"] if h["symbol"] == "HPG")
        sell_qty = holding["quantity"] // 2
        r.step("stocks.place_order", {"symbol": "HPG", "side": "sell", "quantity": sell_qty, "price": holding["price"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_004_stocks_sell_half",
        template_id="tpl_stocks_sell_half",
        split="train",
        scenario_type="arithmetic_condition",
        primary_apps=["stocks"],
        installed_apps=["stocks", "banking", "ewallet"],
        user_request="Kiểm tra danh mục đầu tư xem mình đang giữ bao nhiêu cổ phiếu HPG rồi đặt lệnh bán một nửa số đó theo giá thị trường hiện tại nhé.",
        current_time="2026-11-20T13:45:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "stocks": {
                "holdings": [
                    {"symbol": "HPG", "name": "Tập đoàn Hòa Phát", "quantity": 800, "avg_price": 26500, "price": 28900},
                    {"symbol": "VNM", "name": "Vinamilk", "quantity": 300, "avg_price": 66000, "price": 67500},
                ],
                "quotes": [],
                "orders": [],
                "alerts": [],
            }
        },
        objective="Inspect portfolio, compute half of 800 HPG shares (400), and place sell order.",
        expect_mutation=True,
        plan_fn=plan_p04,
        verify_fn=lambda final_state, **_: (
            len(final_state["stocks"]["orders"]) == 1
            and final_state["stocks"]["orders"][0]["side"] == "sell"
            and final_state["stocks"]["orders"][0]["quantity"] == 400,
            "Expected sell order for 400 HPG",
        ),
    ))

    # 5. stocks.get_portfolio -> sell keeping X ("giữ lại 200 cổ") -> stocks.place_order
    city, prof, loc = _make_profile(3)
    def plan_p05(r: SimulatorRunner):
        p_res = r.step("stocks.get_portfolio", {})
        holding = next(h for h in p_res["holdings"] if h["symbol"] == "ZMX")
        sell_qty = holding["quantity"] - 200
        r.step("stocks.place_order", {"symbol": "ZMX", "side": "sell", "quantity": sell_qty, "price": holding["price"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_005_stocks_sell_keep",
        template_id="tpl_stocks_sell_keep",
        split="train",
        scenario_type="arithmetic_condition",
        primary_apps=["stocks"],
        installed_apps=["stocks", "banking"],
        user_request="Bán bớt cổ phiếu ZMX trong danh mục của chị đi, chỉ giữ lại 200 cổ phiếu thôi, phần còn lại bán hết theo giá hiện tại.",
        current_time="2026-11-20T14:10:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "stocks": {
                "holdings": [
                    {"symbol": "ZMX", "name": "CTCP Sữa Mộc Xanh", "quantity": 700, "avg_price": 55000, "price": 61000},
                ],
                "quotes": [],
                "orders": [],
                "alerts": [],
            }
        },
        objective="Check portfolio (700 ZMX), keep 200 and sell 500 shares.",
        expect_mutation=True,
        plan_fn=plan_p05,
        verify_fn=lambda final_state, **_: (
            len(final_state["stocks"]["orders"]) == 1
            and final_state["stocks"]["orders"][0]["quantity"] == 500,
            "Expected sell order for 500 ZMX",
        ),
    ))

    # 6. train_bus_tickets.search -> select trip with enough seats_left -> book
    city, prof, loc = _make_profile(0)
    def plan_p06(r: SimulatorRunner):
        s_res = r.step("train_bus_tickets.search", {"origin": "Hà Nội", "destination": "Hải Phòng", "date": "2026-11-22", "type": "train"})
        chosen = next(t for t in s_res["trips"] if t.get("seats_left", 0) >= 3)
        r.step("train_bus_tickets.book", {"trip_id": chosen["id"], "seats": 3})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_006_train_bus_book_seats",
        template_id="tpl_train_bus_book_seats",
        split="train",
        scenario_type="multi_result_selection",
        primary_apps=["train_bus_tickets"],
        installed_apps=["train_bus_tickets", "flights", "calendar", "ewallet"],
        user_request="Tìm vé tàu hỏa từ Hà Nội đi Hải Phòng ngày 2026-11-22, chọn chuyến còn đủ chỗ cho 3 người rồi đặt 3 vé giúp mình.",
        current_time="2026-11-20T09:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "train_bus_tickets": {
                "trips": [
                    {"id": "trip_hp_01", "type": "train", "operator": "Đường sắt VN HP1", "origin": "Hà Nội", "destination": "Hải Phòng", "date": "2026-11-22", "depart": "06:00", "arrive": "08:30", "price": 120000, "seats_left": 2},
                    {"id": "trip_hp_02", "type": "train", "operator": "Đường sắt VN LP3", "origin": "Hà Nội", "destination": "Hải Phòng", "date": "2026-11-22", "depart": "09:20", "arrive": "12:00", "price": 120000, "seats_left": 14},
                ],
                "bookings": [],
            }
        },
        objective="Search train trips to Hải Phòng, skip trip with only 2 seats_left, book 3 seats on trip_hp_02.",
        expect_mutation=True,
        plan_fn=plan_p06,
        verify_fn=lambda final_state, **_: (
            len(final_state["train_bus_tickets"]["bookings"]) == 1
            and final_state["train_bus_tickets"]["bookings"][0]["trip_id"] == "trip_hp_02"
            and final_state["train_bus_tickets"]["bookings"][0]["seats"] == 3,
            "Expected 3 seats booked on trip_hp_02",
        ),
    ))

    # 7. train_bus_tickets.my_bookings -> cancel active booking
    city, prof, loc = _make_profile(1)
    def plan_p07(r: SimulatorRunner):
        b_res = r.step("train_bus_tickets.my_bookings", {})
        target = next(b for b in b_res["bookings"] if b.get("status") == "confirmed")
        r.step("train_bus_tickets.cancel", {"booking_id": target["id"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_007_train_bus_cancel",
        template_id="tpl_train_bus_cancel",
        split="train",
        scenario_type="read_before_write",
        primary_apps=["train_bus_tickets"],
        installed_apps=["train_bus_tickets", "calendar"],
        user_request="Kiểm tra vé xe khách đã đặt của tôi rồi hủy vé đang ở trạng thái xác nhận nhé.",
        current_time="2026-11-20T11:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "train_bus_tickets": {
                "trips": [
                    {"id": "trip_dl_01", "type": "bus", "operator": "Phương Trang", "origin": "TP.HCM", "destination": "Đà Lạt", "date": "2026-11-24", "depart": "22:00", "arrive": "05:00", "price": 310000, "seats_left": 8},
                ],
                "bookings": [
                    {"id": "tk_dl_01", "trip_id": "trip_dl_01", "seats": 2, "total": 620000, "status": "confirmed"},
                ],
            }
        },
        objective="Read bookings to obtain booking_id tk_dl_01 and cancel it.",
        expect_mutation=True,
        plan_fn=plan_p07,
        verify_fn=lambda final_state, **_: (
            final_state["train_bus_tickets"]["bookings"][0]["status"] == "cancelled"
            and final_state["train_bus_tickets"]["trips"][0]["seats_left"] == 10,
            "Expected booking tk_dl_01 cancelled and seats restored",
        ),
    ))

    # 8. ecommerce.search -> select in_stock product -> add_to_cart -> checkout
    city, prof, loc = _make_profile(0)
    def plan_p08(r: SimulatorRunner):
        s_res = r.step("ecommerce.search", {"query": "tai nghe Bluetooth", "max_price": 600000})
        prod = next(p for p in s_res["products"] if p.get("in_stock") is True)
        r.step("ecommerce.add_to_cart", {"product_id": prod["id"], "quantity": 1})
        r.step("ecommerce.checkout", {"address": prof["work_address"], "payment": "ewallet"})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_008_ecommerce_buy_instock",
        template_id="tpl_ecommerce_buy_instock",
        split="train",
        scenario_type="multi_step_chain",
        primary_apps=["ecommerce"],
        installed_apps=["ecommerce", "ewallet", "banking", "grocery"],
        user_request="Tìm mua 1 chiếc tai nghe Bluetooth giá tối đa 600 nghìn trên sàn thương mại điện tử, chọn mẫu đang còn hàng thêm vào giỏ rồi đặt giao về địa chỉ công ty, thanh toán bằng ví điện tử nhé.",
        current_time="2026-11-20T10:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "ecommerce": {
                "products": [
                    {"id": "prod_bt_01", "name": "Tai nghe Bluetooth Sony WF", "price": 550000, "seller": "Sony Store", "category": "âm thanh", "rating": 4.8, "in_stock": False},
                    {"id": "prod_bt_02", "name": "Tai nghe Bluetooth Baseus Bowie", "price": 460000, "seller": "Baseus Official", "category": "âm thanh", "rating": 4.7, "in_stock": True},
                ],
                "cart": [],
                "orders": [],
                "returns": [],
            }
        },
        objective="Search Bluetooth headphones <= 600k, pick in_stock prod_bt_02, add to cart, and checkout to work_address.",
        expect_mutation=True,
        plan_fn=plan_p08,
        verify_fn=lambda final_state, **_: (
            len(final_state["ecommerce"]["orders"]) == 1
            and final_state["ecommerce"]["orders"][0]["items"][0]["product_id"] == "prod_bt_02"
            and final_state["ecommerce"]["orders"][0]["payment"] == "ewallet",
            "Expected placed order for prod_bt_02 with ewallet",
        ),
    ))

    # 9. ecommerce.order_status -> select delivered order -> request_return
    city, prof, loc = _make_profile(1)
    def plan_p09(r: SimulatorRunner):
        o_res = r.step("ecommerce.order_status", {})
        delivered = next(o for o in o_res["orders"] if o.get("status") == "delivered")
        r.step("ecommerce.request_return", {"order_id": delivered["id"], "reason": "damaged"})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_009_ecommerce_return_delivered",
        template_id="tpl_ecommerce_return_delivered",
        split="train",
        scenario_type="multi_result_selection",
        primary_apps=["ecommerce"],
        installed_apps=["ecommerce", "sms"],
        user_request="Kiểm tra các đơn hàng trên sàn mua sắm, đơn bàn phím cơ nào đã giao mà bị hỏng thì tạo yêu cầu trả hàng giúp chị.",
        current_time="2026-11-20T15:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "ecommerce": {
                "products": [],
                "cart": [],
                "orders": [
                    {"id": "ord_ec_01", "items": [{"product_id": "prod_kb_01", "name": "Bàn phím cơ Akko 3068", "price": 890000, "quantity": 1}], "total": 890000, "status": "delivered"},
                    {"id": "ord_ec_02", "items": [{"product_id": "prod_ms_01", "name": "Chuột không dây Logitech", "price": 320000, "quantity": 1}], "total": 320000, "status": "shipping"},
                ],
                "returns": [],
            }
        },
        objective="Check ecommerce orders, identify delivered order ord_ec_01, and request return with reason='damaged'.",
        expect_mutation=True,
        plan_fn=plan_p09,
        verify_fn=lambda final_state, **_: (
            len(final_state["ecommerce"]["returns"]) == 1
            and final_state["ecommerce"]["returns"][0]["order_id"] == "ord_ec_01"
            and final_state["ecommerce"]["orders"][0]["status"] == "return_requested",
            "Expected return requested on ord_ec_01",
        ),
    ))

    # 10. ecommerce.order_status -> select shipping/placed order -> cancel_order
    city, prof, loc = _make_profile(3)
    def plan_p10(r: SimulatorRunner):
        o_res = r.step("ecommerce.order_status", {})
        active = next(o for o in o_res["orders"] if o.get("status") in ("placed", "shipping"))
        r.step("ecommerce.cancel_order", {"order_id": active["id"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_010_ecommerce_cancel_shipping",
        template_id="tpl_ecommerce_cancel_shipping",
        split="val",
        scenario_type="read_before_write",
        primary_apps=["ecommerce"],
        installed_apps=["ecommerce", "ewallet"],
        user_request="Xem đơn hàng đang vận chuyển trên sàn thương mại điện tử rồi hủy đơn đó giúp mình nhé.",
        current_time="2026-11-20T11:20:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "ecommerce": {
                "products": [],
                "cart": [],
                "orders": [
                    {"id": "ord_ec_09", "items": [{"product_id": "prod_09", "name": "Đèn bàn học chống cận", "price": 280000, "quantity": 1}], "total": 280000, "status": "shipping"},
                ],
                "returns": [],
            }
        },
        objective="Check order status and cancel shipping order ord_ec_09.",
        expect_mutation=True,
        plan_fn=plan_p10,
        verify_fn=lambda final_state, **_: (
            final_state["ecommerce"]["orders"][0]["status"] == "cancelled",
            "Expected ord_ec_09 cancelled",
        ),
    ))

    # 11. grocery.search -> add_to_cart -> checkout
    city, prof, loc = _make_profile(0)
    def plan_p11(r: SimulatorRunner):
        g_res = r.step("grocery.search", {"query": "sữa tươi"})
        prod = g_res["products"][0]
        r.step("grocery.add_to_cart", {"product_id": prod["id"], "quantity": 2})
        r.step("grocery.checkout", {"address": prof["home_address"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_011_grocery_buy_checkout",
        template_id="tpl_grocery_buy_checkout",
        split="train",
        scenario_type="multi_step_chain",
        primary_apps=["grocery"],
        installed_apps=["grocery", "pantry", "shopping_list", "ewallet"],
        user_request="Tìm mua 2 hộp sữa tươi trên ứng dụng đi chợ rồi thanh toán đặt giao về địa chỉ nhà giúp anh.",
        current_time="2026-11-20T08:45:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "grocery": {
                "products": [
                    {"id": "gro_milk_01", "name": "Sữa tươi tiệt trùng Vinamilk 1L", "price": 36000, "unit": "hộp", "category": "sữa", "in_stock": True},
                ],
                "cart": [],
                "orders": [],
            }
        },
        objective="Search milk on grocery app, add 2 to cart, and checkout to home_address.",
        expect_mutation=True,
        plan_fn=plan_p11,
        verify_fn=lambda final_state, **_: (
            len(final_state["grocery"]["orders"]) == 1
            and final_state["grocery"]["orders"][0]["total"] == 72000,
            "Expected grocery order with total 72000",
        ),
    ))

    # 12. grocery.search -> multiple ambiguous matches -> sys.ask_missing_info
    city, prof, loc = _make_profile(1)
    def plan_p12(r: SimulatorRunner):
        r.step("grocery.search", {"query": "trứng"})
        r.step("sys.ask_missing_info", {"tool_name": "grocery.add_to_cart", "params": ["product_id"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_012_grocery_ambiguous_ask",
        template_id="tpl_grocery_ambiguous_ask",
        split="train",
        scenario_type="missing_info",
        primary_apps=["grocery", "sys"],
        installed_apps=["grocery", "notes"],
        user_request="Đặt mua giúp chị 1 hộp trứng trên app đi chợ nhé.",
        current_time="2026-11-20T09:15:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "grocery": {
                "products": [
                    {"id": "gro_egg_01", "name": "Trứng gà ta Ba Huân hộp 10 quả", "price": 38000, "unit": "hộp", "category": "trứng", "in_stock": True},
                    {"id": "gro_egg_02", "name": "Trứng vịt sạch CP hộp 10 quả", "price": 34000, "unit": "hộp", "category": "trứng", "in_stock": True},
                ],
                "cart": [],
                "orders": [],
            }
        },
        objective="Search eggs on grocery, see 2 distinct egg types (gà vs vịt), and call sys.ask_missing_info.",
        expect_mutation=False,
        plan_fn=plan_p12,
        verify_fn=lambda tool_calls, **_: (
            tool_calls[-1]["name"] == "sys.ask_missing_info"
            and tool_calls[-1]["arguments"]["tool_name"] == "grocery.add_to_cart",
            "Expected sys.ask_missing_info for grocery.add_to_cart",
        ),
    ))

    # 13. Cross-app: recipes.search -> pantry.list_items -> grocery.search -> grocery.add_to_cart -> grocery.checkout
    city, prof, loc = _make_profile(0)
    def plan_p13(r: SimulatorRunner):
        r.step("recipes.search", {"query": "canh chua cá lóc", "servings": 4})
        p_res = r.step("pantry.list_items", {})
        missing = next(it for it in p_res["items"] if it.get("stock", 0) == 0)
        g_res = r.step("grocery.search", {"query": missing["name"]})
        g_prod = g_res["products"][0]
        r.step("grocery.add_to_cart", {"product_id": g_prod["id"], "quantity": 1})
        r.step("grocery.checkout", {"address": prof["home_address"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_013_chain_recipes_pantry_grocery",
        template_id="tpl_chain_recipes_pantry_grocery",
        split="train",
        scenario_type="multi_step_chain",
        primary_apps=["recipes", "pantry", "grocery"],
        installed_apps=["recipes", "pantry", "grocery", "notes"],
        user_request="Tìm công thức nấu canh chua cá lóc cho 4 người, kiểm tra tủ bếp xem nguyên liệu nào đã hết (stock=0) thì tìm mua 1 phần trên app đi chợ và đặt giao về nhà nhé.",
        current_time="2026-11-20T10:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "recipes": {
                "recipes": [
                    {
                        "id": "rcp_cc_01",
                        "name": "Canh chua cá lóc miền Nam",
                        "servings": 4,
                        "ingredients": [
                            {"name": "cá lóc", "quantity": 1, "unit": "con"},
                            {"name": "cà chua", "quantity": 3, "unit": "quả"},
                        ],
                    }
                ],
                "saved": [],
            },
            "pantry": {
                "items": [
                    {"id": "pan_01", "name": "cà chua", "stock": 5, "unit": "quả"},
                    {"id": "pan_02", "name": "cá lóc", "stock": 0, "unit": "con"},
                ]
            },
            "grocery": {
                "products": [
                    {"id": "gro_fish_01", "name": "Cá lóc đồng làm sẵn 700g", "price": 92000, "unit": "con", "category": "thủy sản", "in_stock": True},
                ],
                "cart": [],
                "orders": [],
            },
        },
        objective="Chain recipes.search -> pantry.list_items -> grocery.search -> grocery.add_to_cart (using gro_fish_01, never rcp_ or pan_) -> grocery.checkout.",
        expect_mutation=True,
        plan_fn=plan_p13,
        verify_fn=lambda final_state, **_: (
            len(final_state["grocery"]["orders"]) == 1
            and final_state["grocery"]["orders"][0]["items"][0]["product_id"] == "gro_fish_01",
            "Expected grocery order for gro_fish_01",
        ),
    ))

    # 14. pantry.list_items -> pantry.consume
    city, prof, loc = _make_profile(2)
    def plan_p14(r: SimulatorRunner):
        p_res = r.step("pantry.list_items", {})
        target = next(it for it in p_res["items"] if "sữa tươi" in it["name"].lower())
        r.step("pantry.consume", {"item_id": target["id"], "quantity": 2})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_014_pantry_consume",
        template_id="tpl_pantry_consume",
        split="train",
        scenario_type="read_before_write",
        primary_apps=["pantry"],
        installed_apps=["pantry", "grocery", "shopping_list"],
        user_request="Nhà mình vừa uống hết 2 hộp sữa tươi trong tủ bếp, trừ giúp mình số lượng tồn kho trong ứng dụng tủ bếp nhé.",
        current_time="2026-11-20T08:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "pantry": {
                "items": [
                    {"id": "pan_milk_01", "name": "sữa tươi", "stock": 5, "unit": "hộp"},
                    {"id": "pan_oil_02", "name": "dầu ăn", "stock": 2, "unit": "chai"},
                ]
            }
        },
        objective="List pantry items, find pan_milk_01, and consume 2.",
        expect_mutation=True,
        plan_fn=plan_p14,
        verify_fn=lambda final_state, **_: (
            final_state["pantry"]["items"][0]["stock"] == 3,
            "Expected stock of pan_milk_01 reduced from 5 to 3",
        ),
    ))

    # 15. recipes.search -> recipes.save
    city, prof, loc = _make_profile(3)
    def plan_p15(r: SimulatorRunner):
        s_res = r.step("recipes.search", {"query": "sườn xào chua ngọt", "servings": 4})
        rcp = s_res["recipes"][0]
        r.step("recipes.save", {"recipe_id": rcp["id"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_015_recipes_search_save",
        template_id="tpl_recipes_search_save",
        split="train",
        scenario_type="read_before_write",
        primary_apps=["recipes"],
        installed_apps=["recipes", "pantry", "notes"],
        user_request="Tìm công thức làm món sườn xào chua ngọt khẩu phần 4 người rồi lưu công thức đó lại giúp chị.",
        current_time="2026-11-20T17:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "recipes": {
                "recipes": [
                    {"id": "rcp_sx_01", "name": "Sườn xào chua ngọt chuẩn vị", "servings": 2, "ingredients": [{"name": "sườn non", "quantity": 300, "unit": "g"}]},
                ],
                "saved": [],
            }
        },
        objective="Search recipe scaled to 4 servings and save recipe_id rcp_sx_01.",
        expect_mutation=True,
        plan_fn=plan_p15,
        verify_fn=lambda final_state, **_: (
            len(final_state["recipes"]["saved"]) == 1
            and final_state["recipes"]["saved"][0]["recipe_id"] == "rcp_sx_01",
            "Expected rcp_sx_01 saved",
        ),
    ))

    # 16. event_tickets.search -> event_tickets.book
    city, prof, loc = _make_profile(1)
    def plan_p16(r: SimulatorRunner):
        e_res = r.step("event_tickets.search", {"query": "Sài Gòn Cười"})
        ev = e_res["events"][0]
        r.step("event_tickets.book", {"event_id": ev["id"], "quantity": 3})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_016_event_tickets_book",
        template_id="tpl_event_tickets_book",
        split="train",
        scenario_type="read_before_write",
        primary_apps=["event_tickets"],
        installed_apps=["event_tickets", "calendar", "ewallet"],
        user_request="Tìm vé đêm hài kịch Sài Gòn Cười rồi đặt cho gia đình mình 3 vé nhé.",
        current_time="2026-11-20T12:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "event_tickets": {
                "events": [
                    {"id": "event_sgc_01", "name": "Đêm hài kịch Sài Gòn Cười", "category": "hài kịch", "date": "2026-11-28", "time": "20:00", "venue": "Nhà hát Bến Thành", "price": 350000, "available": 25},
                ],
                "bookings": [],
            }
        },
        objective="Search event Sài Gòn Cười and book 3 tickets using event_id event_sgc_01.",
        expect_mutation=True,
        plan_fn=plan_p16,
        verify_fn=lambda final_state, **_: (
            len(final_state["event_tickets"]["bookings"]) == 1
            and final_state["event_tickets"]["bookings"][0]["quantity"] == 3
            and final_state["event_tickets"]["events"][0]["available"] == 22,
            "Expected 3 event tickets booked",
        ),
    ))

    # 17. public_transit.find_routes -> public_transit.topup_card
    city, prof, loc = _make_profile(0)
    def plan_p17(r: SimulatorRunner):
        r.step("public_transit.find_routes", {"origin": "Nhổn", "destination": "Ga Hà Nội"})
        r.step("public_transit.topup_card", {"amount": 100000})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_017_public_transit_topup",
        template_id="tpl_public_transit_topup",
        split="train",
        scenario_type="multi_step_chain",
        primary_apps=["public_transit"],
        installed_apps=["public_transit", "maps", "ewallet"],
        user_request="Tra cứu tuyến tàu điện từ Nhổn đi Ga Hà Nội rồi nạp thêm 100 nghìn đồng vào thẻ vé giao thông công cộng giúp tôi.",
        current_time="2026-11-20T07:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "public_transit": {
                "card_balance": 25000,
                "routes": [
                    {"id": "metro_03", "name": "Tuyến Metro số 3 Nhổn - Ga Hà Nội", "type": "metro", "stops": ["Nhổn", "Cầu Giấy", "Kim Mã", "Cát Linh", "Ga Hà Nội"], "fare": 12000, "duration_min": 24, "headway_min": 10}
                ],
                "departures": [],
                "topups": [],
            }
        },
        objective="Find transit route from Nhổn to Ga Hà Nội and top up transit card by 100,000 VND.",
        expect_mutation=True,
        plan_fn=plan_p17,
        verify_fn=lambda final_state, **_: (
            final_state["public_transit"]["card_balance"] == 125000
            and len(final_state["public_transit"]["topups"]) == 1,
            "Expected card_balance=125000",
        ),
    ))

    # 18. translator.translate -> chat_ott.list_chats -> chat_ott.send_message
    city, prof, loc = _make_profile(1)
    def plan_p18(r: SimulatorRunner):
        t_res = r.step("translator.translate", {"text": "Tôi bị dị ứng hải sản", "to_lang": "en"})
        c_res = r.step("chat_ott.list_chats", {})
        target_chat = next(c for c in c_res["chats"] if "Lan" in c.get("name", ""))
        r.step("chat_ott.send_message", {"chat_id": target_chat["id"], "content": t_res["translation"]})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_018_translator_chat_ott",
        template_id="tpl_translator_chat_ott",
        split="train",
        scenario_type="multi_step_chain",
        primary_apps=["translator", "chat_ott"],
        installed_apps=["translator", "chat_ott", "sms", "contacts"],
        user_request="Dịch câu 'Tôi bị dị ứng hải sản' sang tiếng Anh rồi gửi tin nhắn qua ứng dụng chat cho Lan giúp mình nhé.",
        current_time="2026-11-20T18:10:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "translator": {
                "dictionary": [
                    {"text": "Tôi bị dị ứng hải sản", "to_lang": "en", "translation": "I am allergic to seafood"}
                ]
            },
            "chat_ott": {
                "chats": [
                    {"id": "chat_tu_01", "name": "Anh Tuấn", "type": "direct", "messages": []},
                    {"id": "chat_lan_02", "name": "Lan Nguyễn", "type": "direct", "messages": []},
                ]
            },
        },
        objective="Translate sentence to English, list OTT chats to find Lan's chat_id, and send the exact translated text.",
        expect_mutation=True,
        plan_fn=plan_p18,
        verify_fn=lambda final_state, **_: (
            len(final_state["chat_ott"]["chats"][1]["messages"]) == 1
            and final_state["chat_ott"]["chats"][1]["messages"][0]["content"] == "I am allergic to seafood",
            "Expected translated message sent to chat_lan_02",
        ),
    ))

    # 19. habit_tracker.list_habits -> habit_tracker.log
    city, prof, loc = _make_profile(0)
    def plan_p19(r: SimulatorRunner):
        h_res = r.step("habit_tracker.list_habits", {})
        target = next(h for h in h_res["habits"] if "đọc sách" in h["name"].lower())
        r.step("habit_tracker.log", {"habit_id": target["id"], "date": "2026-11-20"})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_019_habit_tracker_log",
        template_id="tpl_habit_tracker_log",
        split="train",
        scenario_type="read_before_write",
        primary_apps=["habit_tracker"],
        installed_apps=["habit_tracker", "reminders", "tasks"],
        user_request="Đánh dấu hoàn thành thói quen đọc sách 30 phút cho ngày hôm nay giúp mình.",
        current_time="2026-11-20T21:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "habit_tracker": {
                "habits": [
                    {"id": "habit_water_01", "name": "Uống đủ 2 lít nước", "streak": 4, "last_logged": "2026-11-19"},
                    {"id": "habit_read_02", "name": "Đọc sách 30 phút", "streak": 6, "last_logged": "2026-11-19"},
                ],
                "logs": [],
            }
        },
        objective="List habits, find habit_read_02, and log completion for 2026-11-20.",
        expect_mutation=True,
        plan_fn=plan_p19,
        verify_fn=lambda final_state, **_: (
            final_state["habit_tracker"]["habits"][1]["streak"] == 7
            and len(final_state["habit_tracker"]["logs"]) == 1,
            "Expected streak incremented to 7",
        ),
    ))

    # 20. bill_payment.list_bills -> ewallet.get_balance -> insufficient wallet fallback to source='bank'
    city, prof, loc = _make_profile(1)
    def plan_p20(r: SimulatorRunner):
        b_res = r.step("bill_payment.list_bills", {"type": "water"})
        bill = b_res["bills"][0]
        w_res = r.step("ewallet.get_balance", {})
        src = "ewallet" if w_res["balance"] >= bill["amount"] else "bank"
        r.step("bill_payment.pay", {"bill_id": bill["id"], "source": src})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_020_bill_ewallet_fallback_bank",
        template_id="tpl_bill_ewallet_fallback_bank",
        split="train",
        scenario_type="arithmetic_condition",
        primary_apps=["bill_payment", "ewallet"],
        installed_apps=["bill_payment", "ewallet", "banking", "sms"],
        user_request="Kiểm tra hóa đơn tiền nước tháng này rồi thanh toán bằng ví điện tử, nếu ví không đủ tiền thì chuyển sang thanh toán qua ngân hàng nhé.",
        current_time="2026-11-20T09:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "bill_payment": {
                "bills": [
                    {"id": "bill_wt_01", "type": "water", "amount": 245000, "paid": False}
                ],
                "autopays": [],
            },
            "ewallet": {
                "balance": 120000,
                "transactions": [],
            },
        },
        objective="Check water bill (245,000) and ewallet balance (120,000); since wallet < bill, pay via source='bank'.",
        expect_mutation=True,
        plan_fn=plan_p20,
        verify_fn=lambda final_state, **_: (
            final_state["bill_payment"]["bills"][0]["paid"] is True
            and final_state["bill_payment"]["bills"][0]["paid_with"] == "bank",
            "Expected bill paid with bank",
        ),
    ))

    # 21. smart_home.get_status -> selective fan_set in rooms where climate is off
    city, prof, loc = _make_profile(0)
    def plan_p21(r: SimulatorRunner):
        st = r.step("smart_home.get_status", {})
        devices = st["devices"]
        ac_on_rooms = {
            d.get("room")
            for d in devices
            if d.get("type") == "climate" and (d.get("state") or {}).get("mode") not in (None, "off")
        }
        for d in devices:
            if d.get("type") == "fan" and d.get("room") not in ac_on_rooms:
                r.step("smart_home.fan_set", {"device_id": d["id"], "power": "on", "speed": 3})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_021_smarthome_selective_fan",
        template_id="tpl_smarthome_selective_fan",
        split="train",
        scenario_type="multi_result_selection",
        primary_apps=["smart_home"],
        installed_apps=["smart_home", "alarm"],
        user_request="Bật hết quạt trong nhà lên tốc độ số 3 giúp anh, nhưng phòng nào điều hòa đang bật làm mát thì thôi không bật quạt phòng đó.",
        current_time="2026-11-20T13:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "smart_home": {
                "devices": [
                    {"id": "dev_light_pk", "name": "Đèn phòng khách", "room": "phòng khách", "type": "light", "state": {"power": "on"}},
                    {"id": "dev_fan_pk", "name": "Quạt trần phòng khách", "room": "phòng khách", "type": "fan", "state": {"power": "off", "speed": 1}},
                    {"id": "dev_ac_pk", "name": "Điều hòa phòng khách", "room": "phòng khách", "type": "climate", "state": {"mode": "off", "temperature": 26}},
                    {"id": "dev_fan_pn", "name": "Quạt cây phòng ngủ", "room": "phòng ngủ", "type": "fan", "state": {"power": "off", "speed": 1}},
                    {"id": "dev_ac_pn", "name": "Điều hòa phòng ngủ", "room": "phòng ngủ", "type": "climate", "state": {"mode": "cool", "temperature": 25}},
                ],
                "scenes": [],
            }
        },
        objective="Check device statuses, identify that AC is off in phòng khách and cool in phòng ngủ, turn on only dev_fan_pk at speed 3.",
        expect_mutation=True,
        plan_fn=plan_p21,
        verify_fn=lambda final_state, **_: (
            final_state["smart_home"]["devices"][1]["state"] == {"power": "on", "speed": 3}
            and final_state["smart_home"]["devices"][3]["state"]["power"] == "off",
            "Expected only dev_fan_pk turned on to speed 3",
        ),
    ))

    # 22. Negative condition: weather.get_forecast -> rain=False -> STOP without mutating calendar
    city, prof, loc = _make_profile(0)
    def plan_p22(r: SimulatorRunner):
        f_res = r.step("weather.get_forecast", {"date": "2026-11-21"})
        if f_res["forecast"] and f_res["forecast"][0].get("rain"):
            r.step("calendar.list_events", {"date_from": "2026-11-21", "date_to": "2026-11-21"})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_022_weather_neg_condition_stop",
        template_id="tpl_weather_neg_condition_stop",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["weather", "calendar"],
        installed_apps=["weather", "calendar", "reminders"],
        user_request="Kiểm tra dự báo thời tiết ngày mai 2026-11-21, nếu trời mưa thì hủy lịch giao hữu bóng đá buổi sáng, còn không mưa thì cứ giữ nguyên lịch nhé.",
        current_time="2026-11-20T19:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "weather": {
                "current": [{"location": "Hà Nội", "condition": "quang mây", "temp": 25, "rain": False}],
                "forecast": [{"location": "Hà Nội", "date": "2026-11-21", "condition": "nắng nhẹ", "temp_min": 21, "temp_max": 28, "rain": False}],
            },
            "calendar": {
                "events": [
                    {"id": "evt_soccer_01", "title": "Giao hữu bóng đá", "start": "2026-11-21T08:00:00+07:00", "end": "2026-11-21T10:00:00+07:00", "status": "confirmed"}
                ]
            },
        },
        objective="Check tomorrow's forecast (rain=False) and stop immediately without modifying calendar.",
        expect_mutation=False,
        plan_fn=plan_p22,
        verify_fn=lambda tool_calls, final_state, **_: (
            len(tool_calls) == 1
            and tool_calls[0]["name"] == "weather.get_forecast"
            and final_state["calendar"]["events"][0]["status"] == "confirmed",
            "Expected only weather.get_forecast and untouched calendar",
        ),
    ))

    # 23. Negative condition: stocks.get_quote -> price above threshold -> STOP without buying
    city, prof, loc = _make_profile(1)
    def plan_p23(r: SimulatorRunner):
        q_res = r.step("stocks.get_quote", {"symbol": "VNM"})
        if q_res["quote"]["price"] < 62000:
            r.step("stocks.place_order", {"symbol": "VNM", "side": "buy", "quantity": 100})

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_023_stocks_neg_condition_stop",
        template_id="tpl_stocks_neg_condition_stop",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["stocks"],
        installed_apps=["stocks", "banking"],
        user_request="Xem giá cổ phiếu VNM hiện tại giúp mình, nếu giá giảm xuống dưới 62 nghìn thì đặt mua 100 cổ, còn cao hơn thì thôi không mua.",
        current_time="2026-11-20T10:45:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "stocks": {
                "quotes": [
                    {"symbol": "VNM", "name": "CTCP Sữa Việt Nam", "price": 66800, "change_pct": 0.5}
                ],
                "holdings": [],
                "orders": [],
                "alerts": [],
            }
        },
        objective="Check VNM quote (66800 >= 62000) and stop without placing an order.",
        expect_mutation=False,
        plan_fn=plan_p23,
        verify_fn=lambda tool_calls, final_state, **_: (
            len(tool_calls) == 1
            and tool_calls[0]["name"] == "stocks.get_quote"
            and len(final_state["stocks"]["orders"]) == 0,
            "Expected only stocks.get_quote and 0 orders",
        ),
    ))

    # 24. Self-cancellation ("...À mà thôi") -> 0 tool calls
    city, prof, loc = _make_profile(2)
    def plan_p24(r: SimulatorRunner):
        return

    scenarios.append(ScenarioSpec(
        scenario_id="pilot_024_self_cancellation_stop",
        template_id="tpl_self_cancellation_stop",
        split="train",
        scenario_type="cancellation",
        primary_apps=["train_bus_tickets"],
        installed_apps=["train_bus_tickets", "parking", "calendar"],
        user_request="Tìm đặt cho anh 2 vé xe khách đi Đà Lạt tối thứ Sáu này nhé. À mà thôi, vừa nghe báo đồng nghiệp cho đi nhờ xe riêng rồi, cứ để nguyên không cần đặt nữa.",
        current_time="2026-11-20T11:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "train_bus_tickets": {
                "trips": [
                    {"id": "trip_dl_09", "type": "bus", "operator": "Thành Bưởi", "origin": "TP.HCM", "destination": "Đà Lạt", "date": "2026-11-20", "depart": "23:00", "arrive": "05:30", "price": 300000, "seats_left": 10}
                ],
                "bookings": [],
            }
        },
        objective="Recognize self-cancellation at the end of user utterance and emit 0 tool calls.",
        expect_mutation=False,
        plan_fn=plan_p24,
        verify_fn=lambda tool_calls, **_: (
            len(tool_calls) == 0,
            "Expected 0 tool calls on complete self-cancellation",
        ),
    ))

    return scenarios


def build_expanded_scenarios() -> list[ScenarioSpec]:
    """Expand the validated pilot with 116 additional high-quality, non-duplicated scenarios
    (total 140 scenarios = 112 train + 28 val, exact 80/20 split with disjoint templates/seeds).
    Covers all 11 unseen apps + cross-app workflows + common high-error apps.
    """
    scenarios = list(build_pilot_24_scenarios())
    # Note: pilot_001..pilot_024 has 20 train and 4 val scenarios.
    # We add 92 train and 24 val scenarios below => 112 train (80.0%) and 28 val (20.0%).

    # =========================================================================
    # FAMILY 1: PARKING (8 train + 2 val)
    # =========================================================================
    parking_train_data = [
        ("sim_parking_tr_01", "Mỹ Đình", "Nam Từ Liêm", "2026-11-21T08:00:00+07:00", "2026-11-21T09:00:00+07:00", 4, 20000,
         "Kiểm tra quanh khu Mỹ Đình xem bãi đỗ xe nào còn trống chỗ thì đặt trước 4 tiếng từ 9 giờ sáng nay giúp mình."),
        ("sim_parking_tr_02", "Hồ Gươm", "Hoàn Kiếm", "2026-11-22T18:00:00+07:00", "2026-11-22T19:00:00+07:00", 2, 35000,
         "Tối nay mình lên phố đi bộ, tìm bãi xe gần Hồ Gươm còn chỗ rồi đặt giữ chỗ 2 tiếng lúc 19 giờ tối nay nhé."),
        ("sim_parking_tr_03", "Bệnh viện Bạch Mai", "Đống Đa", "2026-11-23T06:30:00+07:00", "2026-11-23T07:30:00+07:00", 3, 25000,
         "Sáng nay đưa người nhà đi khám ở Bệnh viện Bạch Mai, tìm bãi đỗ xe gần đó còn chỗ và đặt 3 tiếng từ 07:30 giúp anh."),
    ]
    for idx, (sid, area_name, dist_name, cur_t, from_t, hrs, price, req_text) in enumerate(parking_train_data):
        city, prof, loc = _make_profile(idx)
        lot_full = f"lot_full_{idx+1:02d}"
        lot_ok = f"lot_avail_{idx+1:02d}"
        def _plan_pk_res(r: SimulatorRunner, near_q=area_name, ft=from_t, h=hrs):
            res = r.step("parking.find", {"near": near_q})
            chosen = next(l for l in res["lots"] if l.get("available_spots", 0) > 0)
            r.step("parking.reserve", {"lot_id": chosen["id"], "from_time": ft, "hours": h})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id="tpl_parking_find_select_reserve_train",
            split="train",
            scenario_type="multi_result_selection",
            primary_apps=["parking"],
            installed_apps=["parking", "maps", "ewallet"],
            user_request=req_text,
            current_time=cur_t,
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "parking": {
                    "lots": [
                        {"id": lot_full, "name": f"Bãi xe {area_name} 1", "address": f"Số 10 {area_name}", "area": dist_name, "price_per_hour": price, "available_spots": 0},
                        {"id": lot_ok, "name": f"Bãi đỗ xe trung tâm {area_name}", "address": f"Số 28 {area_name}", "area": dist_name, "price_per_hour": price + 5000, "available_spots": 6 + idx},
                    ],
                    "sessions": [],
                }
            },
            objective=f"Find parking near {area_name}, skip full lot {lot_full}, reserve {lot_ok}.",
            expect_mutation=True,
            plan_fn=_plan_pk_res,
            verify_fn=lambda final_state, target_lot=lot_ok, **_: (
                len(final_state["parking"]["sessions"]) == 1 and final_state["parking"]["sessions"][0]["lot_id"] == target_lot,
                f"Expected reservation on {target_lot}",
            ),
        ))

    # Parking end session + negative condition (all lots full) + ask_missing_info (missing hours/from_time)
    city, prof, loc = _make_profile(0)
    def _plan_pk_neg_full(r: SimulatorRunner):
        res = r.step("parking.find", {"near": "Cầu Giấy"})
        avail = [l for l in res["lots"] if l.get("available_spots", 0) > 0]
        if avail:
            r.step("parking.reserve", {"lot_id": avail[0]["id"], "from_time": "2026-11-21T15:00:00+07:00", "hours": 2})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_parking_tr_05_neg_full",
        template_id="tpl_parking_neg_full_train",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["parking"],
        installed_apps=["parking", "maps"],
        user_request="Kiểm tra bãi đỗ xe gần Cầu Giấy xem còn chỗ trống không, nếu còn thì đặt giữ chỗ 2 tiếng lúc 15 giờ chiều nay, hết chỗ rồi thì thôi nhé.",
        current_time="2026-11-21T13:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "parking": {
                "lots": [
                    {"id": "lot_cg_01", "name": "Bãi đỗ xe Cầu Giấy A", "address": "120 Cầu Giấy", "area": "Cầu Giấy", "price_per_hour": 25000, "available_spots": 0},
                    {"id": "lot_cg_02", "name": "Bãi đỗ xe Công viên Cầu Giấy", "address": "Phố Duy Tân", "area": "Cầu Giấy", "price_per_hour": 30000, "available_spots": 0},
                ],
                "sessions": [],
            }
        },
        objective="Check parking near Cầu Giấy; all lots have available_spots=0, so stop without reserving.",
        expect_mutation=False,
        plan_fn=_plan_pk_neg_full,
        verify_fn=lambda tool_calls, final_state, **_: (
            len(tool_calls) == 1 and len(final_state["parking"]["sessions"]) == 0,
            "Expected no reservation when all lots are full",
        ),
    ))

    # Parking missing info (user didn't provide hours & from_time)
    city, prof, loc = _make_profile(1)
    def _plan_pk_ask(r: SimulatorRunner):
        r.step("parking.find", {"near": "Bitexco"})
        r.step("sys.ask_missing_info", {"tool_name": "parking.reserve", "params": ["from_time", "hours"]})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_parking_tr_06_ask_time",
        template_id="tpl_parking_ask_time_train",
        split="train",
        scenario_type="missing_info",
        primary_apps=["parking", "sys"],
        installed_apps=["parking", "maps"],
        user_request="Tìm bãi đỗ xe gần tòa nhà Bitexco rồi đặt giữ chỗ trước giúp anh nhé.",
        current_time="2026-11-21T10:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "parking": {
                "lots": [
                    {"id": "lot_btx_01", "name": "Bãi đỗ xe Hải Triều Bitexco", "address": "2 Hải Triều, Quận 1", "area": "Quận 1", "price_per_hour": 40000, "available_spots": 10}
                ],
                "sessions": [],
            }
        },
        objective="Find parking lot near Bitexco, then ask user for missing required parameters from_time and hours.",
        expect_mutation=False,
        plan_fn=_plan_pk_ask,
    ))

    # Parking end active session among multiple sessions (2 train + 2 val)
    for idx, (sid, sp, area_label, sess_id, req_text) in enumerate([
        ("sim_parking_tr_07_end", "train", "Keangnam", "park_kn_02", "Mình lấy xe ra khỏi bãi đỗ xe Keangnam rồi, kết thúc phiên đỗ xe đang hoạt động giúp mình."),
        ("sim_parking_tr_08_end", "train", "Sân bay Nội Bài", "park_nb_03", "Kết thúc giúp anh lượt gửi xe đang mở ở bãi xe Sân bay Nội Bài nhé."),
        ("sim_parking_val_01_end", "val", "Nhà hát Lớn", "park_nhl_02", "Chị vừa đánh xe ra khỏi bãi Nhà hát Lớn, đóng phiên đỗ xe đang chạy giúp chị nha."),
    ]):
        city, prof, loc = _make_profile(idx)
        def _plan_pk_end(r: SimulatorRunner, target_area=area_label):
            res = r.step("parking.list_sessions", {})
            chosen = next(s for s in res["sessions"] if s.get("status") != "ended" and target_area.lower() in (s.get("lot_name") or "").lower())
            r.step("parking.end_session", {"session_id": chosen["id"]})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_parking_end_{sp}",
            split=sp,
            scenario_type="read_before_write",
            primary_apps=["parking"],
            installed_apps=["parking", "ewallet"],
            user_request=req_text,
            current_time="2026-11-22T17:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "parking": {
                    "lots": [
                        {"id": "lot_old_01", "name": "Bãi xe Tràng Thi", "address": "15 Tràng Thi", "area": "Hoàn Kiếm", "price_per_hour": 25000, "available_spots": 5},
                        {"id": "lot_cur_02", "name": f"Bãi xe {area_label}", "address": area_label, "area": area_label, "price_per_hour": 30000, "available_spots": 4},
                    ],
                    "sessions": [
                        {"id": "park_old_01", "lot_id": "lot_old_01", "from_time": "2026-11-20T08:00:00+07:00", "hours": 2, "end_time": "2026-11-20T10:00:00+07:00", "total": 50000, "status": "ended"},
                        {"id": sess_id, "lot_id": "lot_cur_02", "from_time": "2026-11-22T14:00:00+07:00", "hours": 3, "end_time": "2026-11-22T17:00:00+07:00", "total": 90000, "status": "reserved"},
                    ],
                }
            },
            objective=f"List parking sessions and end active session {sess_id}.",
            expect_mutation=True,
            plan_fn=_plan_pk_end,
        ))

    # Val parking reserve scenario (disjoint template phrasing & location)
    city, prof, loc = _make_profile(4)
    def _plan_pk_val_res(r: SimulatorRunner):
        res = r.step("parking.find", {"near": "Thảo Điền"})
        chosen = next(l for l in res["lots"] if l.get("available_spots", 0) > 0)
        r.step("parking.reserve", {"lot_id": chosen["id"], "from_time": "2026-11-25T18:30:00+07:00", "hours": 2})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_parking_val_02_reserve",
        template_id="tpl_parking_reserve_val",
        split="val",
        scenario_type="multi_result_selection",
        primary_apps=["parking"],
        installed_apps=["parking", "maps", "ewallet"],
        user_request="Tra giúp anh bãi đậu ô tô khu Thảo Điền còn chỗ trống để đặt giữ chỗ từ 18:30 chiều nay trong vòng 2 giờ.",
        current_time="2026-11-25T16:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "parking": {
                "lots": [
                    {"id": "lot_td_01", "name": "Bãi xe Xuân Thủy Thảo Điền", "address": "40 Xuân Thủy", "area": "Thảo Điền", "price_per_hour": 30000, "available_spots": 0},
                    {"id": "lot_td_02", "name": "Bãi xe Vincom Mega Mall Thảo Điền", "address": "161 Xa lộ Hà Nội", "area": "Thảo Điền", "price_per_hour": 35000, "available_spots": 15},
                ],
                "sessions": [],
            }
        },
        objective="Find parking in Thảo Điền with spots > 0 and reserve lot_td_02.",
        expect_mutation=True,
        plan_fn=_plan_pk_val_res,
    ))

    # =========================================================================
    # FAMILY 2: STOCKS (10 train + 3 val)
    # =========================================================================
    stocks_buy_cases = [
        ("sim_stocks_tr_01_buy", "train", "MWG", "Thế Giới Di Động", 51200, 300,
         "Kiểm tra thị giá mã cổ phiếu MWG hiện tại rồi đặt lệnh mua 300 cổ phiếu theo đúng mức giá đó giúp anh."),
        ("sim_stocks_tr_02_buy", "train", "TCB", "Techcombank", 35800, 500,
         "Xem mã TCB đang giao dịch giá bao nhiêu rồi vào lệnh mua 500 cổ phiếu theo giá hiện tại nhé."),
        ("sim_stocks_val_01_buy", "val", "MBB", "Ngân hàng Quân Đội", 24600, 600,
         "Báo giá mã MBB hiện tại rồi đặt mua giúp chị 600 cổ phiếu MBB theo giá thị trường vừa xem."),
    ]
    for idx, (sid, sp, sym, cname, pr, qty, req_text) in enumerate(stocks_buy_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_stk_buy(r: SimulatorRunner, s=sym, q=qty):
            q_res = r.step("stocks.get_quote", {"symbol": s})
            r.step("stocks.place_order", {"symbol": s, "side": "buy", "quantity": q, "price": q_res["quote"]["price"]})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_stocks_quote_buy_{sp}",
            split=sp,
            scenario_type="read_before_write",
            primary_apps=["stocks"],
            installed_apps=["stocks", "banking", "ewallet"],
            user_request=req_text,
            current_time="2026-11-23T10:15:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "stocks": {
                    "quotes": [{"symbol": sym, "name": cname, "price": pr, "change_pct": 1.4}],
                    "holdings": [],
                    "orders": [],
                    "alerts": [],
                }
            },
            objective=f"Get quote for {sym} and place buy order for {qty} shares at {pr}.",
            expect_mutation=True,
            plan_fn=_plan_stk_buy,
        ))

    # Stocks portfolio arithmetic: sell half & sell keeping X & set price alert
    stocks_arith_cases = [
        ("sim_stocks_tr_04_half", "train", "VIC", "Tập đoàn Vingroup", 1000, 42500, "half", 0,
         "Xem trong danh mục mình có bao nhiêu cổ phiếu VIC rồi đặt bán một nửa số lượng đang nắm giữ theo giá hiện tại nhé."),
        ("sim_stocks_tr_06_keep", "train", "ACB", "Ngân hàng Á Châu", 900, 27500, "keep", 300,
         "Kiểm tra số lượng cổ phiếu ACB trong tài khoản, giữ lại 300 cổ phiếu, số còn lại đặt lệnh bán hết theo giá hiện tại."),
        ("sim_stocks_tr_07_keep", "train", "VHM", "Vinhomes", 500, 41200, "keep", 100,
         "Xem danh mục cổ phiếu VHM của chị rồi bán bớt đi, chỉ để lại 100 cổ phiếu thôi nhé."),
        ("sim_stocks_val_02_half", "val", "CTG", "VietinBank", 1200, 34500, "half", 0,
         "Mở danh mục chứng khoán kiểm tra mã CTG rồi đặt lệnh bán một nửa số cổ phiếu CTG đang sở hữu theo giá hiện tại."),
    ]
    for idx, (sid, sp, sym, cname, held, pr, mode, keep_n, req_text) in enumerate(stocks_arith_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_stk_arith(r: SimulatorRunner, s=sym, m=mode, k=keep_n):
            p_res = r.step("stocks.get_portfolio", {})
            h = next(x for x in p_res["holdings"] if x["symbol"] == s)
            sell_q = (h["quantity"] // 2) if m == "half" else (h["quantity"] - k)
            r.step("stocks.place_order", {"symbol": s, "side": "sell", "quantity": sell_q, "price": h["price"]})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_stocks_arith_{mode}_{sp}",
            split=sp,
            scenario_type="arithmetic_condition",
            primary_apps=["stocks"],
            installed_apps=["stocks", "banking"],
            user_request=req_text,
            current_time="2026-11-23T14:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "stocks": {
                    "holdings": [{"symbol": sym, "name": cname, "quantity": held, "avg_price": pr - 1500, "price": pr}],
                    "quotes": [],
                    "orders": [],
                    "alerts": [],
                }
            },
            objective=f"Read portfolio for {sym} (held={held}) and place arithmetic sell order.",
            expect_mutation=True,
            plan_fn=_plan_stk_arith,
        ))

    # Stocks price alert + negative conditional + quote->alert
    for idx, (sid, sp, sym, cname, cur_p, target_p, direction, req_text) in enumerate([
        ("sim_stocks_tr_08_alert", "train", "FPT", "CTCP FPT", 132000, 140000, "above",
         "Xem giá cổ phiếu FPT hiện tại rồi cài cảnh báo khi giá FPT vượt lên trên 140000 đồng giúp anh."),
        ("sim_stocks_tr_09_alert", "train", "HPG", "Hòa Phát", 28500, 27000, "below",
         "Kiểm tra mã HPG rồi đặt mức cảnh báo giá khi HPG giảm xuống dưới 27000 đồng nhé."),
        ("sim_stocks_val_03_alert", "val", "MWG", "Thế Giới Di Động", 52000, 55000, "above",
         "Tra giá cổ phiếu MWG và bật cảnh báo khi giá vượt mốc 55000 đồng giúp mình."),
    ]):
        city, prof, loc = _make_profile(idx)
        def _plan_stk_alert(r: SimulatorRunner, s=sym, tp=target_p, d=direction):
            r.step("stocks.get_quote", {"symbol": s})
            r.step("stocks.set_price_alert", {"symbol": s, "price": tp, "direction": d})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_stocks_alert_{sp}",
            split=sp,
            scenario_type="read_before_write",
            primary_apps=["stocks"],
            installed_apps=["stocks", "reminders"],
            user_request=req_text,
            current_time="2026-11-23T11:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "stocks": {
                    "quotes": [{"symbol": sym, "name": cname, "price": cur_p, "change_pct": 0.4}],
                    "holdings": [],
                    "orders": [],
                    "alerts": [],
                }
            },
            objective=f"Check quote for {sym} and set price alert {direction} {target_p}.",
            expect_mutation=True,
            plan_fn=_plan_stk_alert,
        ))

    # Stocks negative conditional (train)
    city, prof, loc = _make_profile(3)
    def _plan_stk_neg2(r: SimulatorRunner):
        q_res = r.step("stocks.get_quote", {"symbol": "TCB"})
        if q_res["quote"]["price"] <= 33000:
            r.step("stocks.place_order", {"symbol": "TCB", "side": "buy", "quantity": 300})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_stocks_tr_10_neg_cond",
        template_id="tpl_stocks_neg_cond_train",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["stocks"],
        installed_apps=["stocks", "banking"],
        user_request="Kiểm tra giá mã TCB giúp anh, nếu giá xuống từ 33 nghìn trở xuống thì mua 300 cổ phiếu, còn đang cao hơn thì thôi không đặt lệnh.",
        current_time="2026-11-23T10:50:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "stocks": {
                "quotes": [{"symbol": "TCB", "name": "Techcombank", "price": 36200, "change_pct": 1.8}],
                "holdings": [],
                "orders": [],
                "alerts": [],
            }
        },
        objective="Check TCB quote (36200 > 33000) and stop without buying.",
        expect_mutation=False,
        plan_fn=_plan_stk_neg2,
    ))

    # =========================================================================
    # FAMILY 3: TRAIN_BUS_TICKETS & CROSS-APP CALENDAR (10 train + 3 val)
    # =========================================================================
    tb_routes_cases = [
        ("sim_tb_tr_01_book", "train", "Hà Nội", "Đà Nẵng", "2026-11-26", "train", 2, "07:30", "SE1", 480000,
         "Tìm chuyến tàu hỏa từ Hà Nội đi Đà Nẵng ngày 2026-11-26 còn đủ 2 chỗ rồi đặt 2 vé giúp anh."),
        ("sim_tb_tr_02_book", "train", "TP.HCM", "Nha Trang", "2026-11-27", "train", 4, "08:15", "SNT2", 390000,
         "Tra cứu vé tàu từ TP.HCM đi Nha Trang ngày 2026-11-27, chọn chuyến còn trống từ 4 ghế trở lên và đặt 4 vé nhé."),
        ("sim_tb_tr_03_book", "train", "TP.HCM", "Vũng Tàu", "2026-11-25", "bus", 2, "09:00", "Hoa Mai Limousine", 180000,
         "Tìm xe khách từ TP.HCM đi Vũng Tàu ngày 2026-11-25 rồi đặt 2 ghế ở chuyến còn đủ chỗ trống giúp chị."),
        ("sim_tb_val_01_book", "val", "Hà Nội", "Vinh", "2026-11-29", "train", 2, "21:30", "NA1", 320000,
         "Kiểm tra tàu lửa Hà Nội đi Vinh ngày 2026-11-29 xem chuyến nào còn đủ 2 ghế thì đặt giúp mình 2 vé."),
    ]
    for idx, (sid, sp, orig, dest, dt, vtype, seats, dep, op, pr, req_text) in enumerate(tb_routes_cases):
        city, prof, loc = _make_profile(idx)
        t_full = f"trip_full_{idx+1:02d}"
        t_ok = f"trip_ok_{idx+1:02d}"
        def _plan_tb_book(r: SimulatorRunner, o=orig, d=dest, date_s=dt, vt=vtype, st=seats):
            res = r.step("train_bus_tickets.search", {"origin": o, "destination": d, "date": date_s, "type": vt})
            chosen = next(t for t in res["trips"] if t.get("seats_left", 0) >= st)
            r.step("train_bus_tickets.book", {"trip_id": chosen["id"], "seats": st})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_tb_search_select_book_{sp}",
            split=sp,
            scenario_type="multi_result_selection",
            primary_apps=["train_bus_tickets"],
            installed_apps=["train_bus_tickets", "flights", "ewallet"],
            user_request=req_text,
            current_time="2026-11-21T09:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "train_bus_tickets": {
                    "trips": [
                        {"id": t_full, "type": vtype, "operator": f"{op} Sớm", "origin": orig, "destination": dest, "date": dt, "depart": "05:30", "arrive": "11:00", "price": pr, "seats_left": seats - 1},
                        {"id": t_ok, "type": vtype, "operator": op, "origin": orig, "destination": dest, "date": dt, "depart": dep, "arrive": "14:30", "price": pr, "seats_left": seats + 8},
                    ],
                    "bookings": [],
                }
            },
            objective=f"Search {vtype} from {orig} to {dest}, skip {t_full} with insufficient seats, book {seats} seats on {t_ok}.",
            expect_mutation=True,
            plan_fn=_plan_tb_book,
        ))

    # Cross-app: train_bus_tickets.search -> book -> calendar.create_event (3 train + 1 val)
    tb_cal_cases = [
        ("sim_tb_tr_05_cal", "train", "Hà Nội", "Hải Phòng", "2026-11-25", "train", "07:00",
         "Đặt 1 vé tàu hỏa từ Hà Nội đi Hải Phòng ngày 2026-11-25 rồi tạo sự kiện lịch cho chuyến tàu đó giúp tôi."),
        ("sim_tb_tr_07_cal", "train", "TP.HCM", "Cần Thơ", "2026-11-27", "bus", "08:30",
         "Đặt giúp mình 1 vé xe khách từ TP.HCM đi Cần Thơ ngày 2026-11-27 và ghi vào lịch cá nhân theo giờ xe chạy."),
        ("sim_tb_val_02_cal", "val", "Hà Nội", "Thanh Hóa", "2026-11-28", "train", "08:00",
         "Tìm chuyến tàu Hà Nội đi Thanh Hóa ngày 2026-11-28 đặt 1 vé rồi thêm lịch trình chuyến đi vào Lịch giúp chị."),
    ]
    for idx, (sid, sp, orig, dest, dt, vtype, dep_hm, req_text) in enumerate(tb_cal_cases):
        city, prof, loc = _make_profile(idx)
        tid = f"trip_cal_{idx+1:02d}"
        seats_n = 2 if "2 vé" in req_text else 1
        def _plan_tb_cal(r: SimulatorRunner, o=orig, d=dest, date_s=dt, vt=vtype, st=seats_n):
            res = r.step("train_bus_tickets.search", {"origin": o, "destination": d, "date": date_s, "type": vt})
            chosen = res["trips"][0]
            r.step("train_bus_tickets.book", {"trip_id": chosen["id"], "seats": st})
            start_iso = f"{chosen['date']}T{chosen['depart']}:00+07:00"
            r.step("calendar.create_event", {"title": f"Chuyến {vt} {o} - {d}", "start": start_iso, "location": o})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_tb_book_calendar_{sp}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=["train_bus_tickets", "calendar"],
            installed_apps=["train_bus_tickets", "calendar", "reminders"],
            user_request=req_text,
            current_time="2026-11-21T10:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "train_bus_tickets": {
                    "trips": [
                        {"id": tid, "type": vtype, "operator": "Tàu Chất Lượng Cao", "origin": orig, "destination": dest, "date": dt, "depart": dep_hm, "arrive": "11:30", "price": 220000, "seats_left": 12}
                    ],
                    "bookings": [],
                },
                "calendar": {"events": []},
            },
            objective=f"Search & book {tid}, then create calendar event using the trip's actual date and depart time.",
            expect_mutation=True,
            plan_fn=_plan_tb_cal,
        ))

    # Train/bus cancel + missing_info (3 train + 1 val)
    for idx, (sid, sp, bk_id, route_desc, req_text) in enumerate([
        ("sim_tb_tr_08_cancel", "train", "tk_can_01", "Hà Nội - Sa Pa", "Kiểm tra danh sách vé đã đặt rồi hủy vé xe đi Sa Pa giúp anh nhé."),
        ("sim_tb_tr_09_cancel", "train", "tk_can_02", "TP.HCM - Nha Trang", "Mở mục vé của tôi và hủy vé tàu đi Nha Trang đang xác nhận giúp mình."),
        ("sim_tb_val_03_cancel", "val", "tk_can_03", "Đà Nẵng - Quảng Bình", "Xem lại vé tàu đã đặt rồi hủy giúp chị chiếc vé đi Quảng Bình nhé."),
    ]):
        city, prof, loc = _make_profile(idx)
        def _plan_tb_can(r: SimulatorRunner):
            b_res = r.step("train_bus_tickets.my_bookings", {})
            active = next(b for b in b_res["bookings"] if b.get("status") == "confirmed")
            r.step("train_bus_tickets.cancel", {"booking_id": active["id"]})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_tb_cancel_{sp}",
            split=sp,
            scenario_type="read_before_write",
            primary_apps=["train_bus_tickets"],
            installed_apps=["train_bus_tickets", "ewallet"],
            user_request=req_text,
            current_time="2026-11-22T12:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "train_bus_tickets": {
                    "trips": [{"id": "trip_c_01", "type": "train", "operator": "ĐSVN", "origin": "Hà Nội", "destination": "Sa Pa", "date": "2026-11-29", "depart": "21:00", "arrive": "05:00", "price": 350000, "seats_left": 5}],
                    "bookings": [{"id": bk_id, "trip_id": "trip_c_01", "seats": 1, "total": 350000, "status": "confirmed"}],
                }
            },
            objective=f"Read bookings and cancel {bk_id}.",
            expect_mutation=True,
            plan_fn=_plan_tb_can,
        ))

    # Train/bus missing date -> ask_missing_info
    city, prof, loc = _make_profile(2)
    scenarios.append(ScenarioSpec(
        scenario_id="sim_tb_val_04_ask_date",
        template_id="tpl_tb_ask_date_val",
        split="val",
        scenario_type="missing_info",
        primary_apps=["train_bus_tickets", "sys"],
        installed_apps=["train_bus_tickets", "calendar"],
        user_request="Tìm và đặt giúp anh 2 vé tàu từ Hà Nội đi Đà Nẵng nhé.",
        current_time="2026-11-22T09:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={"train_bus_tickets": {"trips": [], "bookings": []}},
        objective="User did not specify travel date required by train_bus_tickets.search; call sys.ask_missing_info.",
        expect_mutation=False,
        plan_fn=lambda r: r.step("sys.ask_missing_info", {"tool_name": "train_bus_tickets.search", "params": ["date"]}),
    ))

    # =========================================================================
    # FAMILY 4: ECOMMERCE (10 train + 3 val)
    # =========================================================================
    ecom_buy_cases = [
        ("sim_ecom_tr_01_buy", "train", "sạc dự phòng", 450000, "prod_sdp_02", "Sạc dự phòng Anker 10000mAh", 390000, "cod", "home",
         "Tìm sạc dự phòng giá dưới 450 nghìn trên sàn mua sắm, chọn mẫu đang còn hàng thêm 1 cái vào giỏ rồi đặt giao về nhà thanh toán tiền mặt khi nhận (COD)."),
        ("sim_ecom_tr_02_buy", "train", "chuột công thái học", 600000, "prod_ms_02", "Chuột công thái học Logitech Lift", 540000, "ewallet", "work",
         "Tìm mua chuột công thái học tối đa 600 nghìn trên sàn thương mại điện tử, lấy mẫu còn hàng đặt 1 chiếc giao tới công ty, trả qua ví điện tử nhé."),
        ("sim_ecom_tr_03_buy", "train", "balo laptop", 500000, "prod_bl_02", "Balo laptop chống nước Mikkor", 430000, "bank", "home",
         "Kiếm trên ứng dụng mua sắm mẫu balo laptop giá không quá 500 nghìn còn hàng, thêm 1 cái vào giỏ và thanh toán qua ngân hàng giao về nhà."),
        ("sim_ecom_val_01_buy", "val", "nồi chiên không dầu", 1500000, "prod_nc_02", "Nồi chiên không dầu Philips 4.1L", 1350000, "bank", "home",
         "Tìm nồi chiên không dầu tầm giá dưới 1 triệu rưỡi trên sàn thương mại điện tử, chọn loại còn hàng đặt 1 chiếc về nhà thanh toán chuyển khoản ngân hàng."),
    ]
    for idx, (sid, sp, q_str, max_p, pid_ok, pname_ok, pr_ok, pay_m, addr_type, req_text) in enumerate(ecom_buy_cases):
        city, prof, loc = _make_profile(idx)
        target_addr = prof["home_address"] if addr_type == "home" else prof["work_address"]
        qty_n = 2 if "mua 2 cái" in req_text else 1
        def _plan_ec_buy(r: SimulatorRunner, q=q_str, mp=max_p, qt=qty_n, ad=target_addr, pm=pay_m):
            res = r.step("ecommerce.search", {"query": q, "max_price": mp})
            chosen = next(p for p in res["products"] if p.get("in_stock") is True)
            r.step("ecommerce.add_to_cart", {"product_id": chosen["id"], "quantity": qt})
            r.step("ecommerce.checkout", {"address": ad, "payment": pm})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_ecom_search_add_checkout_{sp}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=["ecommerce"],
            installed_apps=["ecommerce", "ewallet", "banking"],
            user_request=req_text,
            current_time="2026-11-22T14:20:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "ecommerce": {
                    "products": [
                        {"id": f"prod_oos_{idx:02d}", "name": f"{q_str.capitalize()} Bản Cũ", "price": pr_ok - 30000, "seller": "Shop A", "category": "gia dụng", "rating": 4.2, "in_stock": False},
                        {"id": pid_ok, "name": pname_ok, "price": pr_ok, "seller": "Official Store", "category": "gia dụng", "rating": 4.9, "in_stock": True},
                    ],
                    "cart": [],
                    "orders": [],
                    "returns": [],
                }
            },
            objective=f"Search {q_str} <= {max_p}, select in_stock {pid_ok}, add {qty_n} to cart, checkout.",
            expect_mutation=True,
            plan_fn=_plan_ec_buy,
        ))

    # Ecommerce returns & cancellations & negative condition
    ecom_post_cases = [
        ("sim_ecom_tr_05_return", "train", "return", "wrong_item", "ord_ec_r1", "Áo sơ mi trắng",
         "Kiểm tra đơn hàng áo sơ mi đã giao trên sàn mua sắm rồi gửi yêu cầu trả hàng vì shop giao sai mẫu nhé."),
        ("sim_ecom_tr_06_return", "train", "return", "changed_mind", "ord_ec_r2", "Giá đỡ laptop nhôm",
         "Xem danh sách đơn hàng đã giao trên sàn thương mại điện tử và tạo yêu cầu hoàn trả đơn giá đỡ laptop do mình đổi ý không dùng nữa."),
        ("sim_ecom_val_02_return", "val", "return", "damaged", "ord_ec_r3", "Gương trang điểm để bàn",
         "Tra cứu đơn hàng gương trang điểm đã nhận trên sàn rồi bấm yêu cầu trả hàng vì mặt gương bị nứt vỡ khi giao."),
        ("sim_ecom_tr_07_cancel", "train", "cancel", "", "ord_ec_c1", "Quạt tích điện mini",
         "Kiểm tra các đơn hàng trên sàn, đơn mua quạt tích điện mini đang chờ lấy hàng thì hủy đơn giúp anh."),
    ]
    for idx, (sid, sp, act, r_reason, oid, item_label, req_text) in enumerate(ecom_post_cases):
        city, prof, loc = _make_profile(idx)
        if act == "return":
            def _plan_ec_post(r: SimulatorRunner, rsn=r_reason):
                res = r.step("ecommerce.order_status", {})
                target = next(o for o in res["orders"] if o.get("status") == "delivered")
                r.step("ecommerce.request_return", {"order_id": target["id"], "reason": rsn})
            orders_init = [
                {"id": oid, "items": [{"product_id": "p_1", "name": item_label, "price": 320000, "quantity": 1}], "total": 320000, "status": "delivered"},
                {"id": "ord_other_ship", "items": [{"product_id": "p_2", "name": "Ốp lưng điện thoại", "price": 50000, "quantity": 1}], "total": 50000, "status": "shipping"},
            ]
        else:
            def _plan_ec_post(r: SimulatorRunner):
                res = r.step("ecommerce.order_status", {})
                target = next(o for o in res["orders"] if o.get("status") in ("placed", "shipping"))
                r.step("ecommerce.cancel_order", {"order_id": target["id"]})
            orders_init = [
                {"id": "ord_old_deliv", "items": [{"product_id": "p_0", "name": "Cáp sạc Type-C", "price": 90000, "quantity": 1}], "total": 90000, "status": "delivered"},
                {"id": oid, "items": [{"product_id": "p_1", "name": item_label, "price": 260000, "quantity": 1}], "total": 260000, "status": "placed" if idx % 2 == 0 else "shipping"},
            ]

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_ecom_{act}_{sp}",
            split=sp,
            scenario_type="multi_result_selection",
            primary_apps=["ecommerce"],
            installed_apps=["ecommerce", "ewallet"],
            user_request=req_text,
            current_time="2026-11-23T15:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={"ecommerce": {"products": [], "cart": [], "orders": orders_init, "returns": []}},
            objective=f"Inspect ecommerce orders and perform {act} on {oid}.",
            expect_mutation=True,
            plan_fn=_plan_ec_post,
        ))

    # Ecommerce negative condition (all products out of stock or above max_price) + ambiguous ask
    city, prof, loc = _make_profile(0)
    def _plan_ec_neg(r: SimulatorRunner):
        res = r.step("ecommerce.search", {"query": "máy đọc sách", "max_price": 2000000})
        avail = [p for p in res["products"] if p.get("in_stock")]
        if avail:
            r.step("ecommerce.add_to_cart", {"product_id": avail[0]["id"], "quantity": 1})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_ecom_tr_09_neg_oos",
        template_id="tpl_ecom_neg_oos_train",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["ecommerce"],
        installed_apps=["ecommerce", "ewallet"],
        user_request="Tìm trên sàn xem có máy đọc sách nào giá dưới 2 triệu mà còn hàng không, nếu có thì thêm vào giỏ hàng, hết hàng hoặc không có thì thôi nhé.",
        current_time="2026-11-23T16:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "ecommerce": {
                "products": [
                    {"id": "prod_knd_01", "name": "Máy đọc sách Kindle Basic", "price": 1950000, "seller": "Kindle VN", "category": "điện tử", "in_stock": False}
                ],
                "cart": [],
                "orders": [],
                "returns": [],
            }
        },
        objective="Search e-reader <= 2M; only match is out of stock (in_stock=False), so stop without adding to cart.",
        expect_mutation=False,
        plan_fn=_plan_ec_neg,
    ))

    city, prof, loc = _make_profile(1)
    def _plan_ec_ask(r: SimulatorRunner):
        r.step("ecommerce.search", {"query": "bàn phím cơ"})
        r.step("sys.ask_missing_info", {"tool_name": "ecommerce.add_to_cart", "params": ["product_id"]})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_ecom_tr_10_ask_ambig",
        template_id="tpl_ecom_ask_ambig_train",
        split="train",
        scenario_type="missing_info",
        primary_apps=["ecommerce", "sys"],
        installed_apps=["ecommerce", "ewallet"],
        user_request="Mua giúp mình 1 chiếc bàn phím cơ trên ứng dụng thương mại điện tử nhé.",
        current_time="2026-11-23T16:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "ecommerce": {
                "products": [
                    {"id": "prod_kb_11", "name": "Bàn phím cơ Keychron K2", "price": 1650000, "seller": "Keychron", "category": "phụ kiện", "in_stock": True},
                    {"id": "prod_kb_12", "name": "Bàn phím cơ Akko 3087", "price": 1150000, "seller": "Akko", "category": "phụ kiện", "in_stock": True},
                ],
                "cart": [],
                "orders": [],
                "returns": [],
            }
        },
        objective="Search mechanical keyboards, find 2 distinct models, ask user to clarify product_id.",
        expect_mutation=False,
        plan_fn=_plan_ec_ask,
    ))

    # =========================================================================
    # FAMILY 5: GROCERY, PANTRY, RECIPES & MULTI-APP CHAINS (18 train + 5 val)
    # =========================================================================
    grocery_buy_cases = [
        ("sim_grocery_tr_01", "train", "gạo ST25", "gro_rice_01", "Gạo ST25 Ông Cua túi 5kg", 185000, 1, "18:00",
         "Tìm mua 1 túi gạo ST25 trên ứng dụng đi chợ rồi đặt giao về nhà lúc 18:00 chiều nay nhé."),
        ("sim_grocery_tr_02", "train", "thịt ba chỉ bò", "gro_beef_02", "Thịt ba chỉ bò Mỹ khay 500g", 145000, 2, None,
         "Đặt mua 2 khay thịt ba chỉ bò trên app đi chợ online và thanh toán giao về địa chỉ nhà giúp chị."),
        ("sim_grocery_tr_03", "train", "cá hồi phi lê", "gro_sal_03", "Cá hồi Nauy phi lê tươi 300g", 175000, 1, "17:30",
         "Tìm cá hồi phi lê trên app đi chợ, thêm 1 phần vào giỏ rồi đặt giao về nhà vào 17:30 giúp mình."),
        ("sim_grocery_val_01", "val", "ức gà phi lê", "gro_chk_01", "Ức gà phi lê sạch CP 500g", 52000, 3, "16:30",
         "Tìm ức gà phi lê trên app đi chợ, thêm 3 khay vào giỏ hàng rồi đặt giao về nhà lúc 16:30 chiều nay."),
    ]
    for idx, (sid, sp, q_str, gid, gname, gprice, qty, deliv_t, req_text) in enumerate(grocery_buy_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_gro(r: SimulatorRunner, q=q_str, qt=qty, ad=prof["home_address"], dt=deliv_t):
            res = r.step("grocery.search", {"query": q})
            chosen = next(p for p in res["products"] if p.get("in_stock") is True)
            r.step("grocery.add_to_cart", {"product_id": chosen["id"], "quantity": qt})
            ck_args = {"address": ad}
            if dt:
                ck_args["delivery_time"] = dt
            r.step("grocery.checkout", ck_args)

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_grocery_buy_{sp}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=["grocery"],
            installed_apps=["grocery", "pantry", "shopping_list"],
            user_request=req_text,
            current_time="2026-11-22T09:30:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "grocery": {
                    "products": [
                        {"id": gid, "name": gname, "price": gprice, "unit": "phần", "category": "thực phẩm", "in_stock": True}
                    ],
                    "cart": [],
                    "orders": [],
                }
            },
            objective=f"Search {q_str}, add {qty} of {gid} to cart, and checkout.",
            expect_mutation=True,
            plan_fn=_plan_gro,
        ))

    # Pantry consume + negative stock check (4 train + 1 val)
    pantry_cases = [
        ("sim_pantry_tr_01", "train", "pan_egg_01", "trứng gà", 10, 4, "quả", True,
         "Sáng nay nhà mình vừa chiên 4 quả trứng gà, cập nhật trừ số lượng trứng gà trong tủ bếp giúp anh."),
        ("sim_pantry_tr_02", "train", "pan_rice_02", "gạo thơm", 8, 2, "kg", True,
         "Trừ giúp chị 2 kg gạo thơm vừa nấu trong danh sách tủ bếp nhé."),
        ("sim_pantry_tr_04_neg_stock", "train", "pan_beef_04", "thịt bò", 1, 3, "khay", False,
         "Kiểm tra trong tủ bếp xem còn đủ 3 khay thịt bò không, nếu đủ thì trừ 3 khay để nấu lẩu, không đủ thì thôi để mình đi mua thêm."),
        ("sim_pantry_val_01", "val", "pan_potato_01", "khoai tây", 6, 2, "củ", True,
         "Cập nhật tủ bếp giúp mình: vừa lấy 2 củ khoai tây ra nấu canh rồi nhé."),
    ]
    for idx, (sid, sp, pid, pname, stock_init, use_q, unit_s, should_mut, req_text) in enumerate(pantry_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_pan(r: SimulatorRunner, target_name=pname, uq=use_q):
            res = r.step("pantry.list_items", {})
            it = next(x for x in res["items"] if target_name in x["name"].lower())
            if it.get("stock", 0) >= uq:
                r.step("pantry.consume", {"item_id": it["id"], "quantity": uq})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_pantry_consume_{sp}_{should_mut}",
            split=sp,
            scenario_type="read_before_write" if should_mut else "negative_condition",
            primary_apps=["pantry"],
            installed_apps=["pantry", "grocery", "recipes"],
            user_request=req_text,
            current_time="2026-11-22T11:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "pantry": {
                    "items": [
                        {"id": pid, "name": pname, "stock": stock_init, "unit": unit_s},
                        {"id": "pan_salt_99", "name": "muối biển", "stock": 2, "unit": "gói"},
                    ]
                }
            },
            objective=f"List pantry items; if stock ({stock_init}) >= {use_q}, consume {use_q}.",
            expect_mutation=should_mut,
            plan_fn=_plan_pan,
        ))

    # Recipes search -> get -> save (4 train + 1 val)
    recipes_cases = [
        ("sim_recipes_tr_01", "train", "thịt kho tàu", 6, "rcp_tkt_01", "Thịt kho tàu trứng vịt",
         "Tìm công thức nấu món thịt kho tàu cho 6 người ăn, xem chi tiết các bước làm rồi lưu công thức đó lại giúp anh."),
        ("sim_recipes_tr_02", "train", "bò sốt vang", 4, "rcp_bsv_02", "Bò sốt vang kiểu Pháp",
         "Tra cứu công thức bò sốt vang khẩu phần 4 người, mở xem chi tiết công thức rồi lưu vào mục yêu thích nhé."),
        ("sim_recipes_tr_03", "train", "gà hấp lá chanh", 5, "rcp_gh_03", "Gà ta hấp lá chanh",
         "Tìm món gà hấp lá chanh cho 5 người ăn rồi bấm lưu công thức lại giúp mình."),
        ("sim_recipes_val_01", "val", "bún chả Hà Nội", 4, "rcp_bc_01", "Bún chả Hà Nội truyền thống",
         "Tìm công thức làm bún chả Hà Nội cho 4 người ăn, xem chi tiết cách làm và lưu lại công thức đó giúp chị."),
    ]
    for idx, (sid, sp, q_str, srv, rid, rtitle, req_text) in enumerate(recipes_cases):
        city, prof, loc = _make_profile(idx)
        need_detail = "xem chi tiết" in req_text
        def _plan_rcp(r: SimulatorRunner, q=q_str, s=srv, detail=need_detail):
            res = r.step("recipes.search", {"query": q, "servings": s})
            rcp = res["recipes"][0]
            if detail:
                r.step("recipes.get", {"recipe_id": rcp["id"]})
            r.step("recipes.save", {"recipe_id": rcp["id"]})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_recipes_save_{sp}_{need_detail}",
            split=sp,
            scenario_type="multi_step_chain" if need_detail else "read_before_write",
            primary_apps=["recipes"],
            installed_apps=["recipes", "pantry", "notes"],
            user_request=req_text,
            current_time="2026-11-22T15:30:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "recipes": {
                    "recipes": [
                        {"id": rid, "name": rtitle, "servings": 2, "steps": ["Sơ chế", "Nấu chín"], "ingredients": [{"name": "thịt", "quantity": 400, "unit": "g"}]}
                    ],
                    "saved": [],
                }
            },
            objective=f"Search recipe {q_str} for {srv} servings, optionally get details, and save {rid}.",
            expect_mutation=True,
            plan_fn=_plan_rcp,
        ))

    # Multi-app chains: recipes -> pantry -> grocery OR shopping_list (6 train + 2 val)
    chain_rpg_cases = [
        ("sim_chain_rpg_tr_01", "train", "lẩu thái hải sản", 4, "rcp_lt_01", "tôm sú", "pan_tom_01", "gro_tom_01", "Tôm sú tươi sống 500g", 165000, "grocery",
         "Tìm công thức nấu lẩu thái hải sản cho 4 người, kiểm tra tủ bếp xem nguyên liệu nào hết hàng (stock=0) thì tìm mua 1 phần trên app đi chợ và đặt giao về nhà."),
        ("sim_chain_rpg_tr_02", "train", "phở bò", 2, "rcp_pb_02", "bánh phở", "pan_bp_02", "gro_bp_02", "Bánh phở tươi gói 500g", 18000, "grocery",
         "Xem công thức nấu phở bò cho 2 người, đối chiếu với tủ bếp xem món nào đang bằng 0 thì đặt mua ngay 1 gói trên ứng dụng đi chợ giao về nhà nhé."),
        ("sim_chain_rpg_tr_04", "train", "gà kho gừng", 4, "rcp_gkg_04", "thịt gà", "pan_ga_04", "", "", 0, "shopping_list",
         "Tìm công thức gà kho gừng cho 4 người, kiểm tra trong tủ bếp xem nguyên liệu nào đã hết (stock=0) thì thêm tên nguyên liệu đó với số lượng 1 vào danh sách mua sắm giúp anh."),
        ("sim_chain_rpg_val_01", "val", "bò lúc lắc", 4, "rcp_bll_01", "thăn bò", "pan_tb_01", "gro_tb_01", "Thăn bò Úc tươi 500g", 195000, "grocery",
         "Tìm công thức bò lúc lắc cho 4 người, kiểm tra tủ bếp xem nguyên liệu nào đã hết (stock=0) thì tìm trên app đi chợ đặt mua 1 phần giao về địa chỉ nhà giúp chị."),
    ]
    for idx, (sid, sp, r_query, srv, rid, miss_ing, pan_id, gro_id, gro_name, gro_pr, target_app, req_text) in enumerate(chain_rpg_cases):
        city, prof, loc = _make_profile(idx)
        if target_app == "grocery":
            def _plan_chain_rpg(r: SimulatorRunner, rq=r_query, s=srv, ad=prof["home_address"]):
                r.step("recipes.search", {"query": rq, "servings": s})
                p_res = r.step("pantry.list_items", {})
                missing = next(it for it in p_res["items"] if it.get("stock", 0) == 0)
                g_res = r.step("grocery.search", {"query": missing["name"]})
                r.step("grocery.add_to_cart", {"product_id": g_res["products"][0]["id"], "quantity": 1})
                r.step("grocery.checkout", {"address": ad})
            inst = ["recipes", "pantry", "grocery", "notes"]
            prim = ["recipes", "pantry", "grocery"]
        else:
            def _plan_chain_rpg(r: SimulatorRunner, rq=r_query, s=srv):
                r.step("recipes.search", {"query": rq, "servings": s})
                p_res = r.step("pantry.list_items", {})
                missing = next(it for it in p_res["items"] if it.get("stock", 0) == 0)
                r.step("shopping_list.add", {"name": missing["name"], "quantity": 1})
            inst = ["recipes", "pantry", "shopping_list", "notes"]
            prim = ["recipes", "pantry", "shopping_list"]

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_chain_recipes_pantry_{target_app}_{sp}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=prim,
            installed_apps=inst,
            user_request=req_text,
            current_time="2026-11-23T09:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "recipes": {
                    "recipes": [
                        {"id": rid, "name": r_query.capitalize(), "servings": srv, "ingredients": [{"name": miss_ing, "quantity": 1, "unit": "phần"}, {"name": "hành tím", "quantity": 3, "unit": "củ"}]}
                    ],
                    "saved": [],
                },
                "pantry": {
                    "items": [
                        {"id": "pan_ok_01", "name": "hành tím", "stock": 10, "unit": "củ"},
                        {"id": pan_id, "name": miss_ing, "stock": 0, "unit": "phần"},
                    ]
                },
                "grocery": {
                    "products": [{"id": gro_id or "gro_x", "name": gro_name or miss_ing, "price": gro_pr or 50000, "unit": "phần", "in_stock": True}],
                    "cart": [],
                    "orders": [],
                },
                "shopping_list": {"items": []},
            },
            objective=f"Chain recipes.search -> pantry.list_items -> {target_app} for missing ingredient '{miss_ing}'.",
            expect_mutation=True,
            plan_fn=_plan_chain_rpg,
        ))

    # =========================================================================
    # FAMILY 6: EVENT_TICKETS & CALENDAR (8 train + 2 val)
    # =========================================================================
    event_cases = [
        ("sim_event_tr_01", "train", "Jazz Mùa Thu", "event_jz_01", "Đêm nhạc Jazz Mùa Thu", "2026-11-28", "20:00", "Nhà hát Lớn", 400000, 2, False,
         "Tìm vé chương trình Đêm nhạc Jazz Mùa Thu rồi đặt giúp anh 2 vé nhé."),
        ("sim_event_tr_02", "train", "Triển lãm Ánh sáng", "event_tl_02", "Triển lãm Ánh sáng Nghệ thuật", "2026-11-29", "09:00", "Bảo tàng Mỹ thuật", 150000, 4, False,
         "Tra cứu sự kiện Triển lãm Ánh sáng Nghệ thuật và đặt mua 4 vé cho nhóm mình nha."),
        ("sim_event_tr_03", "train", "Giao hưởng Mùa Đông", "event_gh_03", "Hòa nhạc Giao hưởng Mùa Đông", "2026-12-02", "19:30", "Học viện Âm nhạc", 500000, 2, True,
         "Tìm đặt 2 vé Hòa nhạc Giao hưởng Mùa Đông rồi thêm lịch sự kiện đó vào ứng dụng Lịch giúp chị."),
        ("sim_event_tr_04", "train", "Hội thảo AI", "event_ai_04", "Hội thảo AI & Dữ liệu lớn", "2026-12-05", "08:30", "Trung tâm Hội nghị", 250000, 1, True,
         "Đặt 1 vé tham dự Hội thảo AI rồi ghi sự kiện đó vào Lịch theo đúng ngày giờ và địa điểm tổ chức nhé."),
        ("sim_event_val_02", "val", "Kịch Thiếu nhi", "event_kt_02", "Vở kịch Thiếu nhi Dế Mèn", "2026-12-01", "18:00", "Nhà hát Tuổi Trẻ", 200000, 3, True,
         "Tìm và đặt 3 vé xem Kịch Thiếu nhi Dế Mèn, sau đó lưu lịch đi xem vào ứng dụng Lịch giúp chị."),
    ]
    for idx, (sid, sp, q_str, evid, evname, evdate, evtime, venue, pr, qty, add_cal, req_text) in enumerate(event_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_ev(r: SimulatorRunner, q=q_str, qt=qty, cal=add_cal):
            res = r.step("event_tickets.search", {"query": q})
            chosen = next(e for e in res["events"] if e.get("available", 0) >= qt)
            r.step("event_tickets.book", {"event_id": chosen["id"], "quantity": qt})
            if cal:
                start_iso = f"{chosen['date']}T{chosen['time']}:00+07:00"
                r.step("calendar.create_event", {"title": chosen["name"], "start": start_iso, "location": chosen.get("venue")})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_event_book_{sp}_{add_cal}",
            split=sp,
            scenario_type="multi_step_chain" if add_cal else "read_before_write",
            primary_apps=["event_tickets", "calendar"] if add_cal else ["event_tickets"],
            installed_apps=["event_tickets", "calendar", "ewallet"],
            user_request=req_text,
            current_time="2026-11-22T10:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "event_tickets": {
                    "events": [
                        {"id": evid, "name": evname, "category": "sự kiện", "date": evdate, "time": evtime, "venue": venue, "price": pr, "available": 30}
                    ],
                    "bookings": [],
                },
                "calendar": {"events": []},
            },
            objective=f"Search event {q_str}, book {qty} tickets on {evid}" + (", and create calendar event." if add_cal else "."),
            expect_mutation=True,
            plan_fn=_plan_ev,
        ))

    # Event tickets: select available event when one is sold out + negative condition when too expensive + ask missing quantity
    city, prof, loc = _make_profile(0)
    def _plan_ev_select(r: SimulatorRunner):
        res = r.step("event_tickets.search", {"query": "Hài kịch"})
        chosen = next(e for e in res["events"] if e.get("available", 0) >= 3)
        r.step("event_tickets.book", {"event_id": chosen["id"], "quantity": 3})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_event_tr_06_select_avail",
        template_id="tpl_event_select_avail_train",
        split="train",
        scenario_type="multi_result_selection",
        primary_apps=["event_tickets"],
        installed_apps=["event_tickets", "calendar"],
        user_request="Tìm các suất diễn Hài kịch cuối tuần này, chọn suất nào còn đủ ít nhất 3 vé trống rồi đặt 3 vé giúp anh.",
        current_time="2026-11-22T11:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "event_tickets": {
                "events": [
                    {"id": "event_hk_01", "name": "Hài kịch Tối Thứ Bảy", "category": "hài kịch", "date": "2026-11-28", "time": "20:00", "venue": "Sân khấu Trống Đồng", "price": 250000, "available": 1},
                    {"id": "event_hk_02", "name": "Hài kịch Tối Chủ Nhật", "category": "hài kịch", "date": "2026-11-29", "time": "20:00", "venue": "Sân khấu Trống Đồng", "price": 250000, "available": 18},
                ],
                "bookings": [],
            }
        },
        objective="Search comedy shows, skip event_hk_01 with only 1 ticket left, book 3 tickets on event_hk_02.",
        expect_mutation=True,
        plan_fn=_plan_ev_select,
    ))

    city, prof, loc = _make_profile(1)
    def _plan_ev_neg_price(r: SimulatorRunner):
        res = r.step("event_tickets.search", {"query": "Rock Storm"})
        ev = res["events"][0]
        if ev.get("price", 0) <= 400000:
            r.step("event_tickets.book", {"event_id": ev["id"], "quantity": 2})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_event_tr_07_neg_price",
        template_id="tpl_event_neg_price_train",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["event_tickets"],
        installed_apps=["event_tickets", "ewallet"],
        user_request="Kiểm tra vé đêm nhạc Rock Storm xem giá vé có dưới 400 nghìn một vé không, nếu dưới thì đặt 2 vé, còn đắt hơn thì thôi không đặt nhé.",
        current_time="2026-11-22T12:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "event_tickets": {
                "events": [
                    {"id": "event_rk_01", "name": "Đêm nhạc Rock Storm 2026", "category": "ca nhạc", "date": "2026-12-10", "time": "19:30", "venue": "SVĐ Phú Thọ", "price": 650000, "available": 50}
                ],
                "bookings": [],
            }
        },
        objective="Check Rock Storm ticket price (650,000 > 400,000) and stop without booking.",
        expect_mutation=False,
        plan_fn=_plan_ev_neg_price,
    ))

    city, prof, loc = _make_profile(2)
    def _plan_ev_ask_qty(r: SimulatorRunner):
        r.step("event_tickets.search", {"query": "Triển lãm Tranh Sơn Mài"})
        r.step("sys.ask_missing_info", {"tool_name": "event_tickets.book", "params": ["quantity"]})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_event_tr_08_ask_qty",
        template_id="tpl_event_ask_qty_train",
        split="train",
        scenario_type="missing_info",
        primary_apps=["event_tickets", "sys"],
        installed_apps=["event_tickets", "calendar"],
        user_request="Tìm sự kiện Triển lãm Tranh Sơn Mài rồi đặt vé giúp mình với.",
        current_time="2026-11-22T13:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "event_tickets": {
                "events": [
                    {"id": "event_sm_01", "name": "Triển lãm Tranh Sơn Mài Đương Đại", "category": "triển lãm", "date": "2026-12-03", "time": "09:00", "venue": "Trung tâm Nghệ thuật", "price": 120000, "available": 40}
                ],
                "bookings": [],
            }
        },
        objective="Search event and ask user for missing required parameter quantity.",
        expect_mutation=False,
        plan_fn=_plan_ev_ask_qty,
    ))

    # =========================================================================
    # FAMILY 7: PUBLIC_TRANSIT (6 train + 2 val)
    # =========================================================================
    transit_cases = [
        ("sim_transit_tr_01", "train", "Bến Thành", "Suối Tiên", "Bến Thành", 50000, True,
         "Tìm tuyến metro từ Bến Thành đi Suối Tiên, kiểm tra giờ tàu sắp chạy tại trạm Bến Thành rồi nạp thêm 50 nghìn vào thẻ đi xe giúp anh."),
        ("sim_transit_tr_02", "train", "Cát Linh", "Hà Đông", "Cát Linh", 100000, True,
         "Tra cứu tuyến tàu điện từ Cát Linh đi Hà Đông, xem các chuyến sắp khởi hành ở ga Cát Linh và nạp 100 nghìn đồng vào thẻ giao thông nhé."),
        ("sim_transit_tr_03", "train", "Mỹ Đình", "Giáp Bát", "", 50000, False,
         "Tìm tuyến xe buýt từ Mỹ Đình đi Giáp Bát rồi nạp 50 nghìn vào thẻ vé xe buýt giúp mình."),
        ("sim_transit_val_01", "val", "Yên Nghĩa", "Thái Hà", "Yên Nghĩa", 100000, True,
         "Tìm tuyến tàu điện từ Yên Nghĩa đi Thái Hà, xem giờ tàu sắp chạy tại trạm Yên Nghĩa rồi nạp 100 nghìn vào thẻ vé giúp chị."),
    ]
    for idx, (sid, sp, orig, dest, stop_name, topup_amt, check_dep, req_text) in enumerate(transit_cases):
        city, prof, loc = _make_profile(idx)
        rid = f"rt_tr_{idx+1:02d}"
        def _plan_tr(r: SimulatorRunner, o=orig, d=dest, st=stop_name, amt=topup_amt, cd=check_dep):
            res = r.step("public_transit.find_routes", {"origin": o, "destination": d})
            if cd and st:
                rt_id = res["routes"][0]["route_id"]
                r.step("public_transit.next_departures", {"stop": st, "route": rt_id})
            r.step("public_transit.topup_card", {"amount": amt})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_transit_{sp}_{check_dep}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=["public_transit"],
            installed_apps=["public_transit", "maps", "ewallet"],
            user_request=req_text,
            current_time="2026-11-22T08:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "public_transit": {
                    "card_balance": 15000,
                    "routes": [
                        {"id": rid, "name": f"Tuyến {orig} - {dest}", "type": "metro" if check_dep else "bus", "stops": [orig, "Trạm Trung Gian", dest], "fare": 10000, "duration_min": 25, "headway_min": 8}
                    ],
                    "departures": [
                        {"stop": orig, "route": rid, "time": "08:10"},
                        {"stop": orig, "route": rid, "time": "08:18"},
                    ],
                    "topups": [],
                }
            },
            objective=f"Find transit routes from {orig} to {dest}, optionally check departures, and topup {topup_amt}.",
            expect_mutation=True,
            plan_fn=_plan_tr,
        ))

    # Transit read-only departures + ask_missing_info for topup amount (2 train)
    city, prof, loc = _make_profile(0)
    scenarios.append(ScenarioSpec(
        scenario_id="sim_transit_tr_05_read_only",
        template_id="tpl_transit_readonly_train",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["public_transit"],
        installed_apps=["public_transit", "maps"],
        user_request="Kiểm tra giúp mình tuyến xe buýt từ Long Biên đi Gia Lâm và xem các chuyến sắp khởi hành tại trạm Long Biên nhé, chỉ xem thôi chưa cần nạp thẻ.",
        current_time="2026-11-22T08:30:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "public_transit": {
                "card_balance": 45000,
                "routes": [{"id": "bus_lb_01", "name": "Tuyến Long Biên - Gia Lâm", "type": "bus", "stops": ["Long Biên", "Nguyễn Văn Cừ", "Gia Lâm"], "fare": 7000, "duration_min": 20, "headway_min": 12}],
                "departures": [{"stop": "Long Biên", "route": "bus_lb_01", "time": "08:35"}],
                "topups": [],
            }
        },
        objective="Find routes and check next departures without mutating state.",
        expect_mutation=False,
        plan_fn=lambda r: (
            r.step("public_transit.find_routes", {"origin": "Long Biên", "destination": "Gia Lâm"}),
            r.step("public_transit.next_departures", {"stop": "Long Biên", "route": "bus_lb_01"}),
        ),
    ))

    scenarios.append(ScenarioSpec(
        scenario_id="sim_transit_tr_06_ask_amount",
        template_id="tpl_transit_ask_amount_train",
        split="train",
        scenario_type="missing_info",
        primary_apps=["public_transit", "sys"],
        installed_apps=["public_transit", "ewallet"],
        user_request="Nạp thêm tiền vào thẻ xe buýt công cộng giúp anh với.",
        current_time="2026-11-22T09:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={"public_transit": {"card_balance": 5000, "routes": [], "departures": [], "topups": []}},
        objective="Ask user for missing required parameter amount for public_transit.topup_card.",
        expect_mutation=False,
        plan_fn=lambda r: r.step("sys.ask_missing_info", {"tool_name": "public_transit.topup_card", "params": ["amount"]}),
    ))

    # =========================================================================
    # FAMILY 8: TRANSLATOR + MULTI-APP MESSAGING / EMAIL / NOTES (8 train + 2 val)
    # =========================================================================
    trans_cases = [
        ("sim_trans_tr_01_chat", "train", "Cuộc họp sáng mai sẽ bắt đầu lúc 9 giờ tại phòng họp lớn", "en",
         "Tomorrow morning's meeting will start at 9 AM in the main conference room", "chat_ott", "Minh",
         "Dịch câu 'Cuộc họp sáng mai sẽ bắt đầu lúc 9 giờ tại phòng họp lớn' sang tiếng Anh rồi gửi qua ứng dụng chat cho Minh giúp chị."),
        ("sim_trans_tr_03_sms", "train", "Tôi đang đợi ở sảnh tòa nhà, bạn xuống nhé", "en",
         "I am waiting in the building lobby, please come down", "sms", "David",
         "Dịch câu 'Tôi đang đợi ở sảnh tòa nhà, bạn xuống nhé' sang tiếng Anh rồi gửi tin nhắn SMS cho David trong danh bạ giúp anh."),
        ("sim_trans_tr_04_sms", "train", "Chuyến bay của tôi hạ cánh lúc 6 giờ chiều nay", "ja",
         "私のフライトは今日の午後6時に着陸します", "sms", "Kenji",
         "Dịch câu 'Chuyến bay của tôi hạ cánh lúc 6 giờ chiều nay' sang tiếng Nhật (ja) rồi nhắn SMS cho đối tác Kenji trong danh bạ nhé."),
        ("sim_trans_tr_05_email", "train", "Xin cảm ơn quý đối tác đã tham dự hội nghị thường niên của chúng tôi", "en",
         "Thank you for attending our annual conference", "email", "partner@globalcorp.com",
         "Dịch câu 'Xin cảm ơn quý đối tác đã tham dự hội nghị thường niên của chúng tôi' sang tiếng Anh rồi gửi email tới địa chỉ partner@globalcorp.com với tiêu đề 'Thank You'."),
        ("sim_trans_tr_06_notes", "train", "Không sử dụng đường tinh luyện và thực phẩm chiên rán", "en",
         "Do not use refined sugar and fried foods", "notes", "Chế độ ăn tiếng Anh",
         "Dịch câu 'Không sử dụng đường tinh luyện và thực phẩm chiên rán' sang tiếng Anh rồi tạo ghi chú mới tiêu đề 'Chế độ ăn tiếng Anh' lưu nội dung vừa dịch lại."),
        ("sim_trans_val_01_chat", "val", "Đơn hàng của quý khách đã được xuất kho sáng nay", "en",
         "Your order was dispatched from our warehouse this morning", "chat_ott", "Alex",
         "Dịch câu 'Đơn hàng của quý khách đã được xuất kho sáng nay' sang tiếng Anh rồi gửi tin nhắn qua ứng dụng chat cho Alex giúp mình."),
    ]
    for idx, (sid, sp, src_txt, lang, trans_txt, dest_app, recipient, req_text) in enumerate(trans_cases):
        city, prof, loc = _make_profile(idx)
        if dest_app == "chat_ott":
            def _plan_tr_msg(r: SimulatorRunner, txt=src_txt, lg=lang, rec=recipient):
                t_res = r.step("translator.translate", {"text": txt, "to_lang": lg})
                c_res = r.step("chat_ott.list_chats", {})
                target = next(c for c in c_res["chats"] if rec.lower() in c.get("name", "").lower())
                r.step("chat_ott.send_message", {"chat_id": target["id"], "content": t_res["translation"]})
            inst = ["translator", "chat_ott", "contacts"]
            prim = ["translator", "chat_ott"]
        elif dest_app == "sms":
            def _plan_tr_msg(r: SimulatorRunner, txt=src_txt, lg=lang, rec=recipient):
                t_res = r.step("translator.translate", {"text": txt, "to_lang": lg})
                ct_res = r.step("contacts.search", {"query": rec})
                contact = ct_res["contacts"][0]
                r.step("sms.send", {"contact_id": contact["id"], "phone_number": contact["phone"], "content": t_res["translation"]})
            inst = ["translator", "sms", "contacts"]
            prim = ["translator", "contacts", "sms"]
        elif dest_app == "email":
            def _plan_tr_msg(r: SimulatorRunner, txt=src_txt, lg=lang, rec=recipient):
                t_res = r.step("translator.translate", {"text": txt, "to_lang": lg})
                r.step("email.send", {"to": [rec], "subject": "Thank You", "body": t_res["translation"]})
            inst = ["translator", "email"]
            prim = ["translator", "email"]
        else:
            def _plan_tr_msg(r: SimulatorRunner, txt=src_txt, lg=lang, ttl=recipient):
                t_res = r.step("translator.translate", {"text": txt, "to_lang": lg})
                r.step("notes.create", {"title": ttl, "content": t_res["translation"]})
            inst = ["translator", "notes"]
            prim = ["translator", "notes"]

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_translator_{dest_app}_{sp}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=prim,
            installed_apps=inst,
            user_request=req_text,
            current_time="2026-11-23T09:30:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "translator": {"dictionary": [{"text": src_txt, "to_lang": lang, "translation": trans_txt}]},
                "chat_ott": {"chats": [{"id": "chat_other_01", "name": "Nhóm Gia Đình", "type": "group", "messages": []}, {"id": "chat_target_02", "name": recipient, "type": "direct", "messages": []}]},
                "contacts": {"contacts": [{"id": "ct_target_01", "name": recipient, "phone": "0909123456", "relation": "đối tác"}]},
                "sms": {"messages": []},
                "email": {"emails": [], "sent": [], "drafts": []},
                "notes": {"notes": []},
            },
            objective=f"Translate text to {lang} and forward result to {dest_app}.",
            expect_mutation=True,
            plan_fn=_plan_tr_msg,
        ))

    # =========================================================================
    # FAMILY 9: HABIT_TRACKER (6 train + 2 val)
    # =========================================================================
    habit_cases = [
        ("sim_habit_tr_01", "train", "habit_run_01", "Chạy bộ 5km", "chạy bộ", "2026-11-22",
         "Ghi nhận hôm nay 2026-11-22 mình đã hoàn thành thói quen chạy bộ 5km rồi nhé."),
        ("sim_habit_tr_02", "train", "habit_med_02", "Thiền định 15 phút", "thiền định", "2026-11-22",
         "Đánh dấu hoàn thành mục tiêu thiền định 15 phút cho ngày hôm nay giúp chị."),
        ("sim_habit_tr_03", "train", "habit_eng_03", "Học 20 từ vựng tiếng Anh", "tiếng anh", "2026-11-22",
         "Cập nhật nhật ký thói quen: hôm nay anh đã học xong 20 từ vựng tiếng Anh rồi nhé."),
        ("sim_habit_tr_05", "train", "habit_slp_05", "Đi ngủ trước 23h", "đi ngủ", "2026-11-21",
         "Ghi bù giúp anh thói quen đi ngủ trước 23h cho ngày hôm qua 2026-11-21 nhé."),
        ("sim_habit_val_01", "val", "habit_yg_01", "Tập Yoga 30 phút", "yoga", "2026-11-22",
         "Đánh dấu hoàn thành thói quen tập Yoga 30 phút cho ngày hôm nay 2026-11-22 giúp chị nhé."),
        ("sim_habit_val_02", "val", "habit_wt_02", "Uống đủ 2 lít nước lọc", "nước lọc", "2026-11-22",
         "Ghi nhận mình đã hoàn thành thói quen uống đủ 2 lít nước lọc trong ngày hôm nay nhé."),
    ]
    for idx, (sid, sp, hid, hname, kw, log_dt, req_text) in enumerate(habit_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_hb(r: SimulatorRunner, k=kw, dt=log_dt):
            res = r.step("habit_tracker.list_habits", {})
            target = next(h for h in res["habits"] if k in h["name"].lower())
            r.step("habit_tracker.log", {"habit_id": target["id"], "date": dt})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_habit_log_{sp}",
            split=sp,
            scenario_type="read_before_write",
            primary_apps=["habit_tracker"],
            installed_apps=["habit_tracker", "reminders", "calendar"],
            user_request=req_text,
            current_time="2026-11-22T20:30:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "habit_tracker": {
                    "habits": [
                        {"id": "habit_dummy_00", "name": "Đạp xe buổi sáng", "streak": 2, "last_logged": "2026-11-20"},
                        {"id": hid, "name": hname, "streak": 5, "last_logged": "2026-11-20" if log_dt == "2026-11-21" else "2026-11-21"},
                    ],
                    "logs": [],
                }
            },
            objective=f"List habits, select {hid}, and log completion on {log_dt}.",
            expect_mutation=True,
            plan_fn=_plan_hb,
        ))

    # Habit already logged today -> negative condition (stop without calling log again)
    city, prof, loc = _make_profile(0)
    def _plan_hb_already(r: SimulatorRunner):
        res = r.step("habit_tracker.list_habits", {})
        target = next(h for h in res["habits"] if "chạy bộ" in h["name"].lower())
        if target.get("last_logged") != "2026-11-22":
            r.step("habit_tracker.log", {"habit_id": target["id"], "date": "2026-11-22"})

    scenarios.append(ScenarioSpec(
        scenario_id="sim_habit_tr_06_already_logged",
        template_id="tpl_habit_already_logged_train",
        split="train",
        scenario_type="negative_condition",
        primary_apps=["habit_tracker"],
        installed_apps=["habit_tracker", "reminders"],
        user_request="Kiểm tra xem hôm nay 2026-11-22 mình đã ghi nhận thói quen chạy bộ buổi sáng chưa, nếu chưa ghi thì đánh dấu hoàn thành giúp mình, ghi rồi thì thôi.",
        current_time="2026-11-22T21:00:00+07:00",
        city=city,
        user_profile=prof,
        location=loc,
        initial_app_states={
            "habit_tracker": {
                "habits": [
                    {"id": "habit_run_99", "name": "Chạy bộ buổi sáng", "streak": 8, "last_logged": "2026-11-22"}
                ],
                "logs": [{"id": "log_01", "habit_id": "habit_run_99", "date": "2026-11-22"}],
            }
        },
        objective="Check habits; last_logged is already 2026-11-22, so stop without calling habit_tracker.log.",
        expect_mutation=False,
        plan_fn=_plan_hb_already,
    ))

    # =========================================================================
    # FAMILY 10: HIGH-ERROR COMMON APPS (BILL_PAYMENT, SMART_HOME, FOOD_DELIVERY,
    #            FLIGHTS, BANKING, CALENDAR, WEATHER, SELF-CANCELLATIONS) (28 train + 6 val)
    # =========================================================================
    # 10.1 Bill payment arithmetic: wallet sufficient vs wallet insufficient -> bank vs wallet insufficient -> stop
    bill_cases = [
        ("sim_bill_tr_01_wallet_ok", "train", "electricity", "bill_el_01", 320000, 500000, "ewallet", True,
         "Kiểm tra hóa đơn tiền điện tháng này và số dư ví điện tử, nếu ví đủ tiền thì thanh toán bằng ví điện tử nhé."),
        ("sim_bill_tr_02_fallback_bank", "train", "internet", "bill_in_02", 280000, 150000, "bank", True,
         "Xem hóa đơn tiền mạng internet và số dư ví điện tử, nếu ví đủ thì trả bằng ví, còn ví không đủ tiền thì thanh toán qua ngân hàng giúp anh."),
        ("sim_bill_tr_03_insufficient_stop", "train", "electricity", "bill_el_03", 780000, 210000, "none", False,
         "Kiểm tra hóa đơn tiền điện và số dư ví điện tử giúp chị, nếu số dư ví đủ tiền thì mới thanh toán, còn không đủ thì thôi để mai chị nạp thêm tiền sau."),
        ("sim_bill_val_01_fallback_bank", "val", "electricity", "bill_el_v1", 640000, 300000, "bank", True,
         "Kiểm tra hóa đơn tiền điện kỳ này cùng số dư ví điện tử, ưu tiên trả bằng ví nhưng nếu ví thiếu tiền thì chuyển sang trả bằng tài khoản ngân hàng nhé."),
        ("sim_bill_val_02_insufficient_stop", "val", "water", "bill_wt_v2", 310000, 95000, "none", False,
         "Xem hóa đơn tiền nước hết bao nhiêu rồi kiểm tra ví điện tử, chỉ thanh toán khi ví đủ số dư, nếu thiếu thì dừng lại không thanh toán nhé."),
    ]
    for idx, (sid, sp, btype, bid, b_amt, w_bal, exp_src, should_mut, req_text) in enumerate(bill_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_bill(r: SimulatorRunner, bt=btype, src=exp_src):
            b_res = r.step("bill_payment.list_bills", {"type": bt})
            r.step("ewallet.get_balance", {})
            if src in ("ewallet", "bank"):
                r.step("bill_payment.pay", {"bill_id": b_res["bills"][0]["id"], "source": src})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_bill_arith_{sp}_{exp_src}",
            split=sp,
            scenario_type="arithmetic_condition" if should_mut else "negative_condition",
            primary_apps=["bill_payment", "ewallet"],
            installed_apps=["bill_payment", "ewallet", "banking", "sms"],
            user_request=req_text,
            current_time="2026-11-23T09:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "bill_payment": {"bills": [{"id": bid, "type": btype, "amount": b_amt, "paid": False}], "autopays": []},
                "ewallet": {"balance": w_bal, "transactions": []},
            },
            objective=f"Check {btype} bill ({b_amt}) and wallet balance ({w_bal}) -> action={exp_src}.",
            expect_mutation=should_mut,
            plan_fn=_plan_bill,
        ))

    # 10.2 Food delivery: list_restaurants -> get_menu -> place_order (with nested item_id)
    food_cases = [
        ("sim_food_tr_01", "train", "Phở Thìn", "rest_pt_01", "m_pho_01", "Phở bò tái lăn", 65000, 2, "ewallet",
         "Tìm quán Phở Thìn trên ứng dụng gọi đồ ăn, xem thực đơn rồi đặt 2 bát Phở bò tái lăn giao về nhà, thanh toán qua ví điện tử nhé."),
        ("sim_food_tr_02", "train", "Cơm Tấm Sài Gòn", "rest_ct_02", "m_com_02", "Cơm tấm sườn bì chả", 55000, 3, "cash",
         "Mở app đặt đồ ăn tìm quán Cơm Tấm Sài Gòn, lấy thực đơn của quán rồi đặt 3 suất Cơm tấm sườn bì chả giao đến địa chỉ nhà trả tiền mặt."),
        ("sim_food_tr_03", "train", "Bún Chả Hương Liên", "rest_bc_03", "m_bun_03", "Bún chả đặc biệt", 60000, 2, "ewallet",
         "Tìm nhà hàng Bún Chả Hương Liên, tra cứu menu rồi đặt 2 phần Bún chả đặc biệt giao về nhà thanh toán bằng ví điện tử."),
        ("sim_food_val_01", "val", "Mì Quảng Bà Mua", "rest_mq_01", "m_mi_01", "Mì Quảng tôm thịt", 50000, 2, "cash",
         "Tìm quán Mì Quảng Bà Mua trên ứng dụng đặt món, xem thực đơn và đặt 2 tô Mì Quảng tôm thịt giao về nhà thanh toán tiền mặt nhé."),
    ]
    for idx, (sid, sp, rname, rid, mid, mname, mprice, qty, pay_m, req_text) in enumerate(food_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_fd(r: SimulatorRunner, target_r=rname, target_m=mname, qt=qty, ad=prof["home_address"], pm=pay_m):
            r_res = r.step("food_delivery.list_restaurants", {})
            rest = next(x for x in r_res["restaurants"] if target_r.lower() in x["name"].lower())
            m_res = r.step("food_delivery.get_menu", {"restaurant_id": rest["id"]})
            dish = next(x for x in m_res["menu"] if target_m.lower() in x["name"].lower())
            r.step("food_delivery.place_order", {
                "restaurant_id": rest["id"],
                "items": [{"item_id": dish["id"], "quantity": qt}],
                "address": ad,
                "payment": pm,
            })

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_food_order_{sp}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=["food_delivery"],
            installed_apps=["food_delivery", "ewallet", "grocery"],
            user_request=req_text,
            current_time="2026-11-23T11:30:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "food_delivery": {
                    "restaurants": [
                        {"id": "rest_other_00", "name": "Trà Sữa Tocotoco", "rating": 4.4, "menu": [{"id": "m_ts_00", "name": "Trà sữa trân châu", "price": 35000}]},
                        {"id": rid, "name": rname, "rating": 4.8, "menu": [{"id": mid, "name": mname, "price": mprice}]},
                    ],
                    "orders": [],
                }
            },
            objective=f"List restaurants, get menu for {rid}, and place order for {qty}x {mid}.",
            expect_mutation=True,
            plan_fn=_plan_fd,
        ))

    # 10.3 Flights: search -> select cheapest / earliest -> book -> check_in
    flights_cases = [
        ("sim_flights_tr_01", "train", "HAN", "SGN", "2026-11-28", "fl_vn_02", "Vietnam Airlines", "08:00", 1450000, 1, "economy",
         "Tìm chuyến bay từ Hà Nội đi TP.HCM ngày 2026-11-28, chọn chuyến có giá vé rẻ nhất rồi đặt 1 vé hạng phổ thông (economy) giúp anh."),
        ("sim_flights_tr_02", "train", "SGN", "DAD", "2026-11-29", "fl_vj_02", "Vietjet Air", "07:15", 890000, 2, "economy",
         "Tra cứu vé máy bay từ TP.HCM đi Đà Nẵng ngày 2026-11-29 và đặt 2 vé hạng phổ thông ở chuyến có giá thấp nhất nhé."),
        ("sim_flights_tr_03_checkin", "train", "HAN", "CXR", "2026-11-24", "fl_qh_01", "Bamboo Airways", "09:00", 1250000, 1, "economy",
         "Kiểm tra danh sách vé máy bay đã đặt của mình rồi làm thủ tục check-in online cho chuyến bay đi Nha Trang nhé."),
        ("sim_flights_val_01", "val", "HAN", "DLI", "2026-11-30", "fl_vn_v2", "Vietnam Airlines", "08:30", 1150000, 2, "economy",
         "Tìm chuyến bay từ Hà Nội đi Đà Lạt ngày 2026-11-30, chọn chuyến giá rẻ nhất rồi đặt 2 vé hạng phổ thông giúp chị."),
    ]
    for idx, (sid, sp, orig, dest, dt, fid, airline, dep, pr, pax, sclass, req_text) in enumerate(flights_cases):
        city, prof, loc = _make_profile(idx)
        if "check-in" in req_text:
            def _plan_fl(r: SimulatorRunner):
                b_res = r.step("flights.my_bookings", {})
                bk = next(b for b in b_res["bookings"] if b.get("status") == "confirmed" and not b.get("checked_in"))
                r.step("flights.check_in", {"booking_id": bk["id"]})
            fl_state = {
                "flights": [{"id": fid, "origin": orig, "destination": dest, "date": dt, "depart": dep, "arrive": "11:00", "airline": airline, "price": pr, "flight_no": "QH101"}],
                "bookings": [{"id": "bk_ci_01", "flight_id": fid, "passengers": 1, "seat_class": "economy", "status": "confirmed"}],
            }
        else:
            def _plan_fl(r: SimulatorRunner, o=orig, d=dest, date_s=dt, p=pax, sc=sclass):
                res = r.step("flights.search", {"origin": o, "destination": d, "date": date_s})
                cheapest = min(res["flights"], key=lambda x: x.get("price", 999999999))
                r.step("flights.book", {"flight_id": cheapest["id"], "passengers": p, "seat_class": sc})
            fl_state = {
                "flights": [
                    {"id": f"fl_exp_{idx:02d}", "origin": orig, "destination": dest, "date": dt, "depart": "13:00", "arrive": "15:00", "airline": "Bamboo Airways", "price": pr + 450000, "flight_no": "QH202"},
                    {"id": fid, "origin": orig, "destination": dest, "date": dt, "depart": dep, "arrive": "10:10", "airline": airline, "price": pr, "flight_no": "VN123"},
                ],
                "bookings": [],
            }

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_flights_{sp}_{'checkin' if 'check-in' in req_text else 'book'}",
            split=sp,
            scenario_type="multi_result_selection" if "check-in" not in req_text else "read_before_write",
            primary_apps=["flights"],
            installed_apps=["flights", "calendar", "ewallet"],
            user_request=req_text,
            current_time="2026-11-23T08:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={"flights": fl_state},
            objective=f"Execute flight workflow on {fid}.",
            expect_mutation=True,
            plan_fn=_plan_fl,
        ))

    # 10.4 Banking: get_balance + list_beneficiaries -> transfer (3 train + 1 val)
    banking_cases = [
        ("sim_bank_tr_01", "train", "acc_main_01", 15000000, "ben_mom_01", "Mẹ", 3000000, "Biếu mẹ tiền tiêu vặt",
         "Kiểm tra tài khoản ngân hàng và danh sách người thụ hưởng rồi chuyển khoản cho Mẹ 3 triệu đồng với nội dung 'Biếu mẹ tiền tiêu vặt' nhé."),
        ("sim_bank_tr_02", "train", "acc_main_02", 25000000, "ben_land_02", "Chú Hòa Chủ Nhà", 6500000, "Tiền thuê nhà tháng 11",
         "Tra cứu tài khoản ngân hàng và người thụ hưởng, sau đó chuyển 6 triệu 500 nghìn đồng cho Chú Hòa Chủ Nhà kèm lời nhắn 'Tiền thuê nhà tháng 11'."),
        ("sim_bank_tr_03_savings", "train", "acc_main_03", 50000000, "", "", 20000000, "",
         "Kiểm tra số dư tài khoản ngân hàng rồi mở sổ tiết kiệm kỳ hạn 6 tháng với số tiền 20 triệu đồng giúp anh."),
        ("sim_bank_val_01", "val", "acc_main_v1", 18000000, "ben_sis_01", "Em Trang", 2000000, "Chúc mừng sinh nhật em",
         "Xem tài khoản ngân hàng và danh sách thụ hưởng rồi chuyển cho Em Trang 2 triệu đồng với nội dung 'Chúc mừng sinh nhật em' giúp chị."),
    ]
    for idx, (sid, sp, acc_id, bal, ben_id, ben_name, amt, note_s, req_text) in enumerate(banking_cases):
        city, prof, loc = _make_profile(idx)
        if "tiết kiệm" in req_text:
            def _plan_bk(r: SimulatorRunner, a=amt):
                acc_res = r.step("banking.get_balance", {})
                acc = acc_res["accounts"][0]
                r.step("banking.open_savings", {"from_account_id": acc["id"], "amount": a, "term_months": 6})
        else:
            def _plan_bk(r: SimulatorRunner, bname=ben_name, a=amt, nt=note_s):
                acc_res = r.step("banking.get_balance", {})
                ben_res = r.step("banking.list_beneficiaries", {})
                acc = acc_res["accounts"][0]
                ben = next(b for b in ben_res["beneficiaries"] if bname.lower() in b["name"].lower())
                r.step("banking.transfer", {"from_account_id": acc["id"], "beneficiary_id": ben["id"], "amount": a, "note": nt})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_banking_{sp}_{'savings' if 'tiết kiệm' in req_text else 'transfer'}",
            split=sp,
            scenario_type="multi_step_chain",
            primary_apps=["banking"],
            installed_apps=["banking", "ewallet", "sms"],
            user_request=req_text,
            current_time="2026-11-23T10:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "banking": {
                    "accounts": [{"id": acc_id, "name": "Tài khoản Thanh toán", "balance": bal}],
                    "beneficiaries": [{"id": ben_id or "ben_00", "name": ben_name or "Người nhận", "bank": "VCB", "account_number": "0011223344"}],
                    "transactions": [],
                    "savings": [],
                }
            },
            objective=f"Read banking account/beneficiary IDs and execute transfer/savings of {amt}.",
            expect_mutation=True,
            plan_fn=_plan_bk,
        ))

    # 10.5 Smart Home device type verification + scenes (4 train + 1 val)
    sh_cases = [
        ("sim_sh_tr_01_light", "train", "light", "phòng làm việc", "dev_light_work",
         "Kiểm tra trạng thái thiết bị trong phòng làm việc rồi bật đèn phòng làm việc lên độ sáng 80 giúp anh."),
        ("sim_sh_tr_02_climate", "train", "climate", "phòng ngủ", "dev_ac_bed",
         "Xem các thiết bị ở phòng ngủ rồi chỉnh điều hòa phòng ngủ sang chế độ làm mát (cool) ở mức 26 độ nhé."),
        ("sim_sh_tr_03_scene", "train", "scene", "", "sc_sleep_01",
         "Kiểm tra danh sách thiết bị và ngữ cảnh nhà thông minh rồi kích hoạt ngữ cảnh 'Đi ngủ' giúp mình."),
        ("sim_sh_tr_04_multi_fan", "train", "multi_fan", "", "dev_fan_living",
         "Bật hết quạt trong nhà lên số 3 giúp anh, phòng nào điều hòa đang chạy thì thôi không cần bật quạt."),
        ("sim_sh_val_01_scene", "val", "scene", "", "sc_cinema_02",
         "Lấy trạng thái nhà thông minh để xem danh sách ngữ cảnh đã lưu rồi chạy ngữ cảnh 'Xem phim' giúp chị nhé."),
    ]
    for idx, (sid, sp, mode_s, room_s, target_id, req_text) in enumerate(sh_cases):
        city, prof, loc = _make_profile(idx)
        if mode_s == "light":
            def _plan_sh(r: SimulatorRunner, rm=room_s):
                res = r.step("smart_home.get_status", {"room": rm})
                dev = next(d for d in res["devices"] if d.get("type") == "light")
                r.step("smart_home.light_set", {"device_id": dev["id"], "power": "on", "brightness": 80})
        elif mode_s == "climate":
            def _plan_sh(r: SimulatorRunner, rm=room_s):
                res = r.step("smart_home.get_status", {"room": rm})
                dev = next(d for d in res["devices"] if d.get("type") == "climate")
                r.step("smart_home.climate_set", {"device_id": dev["id"], "mode": "cool", "temperature": 26})
        elif mode_s == "scene":
            def _plan_sh(r: SimulatorRunner, tid=target_id):
                res = r.step("smart_home.get_status", {})
                sc = next(s for s in res.get("scenes", []) if s["id"] == tid)
                r.step("smart_home.scene_run", {"scene_id": sc["id"]})
        else:
            def _plan_sh(r: SimulatorRunner):
                res = r.step("smart_home.get_status", {})
                ac_rooms = {d.get("room") for d in res["devices"] if d.get("type") == "climate" and (d.get("state") or {}).get("mode") not in (None, "off")}
                for d in res["devices"]:
                    if d.get("type") == "fan" and d.get("room") not in ac_rooms:
                        r.step("smart_home.fan_set", {"device_id": d["id"], "power": "on", "speed": 3})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_smarthome_{mode_s}_{sp}",
            split=sp,
            scenario_type="read_before_write" if mode_s != "multi_fan" else "multi_result_selection",
            primary_apps=["smart_home"],
            installed_apps=["smart_home", "alarm", "tv"],
            user_request=req_text,
            current_time="2026-11-23T20:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "smart_home": {
                    "devices": [
                        {"id": "dev_light_work", "name": "Đèn bàn làm việc", "room": "phòng làm việc", "type": "light", "state": {"power": "off", "brightness": 50}},
                        {"id": "dev_fan_work", "name": "Quạt phòng làm việc", "room": "phòng làm việc", "type": "fan", "state": {"power": "off", "speed": 1}},
                        {"id": "dev_ac_bed", "name": "Điều hòa phòng ngủ", "room": "phòng ngủ", "type": "climate", "state": {"mode": "cool", "temperature": 27}},
                        {"id": "dev_fan_bed", "name": "Quạt phòng ngủ", "room": "phòng ngủ", "type": "fan", "state": {"power": "off", "speed": 1}},
                        {"id": "dev_fan_living", "name": "Quạt phòng khách", "room": "phòng khách", "type": "fan", "state": {"power": "off", "speed": 1}},
                        {"id": "dev_ac_living", "name": "Điều hòa phòng khách", "room": "phòng khách", "type": "climate", "state": {"mode": "off", "temperature": 26}},
                    ],
                    "scenes": [
                        {"id": "sc_sleep_01", "name": "Đi ngủ", "actions": [{"device_id": "dev_light_work", "state": {"power": "off"}}]},
                        {"id": "sc_cinema_02", "name": "Xem phim", "actions": [{"device_id": "dev_light_work", "state": {"power": "off"}}]},
                    ],
                }
            },
            objective=f"Inspect smart_home status and execute {mode_s} action on {target_id}.",
            expect_mutation=True,
            plan_fn=_plan_sh,
        ))

    # 10.6 Weather + Calendar conditional (rain=True vs rain=False) & Self-cancellations (11 train + 1 val)
    weather_cond_cases = [
        ("sim_weather_tr_01_rain_resched", "train", True, "2026-11-24", "evt_run_01", "Chạy bộ công viên", "2026-11-25T06:30:00+07:00",
         "Kiểm tra thời tiết ngày mai 2026-11-24, nếu trời mưa thì dời lịch 'Chạy bộ công viên' sang 06:30 sáng ngày kia 2026-11-25 giúp anh."),
        ("sim_weather_tr_02_norain_stop", "train", False, "2026-11-24", "evt_ten_02", "Đánh cầu lông ngoài trời", "2026-11-25T17:00:00+07:00",
         "Xem dự báo thời tiết ngày mai 2026-11-24, nếu có mưa thì lùi lịch 'Đánh cầu lông ngoài trời' sang 17:00 ngày kia, còn không mưa thì giữ nguyên nhé."),
        ("sim_weather_tr_03_rain_cancel", "train", True, "2026-11-24", "evt_pic_03", "Dã ngoại cuối tuần", "",
         "Kiểm tra thời tiết ngày mai 2026-11-24 xem có mưa không, nếu mưa thì hủy sự kiện 'Dã ngoại cuối tuần' trong lịch giúp chị."),
    ]
    for idx, (sid, sp, is_rain, f_date, ev_id, ev_title, new_start, req_text) in enumerate(weather_cond_cases):
        city, prof, loc = _make_profile(idx)
        def _plan_w_cal(r: SimulatorRunner, dt=f_date, ns=new_start):
            f_res = r.step("weather.get_forecast", {"date": dt})
            if f_res["forecast"] and f_res["forecast"][0].get("rain"):
                ev_res = r.step("calendar.list_events", {"date_from": dt, "date_to": dt})
                ev = ev_res["events"][0]
                if ns:
                    r.step("calendar.update_event", {"event_id": ev["id"], "start": ns})
                else:
                    r.step("calendar.cancel_event", {"event_id": ev["id"]})

        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_weather_cond_{sp}_{is_rain}",
            split=sp,
            scenario_type="multi_step_chain" if is_rain else "negative_condition",
            primary_apps=["weather", "calendar"],
            installed_apps=["weather", "calendar", "reminders"],
            user_request=req_text,
            current_time="2026-11-23T18:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={
                "weather": {
                    "current": [{"location": city, "condition": "mát mẻ", "temp": 25, "rain": False}],
                    "forecast": [{"location": city, "date": f_date, "condition": "mưa rào" if is_rain else "nắng ráo", "temp_min": 22, "temp_max": 29, "rain": is_rain}],
                },
                "calendar": {
                    "events": [{"id": ev_id, "title": ev_title, "start": f"{f_date}T06:30:00+07:00", "end": f"{f_date}T08:00:00+07:00", "status": "confirmed"}]
                },
            },
            objective=f"Check weather on {f_date} (rain={is_rain}) and conditionally update/cancel calendar event.",
            expect_mutation=is_rain,
            plan_fn=_plan_w_cal,
        ))

    # Self-cancellations & user corrections across apps (8 train + 1 val)
    cancel_cases = [
        ("sim_cancel_tr_01_ecom", "train", ["ecommerce", "ewallet"],
         "Tìm mua giúp anh chiếc bàn phím cơ trên sàn thương mại điện tử. À mà thôi, đồng nghiệp vừa tặng một cái rồi, không cần mua nữa nhé."),
        ("sim_cancel_tr_02_stocks", "train", ["stocks", "banking"],
         "Đặt lệnh bán 300 cổ phiếu FPT trong danh mục đi. Thôi khỏi cần, thị trường đang hồi phục tốt cứ giữ nguyên đấy."),
        ("sim_cancel_tr_03_parking", "train", ["parking", "maps"],
         "Tìm bãi đỗ xe gần phố đi bộ Nguyễn Huệ đặt trước 2 tiếng. À thôi bỏ qua đi, tối nay mình đi taxi cho tiện."),
        ("sim_cancel_tr_04_grocery", "train", ["grocery", "pantry"],
         "Đặt mua 2 hộp sữa tươi trên ứng dụng đi chợ giúp chị. Khoan đã, trong tủ lạnh vẫn còn nguyên 1 thùng, thôi đừng đặt nữa."),
        ("sim_cancel_tr_05_event", "train", ["event_tickets", "calendar"],
         "Đặt 2 vé xem đêm nhạc cuối tuần này giúp mình. À mà thôi, tối đó mình phải đi công tác rồi, cứ để nguyên không đặt nhé."),
        ("sim_cancel_tr_06_transit", "train", ["public_transit", "ewallet"],
         "Nạp 100 nghìn vào thẻ tàu điện công cộng cho anh. Ơ thôi khỏi, trong thẻ vẫn còn nhiều tiền lắm chưa cần nạp đâu."),
        ("sim_cancel_val_01_flights", "val", ["flights", "calendar"],
         "Tìm đặt vé máy bay đi Phú Quốc tuần sau giúp anh. À mà thôi, lịch nghỉ phép vừa bị hoãn rồi, thôi không cần tìm hay đặt gì nữa."),
    ]
    for idx, (sid, sp, apps_l, req_text) in enumerate(cancel_cases):
        city, prof, loc = _make_profile(idx)
        scenarios.append(ScenarioSpec(
            scenario_id=sid,
            template_id=f"tpl_self_cancel_{sp}_{apps_l[0]}",
            split=sp,
            scenario_type="cancellation",
            primary_apps=[apps_l[0]],
            installed_apps=apps_l + ["reminders"],
            user_request=req_text,
            current_time="2026-11-23T12:00:00+07:00",
            city=city,
            user_profile=prof,
            location=loc,
            initial_app_states={a: {} for a in apps_l},
            objective="Recognize self-cancellation phrase at end of request and emit 0 tool calls.",
            expect_mutation=False,
            plan_fn=lambda r: None,
        ))

    return scenarios

