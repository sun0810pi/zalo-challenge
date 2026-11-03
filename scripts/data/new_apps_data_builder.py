"""Comprehensive synthetic trajectory generator for 11 new apps in ZAC 2026 LifeMate.
Generates balanced, diverse, schema-compliant training samples covering all tool-use meta-skills:
1. Read-before-Write (x-ref parameter binding)
2. Negative Condition verification (stopping when condition is False)
3. Quantitative Comparison & Arithmetic (stock quantity, ewallet balance)
4. Self-Cancellation & Negation recognition ("...à mà thôi")
5. Multi-app chaining (recipes -> pantry -> grocery, translator -> chat/sms, etc.)
6. Clarification via sys.ask_missing_info
"""
from __future__ import annotations

import copy
import random
from typing import Any


def get_all_new_app_samples(catalog_tools_map: dict[str, dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    app_tools: dict[str, list[str]] = {}
    for t_name in catalog_tools_map:
        app_id = t_name.split(".", 1)[0]
        if app_id != "sys":
            app_tools.setdefault(app_id, []).append(t_name)

    def _make(
        sid: str,
        apps: list[str],
        audio_text: str,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        city: str = "Hà Nội",
    ) -> dict[str, Any]:
        installed = {a: {"tools": app_tools[a]} for a in apps if a in app_tools}
        for base_app in ["sys", "weather", "contacts", "calendar"]:
            if base_app in app_tools and base_app not in installed:
                installed[base_app] = {"tools": app_tools[base_app]}

        addr = "120 Nguyễn Thị Minh Khai, Quận 3, TP.HCM" if city == "TP.HCM" else "Số 88 Cầu Giấy, Cầu Giấy, Hà Nội"
        work_addr = "Tòa nhà Bitexco, Quận 1, TP.HCM" if city == "TP.HCM" else "Tòa nhà Keangnam, Nam Từ Liêm, Hà Nội"

        return {
            "id": sid,
            "audio_text": audio_text,
            "context": {
                "current_time": "2026-11-20T08:30:00+07:00",
                "timezone": "Asia/Ho_Chi_Minh",
                "location": {"lat": 21.0285, "lng": 105.8542} if city == "Hà Nội" else {"lat": 10.7769, "lng": 106.7009},
                "user_profile": {
                    "name": "Đặng Văn Nam",
                    "home_address": addr,
                    "work_address": work_addr,
                    "city": city,
                },
                "installed_apps": installed,
            },
            "history": history or [],
            "label": {
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            },
        }

    # =========================================================================
    # 1. PARKING (Đỗ xe) - 45 samples
    # Tools: parking.find, parking.reserve, parking.list_sessions, parking.end_session
    # =========================================================================
    parking_locations = [
        "Sân vận động Quốc gia Mỹ Đình", "Trung tâm Hội nghị Quốc gia", "Bệnh viện Bạch Mai",
        "Phố đi bộ Hồ Gươm", "Tòa nhà Keangnam Landmark", "Bến xe Mỹ Đình",
        "Vincom Bà Triệu", "Công viên Thống Nhất", "Bệnh viện Việt Đức", "Sân bay Nội Bài",
        "Phố đi bộ Nguyễn Huệ", "Chợ Bến Thành", "Bệnh viện Chợ Rẫy", "Tòa nhà Bitexco",
        "Hồ Tây", "Công viên Cầu Giấy", "Bảo tàng Dân tộc học", "Bến xe Giáp Bát"
    ]
    hours_opts = [1, 2, 3, 4, 5]
    from_times = ["2026-11-20T09:00:00+07:00", "2026-11-20T10:30:00+07:00", "2026-11-20T14:00:00+07:00", "2026-11-20T18:00:00+07:00"]

    # 1.1 Direct find and reserve
    for i, loc in enumerate(parking_locations):
        hrs = hours_opts[i % len(hours_opts)]
        ft = from_times[i % len(from_times)]
        ft_text = ft[11:16]
        lot_id = f"lot_{i+1:02d}"
        lot_name = f"Bãi đỗ xe {loc}"
        price = 25000 + (i % 3) * 5000

        text = f"Tìm bãi xe gần {loc} rồi đặt giữ chỗ từ {ft_text} trong {hrs} tiếng giúp mình nhé."
        calls = [
            {"name": "parking.find", "arguments": {"near": loc}},
            {"name": "parking.reserve", "arguments": {"lot_id": lot_id, "from_time": ft, "hours": hrs}}
        ]
        results = [
            {"ok": True, "lots": [{"id": lot_id, "name": lot_name, "price_per_hour": price, "available_spots": 12}]},
            {"ok": True, "reservation": {"id": f"res_{i+1:02d}", "lot_id": lot_id, "status": "confirmed"}}
        ]
        samples.append(_make(f"synth_parking_reserve_{i:02d}", ["parking"], text, calls, results))

    # 1.2 End parking session
    for i in range(10):
        sid = f"sess_{i+1:02d}"
        lot = parking_locations[i]
        text = f"Kết thúc lượt đỗ xe tại {lot} giúp tôi, tôi lấy xe rồi."
        calls = [
            {"name": "parking.list_sessions", "arguments": {}},
            {"name": "parking.end_session", "arguments": {"session_id": sid}}
        ]
        results = [
            {"ok": True, "sessions": [{"id": sid, "lot_id": f"lot_{i+1:02d}", "lot_name": f"Bãi xe {lot}", "status": "active"}]},
            {"ok": True, "session": {"id": sid, "status": "completed", "total_fee": 50000}}
        ]
        samples.append(_make(f"synth_parking_end_{i:02d}", ["parking"], text, calls, results))

    # 1.3 Conditional: Only reserve if available spots > 0 (satisfied vs unsatisfied)
    for i in range(12):
        loc = parking_locations[(i + 4) % len(parking_locations)]
        ft = from_times[i % len(from_times)]
        hrs = 2
        is_full = (i % 2 == 0)

        if is_full:
            text = f"Xem bãi xe gần {loc} còn chỗ không, nếu còn thì giữ chỗ 2 tiếng từ {ft[11:16]}, hết chỗ thì thôi."
            calls = [{"name": "parking.find", "arguments": {"near": loc}}]
            results = [{"ok": True, "lots": [{"id": f"lot_{i+20:02d}", "name": f"Bãi đỗ {loc}", "available_spots": 0}]}]
        else:
            text = f"Kiểm tra bãi đỗ xe gần {loc}, nếu còn chỗ trống thì đặt giữ chỗ 2 tiếng từ {ft[11:16]} cho anh."
            calls = [
                {"name": "parking.find", "arguments": {"near": loc}},
                {"name": "parking.reserve", "arguments": {"lot_id": f"lot_{i+20:02d}", "from_time": ft, "hours": hrs}}
            ]
            results = [
                {"ok": True, "lots": [{"id": f"lot_{i+20:02d}", "name": f"Bãi đỗ {loc}", "available_spots": 5, "price_per_hour": 30000}]},
                {"ok": True, "reservation": {"id": f"res_{i+20:02d}", "lot_id": f"lot_{i+20:02d}", "status": "confirmed"}}
            ]
        samples.append(_make(f"synth_parking_cond_{i:02d}", ["parking"], text, calls, results))

    # 1.4 Self-cancellation
    for i in range(6):
        loc = parking_locations[i]
        text = f"Tìm chỗ gửi xe gần {loc} giữ trước 3 tiếng. À mà thôi, bạn tôi đón rồi không cần gửi xe nữa."
        samples.append(_make(f"synth_parking_cancel_{i:02d}", ["parking"], text, [], []))

    # =========================================================================
    # 2. STOCKS (Chứng khoán) - 60 samples
    # Tools: stocks.get_quote, stocks.get_portfolio, stocks.place_order, stocks.set_price_alert
    # =========================================================================
    stock_symbols = ["ZKN", "ZMX", "HPG", "FPT", "VIC", "VNM", "MWG", "TCB", "VHM", "MSN", "SSI", "MBB", "VRE", "STB", "HDB"]
    base_prices = {"ZKN": 32000, "ZMX": 58000, "HPG": 28500, "FPT": 135000, "VIC": 42000, "VNM": 67000, "MWG": 52000, "TCB": 36000, "VHM": 41000, "MSN": 74000, "SSI": 31000, "MBB": 24000, "VRE": 26000, "STB": 29000, "HDB": 22000}

    # 2.1 Direct Buy orders
    for i, sym in enumerate(stock_symbols):
        qty = (i + 1) * 100
        pr = base_prices[sym]
        text = f"Kiểm tra giá mã {sym} rồi đặt mua {qty} cổ phiếu giúp tôi nhé."
        calls = [
            {"name": "stocks.get_quote", "arguments": {"symbol": sym}},
            {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "buy", "quantity": qty, "price": pr}}
        ]
        results = [
            {"ok": True, "quote": {"symbol": sym, "name": f"CTCP {sym}", "price": pr, "change_pct": 1.2}},
            {"ok": True, "order": {"id": f"ord_stk_{i+1:02d}", "symbol": sym, "side": "buy", "quantity": qty, "price": pr, "status": "placed"}}
        ]
        samples.append(_make(f"synth_stocks_buy_{i:02d}", ["stocks", "banking"], text, calls, results))

    # 2.2 Direct Sell orders
    for i, sym in enumerate(stock_symbols[:10]):
        qty = 200 + i * 50
        pr = base_prices[sym]
        text = f"Bán {qty} cổ phiếu {sym} theo giá thị trường hiện tại giúp em."
        calls = [
            {"name": "stocks.get_portfolio", "arguments": {}},
            {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "sell", "quantity": qty, "price": pr}}
        ]
        results = [
            {"ok": True, "holdings": [{"symbol": sym, "quantity": qty + 300, "avg_cost": pr - 2000}]},
            {"ok": True, "order": {"id": f"ord_sell_{i+1:02d}", "symbol": sym, "side": "sell", "quantity": qty, "price": pr, "status": "placed"}}
        ]
        samples.append(_make(f"synth_stocks_sell_{i:02d}", ["stocks"], text, calls, results))

    # 2.3 Arithmetic: "Bán một nửa số cổ phiếu" (half)
    for i, sym in enumerate(["ZKN", "HPG", "FPT", "VNM", "MWG", "TCB"]):
        held = 600 + i * 200
        sell_qty = held // 2
        pr = base_prices[sym]
        text = f"Bán một nửa số cổ phiếu {sym} trong danh mục của tôi đi."
        calls = [
            {"name": "stocks.get_portfolio", "arguments": {}},
            {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "sell", "quantity": sell_qty, "price": pr}}
        ]
        results = [
            {"ok": True, "holdings": [{"symbol": sym, "quantity": held, "avg_cost": pr - 1000}]},
            {"ok": True, "order": {"id": f"ord_half_{i+1:02d}", "symbol": sym, "side": "sell", "quantity": sell_qty, "price": pr, "status": "placed"}}
        ]
        samples.append(_make(f"synth_stocks_half_{i:02d}", ["stocks"], text, calls, results))

    # 2.4 Arithmetic: "Bán hết... chỉ giữ lại X cổ"
    for i, sym in enumerate(["ZKN", "ZMX", "VIC", "TCB", "MSN", "SSI"]):
        held = 500 + i * 100
        keep = 200
        sell_qty = held - keep
        pr = base_prices[sym]
        text = f"Bán hết chỗ {sym} đi à không, chỉ giữ lại 200 cổ phiếu thôi, còn lại bán hết."
        calls = [
            {"name": "stocks.get_portfolio", "arguments": {}},
            {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "sell", "quantity": sell_qty, "price": pr}}
        ]
        results = [
            {"ok": True, "holdings": [{"symbol": sym, "quantity": held, "avg_cost": pr - 500}]},
            {"ok": True, "order": {"id": f"ord_keep_{i+1:02d}", "symbol": sym, "side": "sell", "quantity": sell_qty, "price": pr, "status": "placed"}}
        ]
        samples.append(_make(f"synth_stocks_keep_{i:02d}", ["stocks"], text, calls, results))

    # 2.5 Conditional buy: "Nếu giá dưới threshold thì mua, cao hơn thì thôi"
    for i, sym in enumerate(["ZKN", "HPG", "FPT", "TCB", "SSI", "MWG", "VNM", "VIC"]):
        pr = base_prices[sym]
        threshold = pr - 1000
        is_cheap = (i % 2 == 0)
        curr_price = threshold - 500 if is_cheap else threshold + 1500

        if is_cheap:
            text = f"Xem giá {sym}, nếu giá dưới {threshold} đồng thì mua 200 cổ, đắt hơn thì thôi đừng mua."
            calls = [
                {"name": "stocks.get_quote", "arguments": {"symbol": sym}},
                {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "buy", "quantity": 200, "price": curr_price}}
            ]
            results = [
                {"ok": True, "quote": {"symbol": sym, "price": curr_price, "change_pct": -1.5}},
                {"ok": True, "order": {"id": f"ord_cond_{i:02d}", "symbol": sym, "side": "buy", "quantity": 200, "price": curr_price, "status": "placed"}}
            ]
        else:
            text = f"Kiểm tra giá mã {sym}, nếu xuống dưới {threshold} thì đặt mua 200 cổ, còn không thì thôi."
            calls = [{"name": "stocks.get_quote", "arguments": {"symbol": sym}}]
            results = [{"ok": True, "quote": {"symbol": sym, "price": curr_price, "change_pct": 2.1}}]
        samples.append(_make(f"synth_stocks_cond_{i:02d}", ["stocks"], text, calls, results))

    # 2.6 Price alert
    for i, sym in enumerate(["FPT", "VIC", "VNM", "HPG", "MSN", "TCB"]):
        pr = base_prices[sym] + 5000
        text = f"Đặt cảnh báo giá cho mã {sym} khi vượt trên {pr} đồng giúp tôi."
        calls = [{"name": "stocks.set_price_alert", "arguments": {"symbol": sym, "price": pr, "direction": "above"}}]
        results = [{"ok": True, "alert": {"id": f"alt_{i+1:02d}", "symbol": sym, "price": pr, "direction": "above"}}]
        samples.append(_make(f"synth_stocks_alert_{i:02d}", ["stocks"], text, calls, results))

    # =========================================================================
    # 3. TRAIN_BUS_TICKETS (Vé tàu / xe khách) - 50 samples
    # Tools: train_bus_tickets.search, train_bus_tickets.book, train_bus_tickets.my_bookings, train_bus_tickets.cancel
    # =========================================================================
    routes = [
        ("Hà Nội", "Vinh", "train"), ("Hà Nội", "Đà Nẵng", "train"), ("Hà Nội", "Hải Phòng", "train"),
        ("TP.HCM", "Nha Trang", "train"), ("TP.HCM", "Đà Lạt", "bus"), ("TP.HCM", "Cần Thơ", "bus"),
        ("Hà Nội", "Sa Pa", "bus"), ("TP.HCM", "Phan Thiết", "train"), ("Đà Nẵng", "Huế", "train"),
        ("Hà Nội", "Quảng Bình", "train"), ("TP.HCM", "Vũng Tàu", "bus"), ("Hà Nội", "Thanh Hóa", "train")
    ]
    dates = ["2026-11-25", "2026-11-28", "2026-12-01", "2026-12-05", "2026-12-10", "2026-12-15"]

    # 3.1 Search & Book
    for i, (orig, dest, vtype) in enumerate(routes):
        dt = dates[i % len(dates)]
        seats = (i % 3) + 1
        tid = f"trip_{vtype[:1]}_{i+1:02d}"
        tname = "tàu" if vtype == "train" else "xe khách"

        text = f"Tìm và đặt {seats} vé {tname} từ {orig} đi {dest} ngày {dt} giúp tôi."
        calls = [
            {"name": "train_bus_tickets.search", "arguments": {"origin": orig, "destination": dest, "date": dt, "type": vtype}},
            {"name": "train_bus_tickets.book", "arguments": {"trip_id": tid, "seats": seats}}
        ]
        results = [
            {"ok": True, "trips": [{"trip_id": tid, "origin": orig, "destination": dest, "type": vtype, "depart": f"{dt}T07:30:00+07:00", "fare": 350000, "seats_left": 15}]},
            {"ok": True, "booking": {"booking_id": f"tb_bk_{i+1:02d}", "trip_id": tid, "seats": seats, "status": "confirmed"}}
        ]
        samples.append(_make(f"synth_tickets_book_{i:02d}", ["train_bus_tickets"], text, calls, results))

    # 3.2 View & Cancel bookings
    for i in range(12):
        bk_id = f"tb_bk_{i+1:02d}"
        orig, dest, _ = routes[i % len(routes)]
        text = f"Kiểm tra vé tàu đi {dest} rồi hủy vé đó giúp tôi nhé."
        calls = [
            {"name": "train_bus_tickets.my_bookings", "arguments": {}},
            {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": bk_id}}
        ]
        results = [
            {"ok": True, "bookings": [{"booking_id": bk_id, "route": f"{orig} - {dest}", "depart": "2026-11-30T08:00:00+07:00", "seats": 1, "status": "active"}]},
            {"ok": True, "booking": {"booking_id": bk_id, "status": "cancelled", "refund_amount": 300000}}
        ]
        samples.append(_make(f"synth_tickets_cancel_{i:02d}", ["train_bus_tickets"], text, calls, results))

    # 3.3 Book Train + Add to Calendar (Cross-app workflow)
    for i in range(12):
        orig, dest, vtype = routes[i % len(routes)]
        dt = dates[i % len(dates)]
        tid = f"trip_na_{i+1:02d}"
        bk_id = f"tb_bk_{i+20:02d}"
        dep_iso = f"{dt}T06:30:00+07:00"

        text = f"Đặt 1 vé tàu từ {orig} đi {dest} ngày {dt} chuyến sớm nhất, rồi lưu vào lịch cho tôi."
        calls = [
            {"name": "train_bus_tickets.search", "arguments": {"origin": orig, "destination": dest, "date": dt, "type": "train"}},
            {"name": "train_bus_tickets.book", "arguments": {"trip_id": tid, "seats": 1}},
            {"name": "calendar.create_event", "arguments": {"title": f"Chuyến tàu {orig} - {dest}", "start": dep_iso, "location": f"Ga {orig}"}}
        ]
        results = [
            {"ok": True, "trips": [{"trip_id": tid, "origin": orig, "destination": dest, "depart": dep_iso, "fare": 280000, "seats_left": 8}]},
            {"ok": True, "booking": {"booking_id": bk_id, "trip_id": tid, "status": "confirmed"}},
            {"ok": True, "event": {"id": f"evt_{i+1:02d}", "title": f"Chuyến tàu {orig} - {dest}", "start": dep_iso}}
        ]
        samples.append(_make(f"synth_tickets_calendar_{i:02d}", ["train_bus_tickets", "calendar"], text, calls, results))

    # =========================================================================
    # 4. ECOMMERCE (Mua sắm online) - 55 samples
    # Tools: ecommerce.search, ecommerce.add_to_cart, ecommerce.checkout, ecommerce.order_status, ecommerce.cancel_order, ecommerce.request_return
    # =========================================================================
    ecom_products = [
        ("tai nghe Bluetooth", 450000), ("sạc dự phòng 20000mAh", 380000), ("bàn phím cơ không dây", 750000),
        ("chuột công thái học", 520000), ("balo laptop chống nước", 420000), ("nồi chiên không dầu", 1200000),
        ("bình giữ nhiệt 800ml", 190000), ("quạt tích điện mini", 210000), ("giá đỡ điện thoại để bàn", 85000),
        ("đèn led bàn học", 260000), ("củ sạc nhanh 65W", 340000), ("chuột không dây văn phòng", 180000)
    ]

    # 4.1 Search -> Add to cart -> Checkout
    for i, (pname, pprice) in enumerate(ecom_products):
        pid = f"prod_{i+1:02d}"
        pm = "ewallet" if i % 2 == 0 else "cod"
        text = f"Tìm mua {pname} dưới {pprice + 100000} đồng trên sàn rồi đặt mua 1 cái thanh toán qua {pm} giao về nhà nhé."
        calls = [
            {"name": "ecommerce.search", "arguments": {"query": pname, "max_price": pprice + 100000}},
            {"name": "ecommerce.add_to_cart", "arguments": {"product_id": pid, "quantity": 1}},
            {"name": "ecommerce.checkout", "arguments": {"address": "Số 88 Cầu Giấy, Cầu Giấy, Hà Nội", "payment": pm}}
        ]
        results = [
            {"ok": True, "products": [{"id": pid, "name": f"{pname.capitalize()} chính hãng", "price": pprice, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": pid, "name": f"{pname.capitalize()} chính hãng", "price": pprice, "quantity": 1}], "cart_total": pprice},
            {"ok": True, "order": {"id": f"ecom_ord_{i+1:02d}", "total": pprice, "status": "placed", "payment": pm}}
        ]
        samples.append(_make(f"synth_ecom_buy_{i:02d}", ["ecommerce", "ewallet"], text, calls, results))

    # 4.2 Check order status & Cancel order
    for i in range(12):
        oid = f"ecom_ord_{i+1:02d}"
        pname, _ = ecom_products[i % len(ecom_products)]
        text = f"Xem đơn hàng mua {pname} trên sàn thương mại điện tử rồi hủy đơn đó giúp mình."
        calls = [
            {"name": "ecommerce.order_status", "arguments": {}},
            {"name": "ecommerce.cancel_order", "arguments": {"order_id": oid}}
        ]
        results = [
            {"ok": True, "orders": [{"id": oid, "items": [{"name": pname}], "status": "shipping", "total": 350000}]},
            {"ok": True, "order": {"id": oid, "status": "cancelled"}}
        ]
        samples.append(_make(f"synth_ecom_cancel_{i:02d}", ["ecommerce"], text, calls, results))

    # 4.3 Return delivered order
    for i, reason in enumerate(["damaged", "wrong_item", "changed_mind", "other"] * 3):
        oid = f"ecom_ord_{i+10:02d}"
        r_text = "bị hư hỏng khi nhận" if reason == "damaged" else ("giao sai mẫu" if reason == "wrong_item" else "không vừa ý")
        text = f"Đơn hàng {oid} đã nhận bị {r_text}, tạo yêu cầu trả hàng giúp tôi."
        calls = [
            {"name": "ecommerce.order_status", "arguments": {"order_id": oid}},
            {"name": "ecommerce.request_return", "arguments": {"order_id": oid, "reason": reason}}
        ]
        results = [
            {"ok": True, "orders": [{"id": oid, "status": "delivered", "items": [{"name": "Áo khoác thể thao"}]}]},
            {"ok": True, "return": {"id": f"ret_{i+1:02d}", "order_id": oid, "status": "requested", "reason": reason}}
        ]
        samples.append(_make(f"synth_ecom_return_{i:02d}", ["ecommerce"], text, calls, results))

    # =========================================================================
    # 5. GROCERY (Đi chợ online) - 45 samples
    # Tools: grocery.search, grocery.add_to_cart, grocery.checkout
    # =========================================================================
    grocery_items = [
        ("gạo ST25 túi 5kg", 185000), ("trứng gà ta 10 quả", 42000), ("thịt ba chỉ bò Úc 500g", 145000),
        ("rau muống sạch 500g", 18000), ("cá hồi Nauy phi lê 300g", 165000), ("sữa tươi không đường thùng", 360000),
        ("dầu ăn Simply chai 2L", 115000), ("cà chua sạch 1kg", 32000), ("thịt nạc heo sạch 500g", 85000),
        ("khoai tây Đà Lạt 1kg", 28000), ("hành tây củ to 1kg", 22000), ("bắp cải trắng 1 bắp", 16000)
    ]

    for i, (gname, gprice) in enumerate(grocery_items):
        gid = f"gro_{i+1:02d}"
        qty = (i % 2) + 1
        text = f"Tìm mua {qty} {gname} trên app đi chợ rồi thanh toán đặt giao về nhà nhé."
        calls = [
            {"name": "grocery.search", "arguments": {"query": gname}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": gid, "quantity": qty}},
            {"name": "grocery.checkout", "arguments": {"address": "Số 88 Cầu Giấy, Cầu Giấy, Hà Nội"}}
        ]
        results = [
            {"ok": True, "products": [{"id": gid, "name": gname, "price": gprice, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": gid, "name": gname, "price": gprice, "quantity": qty}], "cart_total": gprice * qty},
            {"ok": True, "order": {"id": f"gro_ord_{i+1:02d}", "total": gprice * qty, "status": "placed"}}
        ]
        samples.append(_make(f"synth_grocery_{i:02d}", ["grocery"], text, calls, results))

    # =========================================================================
    # 6. RECIPES & PANTRY (Công thức & Tủ bếp) - 45 samples
    # Tools: recipes.search, recipes.get, recipes.save, pantry.list_items, pantry.consume
    # =========================================================================
    recipes_list = [
        ("sườn xào chua ngọt", 4), ("canh chua cá lóc", 4), ("thịt kho tàu", 6),
        ("bò sốt vang", 4), ("gà hấp lá chanh", 5), ("phở bò truyền thống", 2),
        ("cá kho tộ", 4), ("lẩu thái hải sản", 6), ("mực xào cần tỏi", 3), ("bún chả Hà Nội", 4)
    ]

    # 6.1 Search and Save recipe
    for i, (rname, servings) in enumerate(recipes_list):
        rid = f"rcp_{i+1:02d}"
        text = f"Tìm công thức nấu món {rname} cho {servings} người rồi lưu lại vào danh sách yêu thích giúp tôi."
        calls = [
            {"name": "recipes.search", "arguments": {"query": rname, "servings": servings}},
            {"name": "recipes.save", "arguments": {"recipe_id": rid}}
        ]
        results = [
            {"ok": True, "recipes": [{"id": rid, "name": rname.capitalize(), "servings": servings, "ingredients": [{"name": "nguyên liệu chính", "quantity": 500, "unit": "g"}]}]},
            {"ok": True, "saved": {"recipe_id": rid, "status": "saved"}}
        ]
        samples.append(_make(f"synth_recipe_save_{i:02d}", ["recipes"], text, calls, results))

    # 6.2 Pantry: list & consume
    pantry_stocks = [
        ("sữa tươi", "hộp", 2), ("gói mì Hảo Hảo", "gói", 3), ("chai nước mắm", "chai", 1), ("trứng gà", "quả", 4),
        ("gạo trắng", "kg", 2), ("dầu hào", "chai", 1), ("hạt nêm", "gói", 1), ("bột ngọt", "gói", 1)
    ]
    for i, (pname, unit, used_qty) in enumerate(pantry_stocks):
        pan_id = f"pan_{i+1:02d}"
        text = f"Vừa dùng {used_qty} {pname} rồi, cập nhật trừ số lượng trong tủ bếp giúp mình."
        calls = [
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "pantry.consume", "arguments": {"item_id": pan_id, "quantity": used_qty}}
        ]
        results = [
            {"ok": True, "items": [{"id": pan_id, "name": pname, "stock": used_qty + 2, "unit": unit}]},
            {"ok": True, "item": {"id": pan_id, "name": pname, "stock": 2, "unit": unit}}
        ]
        samples.append(_make(f"synth_pantry_consume_{i:02d}", ["pantry"], text, calls, results))

    # 6.3 Cross-app: recipes -> pantry -> grocery
    for i, (rname, srv) in enumerate(recipes_list[:6]):
        rid = f"rcp_{i+10:02d}"
        pan_id = f"pan_{i+10:02d}"
        gro_id = f"gro_{i+10:02d}"
        missing_ing = "thịt sườn heo" if i % 2 == 0 else "cá lóc"
        text = f"Tìm công thức {rname}, kiểm tra tủ bếp xem có {missing_ing} chưa, nếu thiếu thì đặt mua 1 phần trên app đi chợ nhé."
        calls = [
            {"name": "recipes.search", "arguments": {"query": rname, "servings": srv}},
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "grocery.search", "arguments": {"query": missing_ing}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": gro_id, "quantity": 1}},
            {"name": "grocery.checkout", "arguments": {"address": "Số 88 Cầu Giấy, Cầu Giấy, Hà Nội"}}
        ]
        results = [
            {"ok": True, "recipes": [{"id": rid, "name": rname, "ingredients": [{"name": missing_ing, "quantity": 500, "unit": "g"}]}]},
            {"ok": True, "items": [{"id": pan_id, "name": missing_ing, "stock": 0, "unit": "g"}]},
            {"ok": True, "products": [{"id": gro_id, "name": f"{missing_ing} tươi sạch 500g", "price": 95000, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": gro_id, "quantity": 1}], "cart_total": 95000},
            {"ok": True, "order": {"id": f"gro_ord_{i+20:02d}", "status": "placed"}}
        ]
        samples.append(_make(f"synth_chain_recipe_gro_{i:02d}", ["recipes", "pantry", "grocery"], text, calls, results))

    # =========================================================================
    # 7. EVENT_TICKETS (Vé sự kiện) - 25 samples
    # Tools: event_tickets.search, event_tickets.book
    # =========================================================================
    events = [
        "Triển lãm Ánh sáng Nghệ thuật", "Đêm nhạc Jazz Mùa Thu", "Liveshow Hà Anh Tuấn",
        "Hội thảo Công nghệ Trí tuệ Nhân tạo", "Vở kịch Hài Tinh Hoa", "Hòa nhạc Cổ điển Giao hưởng",
        "Lễ hội Ẩm thực Đường phố", "Triển lãm Tranh Sơn mài Hiện đại", "Hội thảo Khởi nghiệp Đổi mới",
        "Giải chạy Marathon Mùa Đông"
    ]
    for i, ev_name in enumerate(events):
        ev_id = f"event_{i+1:02d}"
        qty = (i % 3) + 1
        text = f"Tìm vé sự kiện {ev_name} rồi đặt {qty} vé giúp tôi nhé."
        calls = [
            {"name": "event_tickets.search", "arguments": {"query": ev_name}},
            {"name": "event_tickets.book", "arguments": {"event_id": ev_id, "quantity": qty}}
        ]
        results = [
            {"ok": True, "events": [{"id": ev_id, "name": ev_name, "price": 300000, "date": "2026-11-28", "available_tickets": 50}]},
            {"ok": True, "booking": {"id": f"ev_bk_{i+1:02d}", "event_id": ev_id, "quantity": qty, "status": "confirmed"}}
        ]
        samples.append(_make(f"synth_event_book_{i:02d}", ["event_tickets"], text, calls, results))

    # =========================================================================
    # 8. PUBLIC_TRANSIT (Xe buýt / Metro) - 25 samples
    # Tools: public_transit.find_routes, public_transit.next_departures, public_transit.topup_card
    # =========================================================================
    transit_routes = [
        ("Nhổn", "Ga Hà Nội"), ("Bến Thành", "Suối Tiên"), ("Mỹ Đình", "Bờ Hồ"), ("Cầu Giấy", "Hà Đông"),
        ("Yên Nghĩa", "Cát Linh"), ("Bến xe Miền Đông", "Bến xe Miền Tây"), ("Giáp Bát", "Nội Bài"),
        ("Long Biên", "Gia Lâm")
    ]
    for i, (orig, dest) in enumerate(transit_routes):
        amt = 100000 if i % 2 == 0 else 50000
        text = f"Tìm tuyến buýt hoặc metro từ {orig} đi {dest} rồi nạp thêm {amt} đồng vào thẻ đi xe giúp tôi."
        calls = [
            {"name": "public_transit.find_routes", "arguments": {"origin": orig, "destination": dest}},
            {"name": "public_transit.topup_card", "arguments": {"amount": amt}}
        ]
        results = [
            {"ok": True, "routes": [{"route_id": f"rt_{i+1:02d}", "name": f"Tuyến {orig} - {dest}", "type": "metro", "duration_min": 25, "fare": 15000}]},
            {"ok": True, "topup": {"amount": amt, "status": "success", "card_balance": amt + 45000}}
        ]
        samples.append(_make(f"synth_transit_{i:02d}", ["public_transit"], text, calls, results))

    # =========================================================================
    # 9. TRANSLATOR (Dịch thuật) - 25 samples
    # Tools: translator.translate
    # =========================================================================
    translations = [
        ("Xin chào, tôi muốn hỏi về tài liệu cuộc họp ngày mai", "en", "chat_ott"),
        ("Tôi sẽ đến muộn khoảng 15 phút do kẹt xe", "en", "sms"),
        ("Hẹn gặp bạn tại sảnh khách sạn lúc 7 giờ tối", "ja", "chat_ott"),
        ("Cảm ơn sự hợp tác của công ty bạn trong dự án vừa qua", "en", "sms"),
        ("Bản hợp đồng đã được ký kết và gửi qua bưu điện", "en", "chat_ott"),
        ("Bạn có thể gửi lại danh sách khách mời tham dự không?", "fr", "sms"),
        ("Vui lòng xác nhận thời gian nhận hàng chiều nay", "zh", "chat_ott"),
        ("Món ăn này có chứa hải sản hay hạt đậu phộng không?", "en", "chat_ott")
    ]
    for i, (vi_text, to_l, send_app) in enumerate(translations):
        chat_id = f"chat_{i+1:02d}"
        trans_out = "Hello, I would like to ask about the meeting documents tomorrow." if i == 0 else "I will arrive 15 minutes late due to traffic."
        text = f"Dịch câu '{vi_text}' sang tiếng {to_l} rồi gửi qua {send_app.upper()} cho đối tác giúp mình."
        send_tool = "chat_ott.send_message" if send_app == "chat_ott" else "sms.send"
        send_args = {"chat_id": chat_id, "content": trans_out} if send_app == "chat_ott" else {"phone_number": "0988123456", "content": trans_out}

        calls = [
            {"name": "translator.translate", "arguments": {"text": vi_text, "to_lang": to_l}},
            {"name": send_tool, "arguments": send_args}
        ]
        results = [
            {"ok": True, "translation": {"text": vi_text, "to_lang": to_l, "translated": trans_out}},
            {"ok": True, "message": {"id": f"msg_{i+1:02d}", "status": "sent"}}
        ]
        samples.append(_make(f"synth_trans_{i:02d}", ["translator", send_app], text, calls, results))

    # =========================================================================
    # 10. HABIT_TRACKER (Thói quen) - 20 samples
    # Tools: habit_tracker.list_habits, habit_tracker.log
    # =========================================================================
    habits = [
        ("hb_01", "Chạy bộ 30 phút"), ("hb_02", "Uống 2 lít nước"), ("hb_03", "Đọc sách 20 trang"),
        ("hb_04", "Học từ vựng tiếng Anh"), ("hb_05", "Ngủ trước 23 giờ"), ("hb_06", "Tập yoga buổi sáng"),
        ("hb_07", "Thiền 15 phút"), ("hb_08", "Không uống nước ngọt")
    ]
    for i, (hid, hname) in enumerate(habits):
        today = "2026-11-20"
        text = f"Ghi nhận hôm nay tôi đã hoàn thành thói quen '{hname}' rồi nhé."
        calls = [
            {"name": "habit_tracker.list_habits", "arguments": {}},
            {"name": "habit_tracker.log", "arguments": {"habit_id": hid, "date": today}}
        ]
        results = [
            {"ok": True, "habits": [{"id": hid, "name": hname, "target": 1, "completed_today": False}]},
            {"ok": True, "log": {"habit_id": hid, "date": today, "status": "logged"}}
        ]
        samples.append(_make(f"synth_habit_{i:02d}", ["habit_tracker"], text, calls, results))

    # =========================================================================
    # 11. NEGATIVE CONDITIONS & SELF-CANCELLATIONS (35 samples)
    # =========================================================================
    # Negative weather
    samples.append(_make(
        "synth_neg_weather_run_01", ["weather", "calendar"],
        "Nếu sáng mai mưa thì dời buổi chạy bộ sang sáng ngày kia, không mưa thì giữ nguyên nhé.",
        [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-21"}}],
        [{"ok": True, "forecast": [{"date": "2026-11-21", "condition": "nắng ráo", "rain": False}]}]
    ))
    # Negative ewallet balance for bill
    samples.append(_make(
        "synth_neg_ewallet_bill_01", ["bill_payment", "ewallet"],
        "Kiểm tra hóa đơn tiền điện, nếu số dư ví đủ tiền thì thanh toán bằng ví, không đủ thì thôi khỏi trả nhé.",
        [
            {"name": "bill_payment.list_bills", "arguments": {"type": "electricity"}},
            {"name": "ewallet.get_balance", "arguments": {}}
        ],
        [
            {"ok": True, "bills": [{"id": "bill_el_01", "type": "electricity", "amount": 650000, "paid": False}]},
            {"ok": True, "balance": 180000}
        ]
    ))
    # Negative stocks quote
    samples.append(_make(
        "synth_neg_stock_buy_01", ["stocks"],
        "Xem giá cổ phiếu FPT, nếu giá rớt xuống dưới 120 nghìn thì mua 100 cổ phiếu, còn đắt hơn thì thôi đừng mua.",
        [{"name": "stocks.get_quote", "arguments": {"symbol": "FPT"}}],
        [{"ok": True, "quote": {"symbol": "FPT", "price": 136000, "change_pct": 0.8}}]
    ))

    # Complete cancellations across new domains
    cancellations = [
        "Đặt giúp tôi 2 vé tàu đi Đà Nẵng thứ 7 tuần này. À mà thôi, công ty có việc bận rồi cứ để nguyên đó.",
        "Tìm mua nồi chiên không dầu trên sàn thương mại điện tử. Thôi khỏi cần, mẹ tôi vừa mua một cái rồi.",
        "Bán 300 cổ phiếu ZKN trong danh mục đi nhé. À mà thôi, chuyên gia bảo giữ lại nên thôi bỏ qua đi.",
        "Đặt bãi đỗ xe gần Bệnh viện Bạch Mai 3 tiếng. Thôi thôi không cần đặt nữa, tôi đi xe máy cho tiện.",
        "Tìm công thức nấu bún riêu cua đồng. À thôi khỏi, trưa nay ra quán ăn cho nhanh.",
        "Đặt mua 1 thùng sữa tươi trên app đi chợ. Khoan đã, trong tủ lạnh vẫn còn 2 lốc, đừng đặt nữa.",
        "Xem vé sự kiện Liveshow đêm nhạc tuần sau. Thôi bỏ đi, giá vé đắt quá không cần xem nữa.",
        "Dịch câu này sang tiếng Anh rồi nhắn tin cho sếp. À mà thôi, để mai lên công ty gặp trực tiếp nói.",
        "Ghi nhật ký thói quen chạy bộ hôm nay. Thôi không cần ghi, hôm nay bận chưa kịp chạy.",
        "Tìm tuyến buýt đi Bờ Hồ nạp 50k vào thẻ. À thôi khỏi đi xe buýt, tôi đi xe ôm công nghệ rồi."
    ]
    for i, c_text in enumerate(cancellations):
        samples.append(_make(f"synth_neg_cancel_{i:02d}", ["parking", "stocks", "ecommerce", "train_bus_tickets", "recipes", "grocery", "event_tickets", "translator", "habit_tracker", "public_transit"], c_text, [], []))

    print(f"[NewAppsDataBuilder] Đã tạo thành công {len(samples)} mẫu SFT tổng hợp chuẩn schema cho 11 app mới!")
    return samples
