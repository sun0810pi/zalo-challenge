"""Hybrid Program-Aided Post-Processor & Replay Verifier for ZAC 2026 LifeMate.
Performs deterministic logic repairs on generated submission files:
1. Complete cancellation interception ("...à mà thôi" -> 0 calls)
2. Stock quantity arithmetic calculation ("giữ lại X" -> held_qty - X)
3. Bill payment balance matching (ewallet vs bank) & fake SMS suppression
4. Smart home multi-fan expansion
5. Over-call pruning (no parking when taking bike)
6. 100% official simulator ToolEnv replay to recompute exact state hashes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(".")
sys.path.insert(0, str((ROOT / "public_test").resolve()))
sys.path.insert(0, str(ROOT.resolve()))

from zaic_tools import ToolEnv
from src.hybrid_engine import HybridProgramAidedEngine
from src.preprocessor_a import ContextLanguagePreprocessorA


def main():
    parser = argparse.ArgumentParser(description="Chuẩn hóa logic và tính lại state hash cho file submission")
    parser.add_argument("--input", default="submission.jsonl", help="Đường dẫn file submission đầu vào")
    parser.add_argument("--output", default="submissions/submission_final.jsonl", help="Đường dẫn file submission đầu ra")
    parser.add_argument("--split", default="public_test", choices=["public_test", "train"])
    args = parser.parse_args()

    sub_path = Path(args.input)
    if not sub_path.is_file():
        print(f"Lỗi: Không tìm thấy file {sub_path}")
        return

    lines = [json.loads(l) for l in sub_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    split_dir = ROOT / args.split
    pub_samples = {
        json.loads(l)["id"]: json.loads(l)
        for l in open(split_dir / f"{args.split}.jsonl", encoding="utf-8")
        if l.strip()
    }

    # Load clean ASR transcripts from cache
    asr_map = {}
    for cname in [f"cache/asr_{args.split}.json", f"cache/asr_{args.split}_small.json"]:
        cp = ROOT / cname
        if cp.is_file():
            data = json.load(open(cp, encoding="utf-8"))
            for k, v in data.items():
                asr_map[k] = v
                asr_map[Path(k).name] = v

    preprocessor = ContextLanguagePreprocessorA()

    print(f"=== CHẠY HYBRID PROGRAM-AIDED REPAIR TRÊN {len(lines)} MẪU ({args.input}) ===")

    corrected_rows = []
    modifications = 0
    cancel_fixed = 0
    stock_fixed = 0
    bill_fixed = 0
    fan_fixed = 0
    parking_pruned = 0

    for item in lines:
        sid = item["id"]
        calls = item.get("tool_calls") or []
        pub_item = pub_samples.get(sid, {})
        audio_key = pub_item.get("audio", "")
        raw_user_text = asr_map.get(audio_key) or asr_map.get(Path(audio_key).name) or ""
        user_text = preprocessor.preprocess(raw_user_text, pub_item)

        # Check 1: Cancellation Interceptor
        if calls and HybridProgramAidedEngine.is_complete_cancellation(user_text):
            env = ToolEnv(args.split, sid, root=ROOT)
            corrected_rows.append({"id": sid, "tool_calls": [], "state_hash": env.get_state_hash()})
            cancel_fixed += 1
            modifications += 1
            print(f"[{sid}] Fixed cancellation -> 0 calls (text: {user_text[:60]!r})")
            continue

        # Replay on ToolEnv with Hybrid Policies
        env = ToolEnv(args.split, sid, root=ROOT)
        observed_entities = {}
        balances = {}
        adjusted_calls = []
        seen_names = set()
        suppress_sms = False

        for c in calls:
            c_copy = dict(c)
            name = c_copy.get("name", "")
            args_map = dict(c_copy.get("arguments") or {})

            # Rule A: Stock Math
            if name == "stocks.place_order":
                args_map = HybridProgramAidedEngine.adjust_stock_order(args_map, user_text, observed_entities)
                if args_map.get("quantity") != c.get("arguments", {}).get("quantity"):
                    stock_fixed += 1
                    modifications += 1
                    print(f"[{sid}] Fixed stock quantity: {c.get('arguments', {}).get('quantity')} -> {args_map.get('quantity')}")

            # Rule B: Bill Payment
            if name == "bill_payment.pay":
                args_map, supp = HybridProgramAidedEngine.adjust_bill_payment(args_map, user_text, observed_entities, balances)
                if args_map is None:
                    suppress_sms = True
                    bill_fixed += 1
                    modifications += 1
                    print(f"[{sid}] Suppressed bill payment (insufficient balance)")
                    continue
                if args_map.get("source") != c.get("arguments", {}).get("source"):
                    bill_fixed += 1
                    modifications += 1
                    print(f"[{sid}] Fixed bill source: {c.get('arguments', {}).get('source')} -> {args_map.get('source')}")
                if supp:
                    suppress_sms = True

            # Rule C: Suppress SMS if condition failed
            if name == "sms.send" and suppress_sms:
                modifications += 1
                print(f"[{sid}] Suppressed fake outgoing SMS")
                continue

            # Rule D: Over-call pruning (no parking when taking bike)
            if name == "parking.reserve" and any("ride_hailing.book" in s and "bike" in s for s in seen_names):
                parking_pruned += 1
                modifications += 1
                print(f"[{sid}] Pruned parking reserve for motorbike ride")
                continue

            # Execute call
            res = env.call(name, args_map)
            seen_names.add(f"{name}:{json.dumps(args_map, sort_keys=True)}")
            adjusted_calls.append({"name": name, "arguments": args_map})

            # Track entities & balances from read results
            if isinstance(res, dict):
                for k, val in res.items():
                    if isinstance(val, list):
                        for obj in val:
                            if isinstance(obj, dict) and obj.get("id"):
                                observed_entities[str(obj["id"])] = obj
                            if isinstance(obj, dict) and obj.get("symbol"):
                                observed_entities[str(obj["symbol"])] = obj
                    elif isinstance(val, dict):
                        if val.get("id"):
                            observed_entities[str(val["id"])] = val
                        if val.get("symbol"):
                            observed_entities[str(val["symbol"])] = val
                    elif isinstance(val, (int, float)) and "balance" in k:
                        balances[f"{name.split('.')[0]}.{k}"] = val

            # Rule E: Weather condition early stop
            if name in ("weather.get_forecast", "weather.get_current"):
                if not HybridProgramAidedEngine.check_weather_condition(user_text, res):
                    print(f"[{sid}] Stopped early on no-rain weather forecast")
                    break

            # Rule F: Smart Home multi-fan expansion
            if name == "smart_home.fan_set":
                extra_fans = HybridProgramAidedEngine.expand_smart_home_fans(env, user_text, observed_entities, args_map)
                for ef in extra_fans:
                    env.call(ef["name"], ef["arguments"])
                    adjusted_calls.append(ef)
                    fan_fixed += 1
                    modifications += 1
                    print(f"[{sid}] Expanded extra fan call: {ef['arguments']['device_id']}")

        final_hash = env.get_state_hash()
        corrected_rows.append({"id": sid, "tool_calls": adjusted_calls, "state_hash": final_hash})

    print(f"\n=== KẾT QUẢ SỬA LỖI HYBRID ===")
    print(f"Tổng số chỉnh sửa chính xác: {modifications}")
    print(f"  - Hủy lệnh chuẩn (Cancellation): {cancel_fixed}")
    print(f"  - Sửa số học chứng khoán (Stock math): {stock_fixed}")
    print(f"  - Sửa nguồn tiền hóa đơn (Bill payment): {bill_fixed}")
    print(f"  - Bật đủ toàn bộ quạt (Fan expansion): {fan_fixed}")
    print(f"  - Cắt tỉa call thừa (Over-call pruned): {parking_pruned}")

    out_p = Path(args.output)
    out_p.parent.mkdir(exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in corrected_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nĐã xuất bản sửa lỗi ra: {out_p}")


if __name__ == "__main__":
    main()
