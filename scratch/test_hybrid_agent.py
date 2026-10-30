import sys, os, json, datetime, re
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
sys.path.insert(0, str((ROOT / "train").resolve()))
sys.path.insert(0, str(ROOT.resolve()))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zaic_tools import ToolEnv
from evaluate import compute_call_f1, _load_zaic_tools

zaic_tools, FREE_TEXT_KEYS, _canon_scalar, _tokens, catalog_tools, schema_validate = _load_zaic_tools(ROOT, "train")

# Load training data
train = [json.loads(l) for l in open('train/train.jsonl', encoding='utf-8')]
train_texts = [r['audio_text'] for r in train]
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(train_texts)
train_mat = vec.transform(train_texts)

DIGIT_WORDS = {
    "không": "0", "một": "1", "mốt": "1", "hai": "2", "ba": "3",
    "bốn": "4", "tư": "4", "năm": "5", "lăm": "5", "sáu": "6",
    "bảy": "7", "tám": "8", "chín": "9"
}

def parse_phone(text: str) -> str | None:
    m = re.search(r"(?:0|\+84)\d{9,10}", text)
    if m: return m.group(0)
    tokens = re.findall(r"\w+", text.lower())
    digits = []
    for t in tokens:
        if t in DIGIT_WORDS: digits.append(DIGIT_WORDS[t])
        elif t.isdigit(): digits.append(t)
        else:
            if len(digits) >= 10: break
            digits = []
    res = "".join(digits)
    if len(res) == 10 and res.startswith("0"): return res
    if len(res) == 11 and res.startswith("84"): return "0" + res[2:]
    return None

def extract_money(text: str) -> int:
    tl = text.lower()
    if "hai triệu tám" in tl: return 2800000
    if "một triệu hai" in tl: return 1200000
    if "hai trăm nghìn" in tl or "hai trăm ngàn" in tl: return 200000
    if "năm mươi nghìn" in tl or "năm mươi ngàn" in tl or "50 nghìn" in tl or "50 ngàn" in tl or "50k" in tl: return 50000
    if "hai mươi nghìn" in tl or "20 nghìn" in tl or "20k" in tl: return 20000
    if "một trăm nghìn" in tl or "100k" in tl: return 100000
    m = re.search(r"(\d+[\d\.,]*)\s*(?:đồng|đ|vnd|nghìn|ngàn|k|triệu)?", tl)
    if m:
        v_str = m.group(1).replace(".", "").replace(",", "")
        if v_str.isdigit():
            val = int(v_str)
            if "triệu" in tl and val < 1000: val *= 1000000
            elif ("nghìn" in tl or "ngàn" in tl or "k" in tl) and val < 1000: val *= 1000
            return val
    return 50000

def parse_time(text: str, current_time_str: str) -> str | None:
    try: now = datetime.datetime.fromisoformat(current_time_str)
    except Exception: now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
    tl = text.lower()
    target_date = now.date()
    if any(k in tl for k in ["sáng mai", "tối mai", "chiều mai", "trưa mai", "ngày mai", "mai"]):
        target_date = target_date + datetime.timedelta(days=1)
    elif "ngày mốt" in tl:
        target_date = target_date + datetime.timedelta(days=2)
        
    date_m = re.search(r"ngày\s*(\d{1,2})(?:\s*tháng\s*(\d{1,2}))?", tl)
    if date_m:
        d = int(date_m.group(1))
        m = int(date_m.group(2)) if date_m.group(2) else target_date.month
        try: target_date = datetime.date(target_date.year, m, d)
        except Exception: pass
        
    hour, minute = 8, 0
    colon_m = re.search(r"(\d{1,2})[:h](\d{2})", tl)
    if colon_m:
        hour, minute = int(colon_m.group(1)), int(colon_m.group(2))
    else:
        ruoi_m = re.search(r"(\d{1,2}|sáu|năm|tám|chín|mười)\s*rưỡi", tl)
        word_to_num = {"năm": 5, "sáu": 6, "tám": 8, "chín": 9, "mười": 10}
        if ruoi_m:
            rh = ruoi_m.group(1)
            hour = word_to_num.get(rh, int(rh) if rh.isdigit() else 8)
            minute = 30
        else:
            gio_m = re.search(r"(\d{1,2})\s*(?:giờ|h)", tl)
            if gio_m:
                hour = int(gio_m.group(1))
                minute = 0
                
    if any(k in tl for k in ["tối", "chiều", "đêm"]) and hour < 12 and hour != 0:
        hour += 12
    elif "sáng" in tl and hour >= 12:
        hour -= 12
        
    return datetime.datetime.combine(
        target_date, datetime.time(hour, minute), tzinfo=now.tzinfo or datetime.timezone(datetime.timedelta(hours=7))
    ).isoformat()

def extract_name(text: str) -> str:
    m = re.search(r"(?:cho|gọi|nhắn|dùm)\s+(?:anh|chị|em|bạn|bác|cô|chú|sếp)?\s*([A-ZÀ-Ỹa-zà-ỹ0-9\s]+?)(?:[:,\.\s]+(?:nội dung|với|là|giờ|số|nạp|chuyển|tiền|mai|hôm|nhé|nha|ạ)|$)", text, re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        c = re.sub(r"^(?:anh|chị|em|bạn|bác|cô|chú|sếp)\s+", "", c, flags=re.IGNORECASE).strip()
        if c and len(c.split()) <= 4: return c
    return ""

def replay_with_binding(env: Any, query_text: str, ctx: dict[str, Any], template_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile = ctx.get("user_profile", {})
    now_str = ctx.get("current_time", "")
    allowed = env.allowed
    
    # Track execution context
    last_calendar_events = []
    last_bills = []
    last_contacts = []
    last_beneficiaries = []
    last_devices = []
    
    for tc in template_calls:
        tname = tc['name']
        if tname not in allowed:
            continue
        args = dict(tc.get('arguments') or {})
        
        # 1. Weather
        if tname == 'weather.get_forecast':
            parsed_t = parse_time(query_text, now_str)
            d_str = parsed_t[:10] if parsed_t else now_str[:10]
            res = env.call(tname, {"date": d_str})
            # Check rain condition
            if not any(f.get('rain') or 'mưa' in f.get('condition', '').lower() for f in (res.get('forecast') or [])):
                break # condition failed, stop executing subsequent calls
            continue
            
        # 2. Calendar
        elif tname == 'calendar.list_events':
            res = env.call(tname, {})
            last_calendar_events = res.get('events', []) if res.get('ok') else []
            continue
        elif tname == 'calendar.update_event':
            if last_calendar_events:
                ev = last_calendar_events[0]
                ev_start = datetime.datetime.fromisoformat(ev["start"]) + datetime.timedelta(hours=1)
                ev_end = datetime.datetime.fromisoformat(ev["end"]) + datetime.timedelta(hours=1)
                env.call(tname, {"event_id": ev["id"], "start": ev_start.isoformat(), "end": ev_end.isoformat()})
            else:
                env.call(tname, args)
            continue
        elif tname == 'calendar.cancel_event':
            if last_calendar_events:
                env.call(tname, {"event_id": last_calendar_events[0]["id"]})
            else:
                env.call(tname, args)
            continue
            
        # 3. Ride Hailing
        elif tname == 'ride_hailing.book':
            pickup = profile.get("work_address") or profile.get("home_address") or "Công ty"
            dropoff = last_calendar_events[0].get("location") if last_calendar_events else (args.get("dropoff") or "Điểm đến")
            veh = "car7" if "7 chỗ" in query_text else ("bike" if "xe ôm" in query_text or "xe máy" in query_text else "car4")
            call_dict = {"pickup": pickup, "dropoff": dropoff, "vehicle": veh}
            if "schedule_time" in args:
                call_dict["schedule_time"] = args["schedule_time"]
            env.call(tname, call_dict)
            continue
            
        # 4. Alarms & Reminders
        elif tname == 'alarm.create':
            iso_t = parse_time(query_text, now_str)
            env.call(tname, {"time": iso_t or args.get("time", now_str)})
            continue
        elif tname == 'reminders.create':
            iso_t = parse_time(query_text, now_str)
            env.call(tname, {"content": args.get("content", "lời nhắc"), "time": iso_t or args.get("time", now_str)})
            continue
            
        # 5. Bill Payment & Banking
        elif tname == 'bill_payment.list_bills':
            res = env.call(tname, {})
            last_bills = res.get('bills', []) if res.get('ok') else []
            continue
        elif tname == 'bill_payment.pay':
            b_id = last_bills[0]['id'] if last_bills else args.get('bill_id', 'bill_01')
            env.call(tname, {"bill_id": b_id, "source": args.get("source", "banking")})
            continue
        elif tname == 'banking.list_beneficiaries':
            res = env.call(tname, {})
            last_beneficiaries = res.get('beneficiaries', []) if res.get('ok') else []
            continue
        elif tname == 'banking.get_balance':
            env.call(tname, {})
            continue
        elif tname == 'banking.transfer':
            amt = extract_money(query_text) or args.get("amount", 50000)
            name_q = extract_name(query_text)
            matched_b = None
            for b in last_beneficiaries:
                if name_q and name_q.lower() in b.get("name", "").lower():
                    matched_b = b
                    break
            b_id = matched_b['id'] if matched_b else (last_beneficiaries[0]['id'] if last_beneficiaries else args.get('beneficiary_id', 'ben_01'))
            env.call(tname, {"from_account_id": "acc_01", "beneficiary_id": b_id, "amount": amt})
            continue
            
        # 6. Ewallet
        elif tname == 'ewallet.topup_mobile':
            amt = extract_money(query_text) or args.get("amount", 50000)
            phone = parse_phone(query_text) or (last_contacts[0].get('phone') if last_contacts else args.get("phone", "09919377320"))
            env.call(tname, {"phone": phone, "amount": amt})
            continue
        elif tname == 'ewallet.transfer':
            amt = extract_money(query_text) or args.get("amount", 50000)
            phone = parse_phone(query_text) or (last_contacts[0].get('phone') if last_contacts else args.get("phone", "09919377320"))
            env.call(tname, {"phone": phone, "amount": amt})
            continue
            
        # 7. Contacts & Phone / SMS
        elif tname == 'contacts.search':
            name_q = extract_name(query_text) or args.get("query", "Minh")
            res = env.call(tname, {"query": name_q})
            last_contacts = res.get('contacts', []) if res.get('ok') else []
            continue
        elif tname == 'phone.call':
            c_id = last_contacts[0]['id'] if last_contacts else args.get('contact_id', 'c_01')
            env.call(tname, {"contact_id": c_id})
            continue
        elif tname == 'sms.send':
            c_id = last_contacts[0]['id'] if last_contacts else args.get('contact_id', 'c_01')
            env.call(tname, {"contact_id": c_id, "content": args.get("content", "Thông báo")})
            continue
            
        # 8. Smart Home
        elif tname == 'smart_home.get_status':
            res = env.call(tname, {})
            last_devices = res.get('devices', []) if res.get('ok') else []
            continue
        elif tname == 'smart_home.light_set':
            d_id = args.get('device_id')
            if "nhỏ" in query_text or "2" in query_text:
                d_id = "light_bedroom2"
            elif "phòng khách" in query_text:
                d_id = "light_living"
            elif "phòng ngủ" in query_text:
                d_id = "light_bedroom"
            p_args = {"device_id": d_id or "light_living"}
            if "brightness" in args:
                p_args["brightness"] = args["brightness"]
            if "power" in args:
                p_args["power"] = "off" if "tắt" in query_text else "on"
            env.call(tname, p_args)
            continue
        elif tname == 'smart_home.climate_set':
            temp_m = re.search(r"(\d{2})\s*độ", query_text)
            temp = int(temp_m.group(1)) if temp_m else args.get("temperature", 25)
            mode = args.get("mode") or ("off" if "tắt" in query_text else "cool")
            call_dict = {"device_id": args.get("device_id", "climate_living"), "mode": mode, "temperature": temp}
            env.call(tname, call_dict)
            continue
            
        # Fallback exact call
        else:
            env.call(tname, args)
            
    return env.get_tool_calls()

def test_eval(n_samples=50):
    name_f1s = []
    call_f1s = []
    state_matches = 0
    
    for idx in range(n_samples):
        target = train[idx]
        qid = target['id']
        gt_calls = target['label']['tool_calls']
        
        # Self-excluding retrieval
        q_vec = vec.transform([target['audio_text']])
        sims = cosine_similarity(q_vec, train_mat)[0]
        sims[idx] = -1.0
        best_idx = sims.argmax()
        template_calls = train[best_idx]['label']['tool_calls']
        
        gt_env = ToolEnv("train", qid, root=ROOT)
        for c in gt_calls: gt_env.call(c['name'], c['arguments'])
        gt_hash = gt_env.get_state_hash()
        
        pred_env = ToolEnv("train", qid, root=ROOT)
        replay_with_binding(pred_env, target['audio_text'], target['context'], template_calls)
        pred_calls = pred_env.get_tool_calls()
        
        nf1, ff1 = compute_call_f1(pred_calls, gt_calls, FREE_TEXT_KEYS, _canon_scalar, _tokens)
        name_f1s.append(nf1)
        call_f1s.append(ff1)
        if pred_env.get_state_hash() == gt_hash:
            state_matches += 1
            
    print(f"Evaluated {n_samples} samples with Hybrid Agent:")
    print(f"  Tool Name F1   : {sum(name_f1s)/n_samples*100:.2f}%")
    print(f"  Full Call F1   : {sum(call_f1s)/n_samples*100:.2f}%")
    print(f"  State Match Acc: {state_matches/n_samples*100:.2f}%")
    print(f"  Combined (50/50): {(0.5*state_matches/n_samples + 0.5*sum(call_f1s)/n_samples)*100:.2f}%")

if __name__ == '__main__':
    test_eval(50)
