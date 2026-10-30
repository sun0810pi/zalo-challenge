import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sub1 = Path(r"c:\Users\naph\Downloads\submission (1).jsonl")
lines = [json.loads(l) for l in sub1.read_text(encoding="utf-8").splitlines() if l.strip()]

pub_lines = [json.loads(l) for l in Path("public_test/public_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
pub_map = {l["id"]: l for l in pub_lines}

new_apps = {'habit_tracker', 'stocks', 'grocery', 'translator', 'ecommerce', 'train_bus_tickets', 'recipes', 'pantry', 'event_tickets', 'public_transit', 'parking'}

app_counter = Counter()
tool_counter = Counter()
new_app_samples = 0
new_app_calls = 0

for item in lines:
    sid = item["id"]
    pub_item = pub_map.get(sid, {})
    installed = set((pub_item.get("context") or {}).get("installed_apps", {}).keys())
    
    calls = item.get("tool_calls", [])
    has_new_app_call = False
    for c in calls:
        name = c.get("name", "")
        tool_counter[name] += 1
        app = name.split(".")[0] if "." in name else name
        app_counter[app] += 1
        if app in new_apps:
            has_new_app_call = True
            new_app_calls += 1
    if has_new_app_call:
        new_app_samples += 1

print(f"Tổng số mẫu nộp: {len(lines)}")
print(f"Số mẫu có gọi tool thuộc 11 App mới: {new_app_samples} / {len(lines)} ({new_app_samples/len(lines)*100:.1f}%)")
print(f"Tổng số cuộc gọi vào 11 App mới: {new_app_calls}")

print("\n--- TOP 20 APPS ĐƯỢC GỌI NHIỀU NHẤT TRONG QWEN V2 ---")
for app, count in app_counter.most_common(20):
    is_new = " [APP MỚI]" if app in new_apps else ""
    print(f"  {app:20s}: {count:4d} calls{is_new}")

print("\n--- SỐ LẦN GỌI 11 APP MỚI TRONG QWEN V2 ---")
for app in sorted(new_apps):
    print(f"  {app:20s}: {app_counter[app]:4d} calls")
