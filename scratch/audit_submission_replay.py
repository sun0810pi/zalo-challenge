import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(".")
sys.path.insert(0, str((ROOT / "public_test").resolve()))
from zaic_tools import ToolEnv
from zaic_tools.validate import catalog_tools

cat_path = ROOT / "public_test/public_test_tools_catalog.json"
catalog = json.loads(cat_path.read_text(encoding="utf-8"))
cat_tools = catalog_tools(catalog)

sub_path = Path(r"c:\Users\naph\Downloads\submission (1).jsonl")
if not sub_path.is_file():
    sub_path = Path("submissions/submission_v1_public27.1.jsonl")

lines = [json.loads(l) for l in sub_path.read_text(encoding="utf-8").splitlines() if l.strip()]

hash_matches = 0
hash_mismatches = 0
total_calls = 0
successful_calls = 0
failed_calls = 0
samples_with_failed_calls = 0
samples_all_calls_ok = 0

error_types = {}
sample_failures = []

for item in lines:
    sid = item["id"]
    calls = item.get("tool_calls", [])
    submitted_hash = item.get("state_hash")

    env = ToolEnv("public_test", sid, root=ROOT)
    sample_had_fail = False
    fail_reasons = []

    for call in calls:
        total_calls += 1
        name = call.get("name")
        args = call.get("arguments", {})
        res = env.call(name, args)
        if res.get("ok") is True:
            successful_calls += 1
        else:
            failed_calls += 1
            sample_had_fail = True
            err = str(res.get("error", "unknown"))
            fail_reasons.append(f"{name}: {err}")
            # Categorize error
            category = err
            if "thiếu tham số" in err:
                category = "thiếu tham số bắt buộc"
            elif "không có trong catalog" in err:
                category = "tool không có trong catalog"
            elif "chưa cài" in err:
                category = "app chưa cài"
            elif "không tìm thấy" in err:
                category = "không tìm thấy ID/dữ liệu"
            elif "đã tồn tại" in err:
                category = "đã tồn tại"
            error_types[category] = error_types.get(category, 0) + 1

    replayed_hash = env.get_state_hash()
    if replayed_hash == submitted_hash:
        hash_matches += 1
    else:
        hash_mismatches += 1

    if sample_had_fail:
        samples_with_failed_calls += 1
        sample_failures.append((sid, fail_reasons))
    else:
        samples_all_calls_ok += 1

print("=== KẾT QUẢ REPLAY SUBMISSION V1/V2 (27.1 điểm) ===")
print(f"File: {sub_path}")
print(f"Tổng số mẫu: {len(lines)}")
print(f"Hash khớp với replay: {hash_matches} / {len(lines)} ({hash_matches/len(lines)*100:.2f}%)")
print(f"Hash KHÔNG KHỚP: {hash_mismatches} / {len(lines)} ({hash_mismatches/len(lines)*100:.2f}%)")
print(f"Tổng số tool calls: {total_calls}")
print(f"Calls thành công (ok=True): {successful_calls} ({successful_calls/max(total_calls,1)*100:.2f}%)")
print(f"Calls THẤT BẠI (ok=False): {failed_calls} ({failed_calls/max(total_calls,1)*100:.2f}%)")
print(f"Số mẫu có ít nhất 1 call thất bại: {samples_with_failed_calls} / {len(lines)} ({samples_with_failed_calls/len(lines)*100:.2f}%)")
print(f"Số mẫu toàn bộ call thành công: {samples_all_calls_ok} / {len(lines)} ({samples_all_calls_ok/len(lines)*100:.2f}%)")
print("\nTop các loại lỗi từ chối của ToolEnv:")
for k, v in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  - [{v} lần] {k}")

print("\nVí dụ 5 mẫu thất bại cụ thể:")
for sid, errs in sample_failures[:5]:
    print(f"  [{sid}]:")
    for e in errs:
        print(f"    -> {e}")
