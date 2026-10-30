import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sub1 = Path(r"c:\Users\naph\Downloads\submission (1).jsonl")
lines = [json.loads(l) for l in sub1.read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"Total lines in submission (1).jsonl: {len(lines)}")
for item in lines[:10]:
    print(f"\nID: {item['id']}")
    print(f"State Hash: {item.get('state_hash')}")
    calls = item.get("tool_calls", [])
    print(f"Tool calls ({len(calls)}):")
    for c in calls:
        print(f"  - {c.get('name')}: {json.dumps(c.get('arguments', {}), ensure_ascii=False)}")
