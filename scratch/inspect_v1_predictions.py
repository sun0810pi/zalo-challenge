import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sub_path = Path("submissions/submission_v1_public27.1.jsonl")
preds = {json.loads(l)["id"]: json.loads(l) for l in sub_path.read_text(encoding="utf-8").splitlines() if l.strip()}

asr_cache_path = Path("cache/asr_public_test.json")
asr_map = json.loads(asr_cache_path.read_text(encoding="utf-8")) if asr_cache_path.is_file() else {}

pub_samples = [json.loads(l) for l in Path("public_test/public_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"Loaded {len(pub_samples)} public samples. Inspecting first 15 samples in submission_v1:")

for s in pub_samples[:15]:
    sid = s["id"]
    audio_path = s["audio"]
    raw_transcript = asr_map.get(audio_path, "<no asr>")
    pred = preds.get(sid, {})
    calls = pred.get("tool_calls", [])
    installed = list((s.get("context") or {}).get("installed_apps", {}).keys())
    
    print(f"\n--- [{sid}] ---")
    print(f"  Transcript : {raw_transcript}")
    print(f"  Installed  : {', '.join(installed)}")
    print(f"  Pred Calls ({len(calls)}):")
    for c in calls:
        print(f"    -> {c['name']}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})")
