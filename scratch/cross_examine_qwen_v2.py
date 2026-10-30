import glob
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sub1_path = Path(r"c:\Users\naph\Downloads\submission (1).jsonl")
preds = {json.loads(l)["id"]: json.loads(l) for l in sub1_path.read_text(encoding="utf-8").splitlines() if l.strip()}

# Load Gemini ASR
gemini_asr = {}
for gf in glob.glob("cache/gemini_asr/*.json"):
    try:
        gemini_asr.update(json.loads(Path(gf).read_text(encoding="utf-8")))
    except Exception:
        pass

whisper_asr = json.loads(Path("cache/asr_public_test.json").read_text(encoding="utf-8"))
pub_samples = [json.loads(l) for l in Path("public_test/public_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"Loaded {len(gemini_asr)} Gemini ground-truth transcripts. Comparing with Qwen v2 predictions:")

count = 0
for s in pub_samples:
    sid = s["id"]
    audio = s["audio"]
    gt_text = gemini_asr.get(audio)
    if not gt_text:
        continue
    
    w_text = whisper_asr.get(audio, "")
    pred = preds.get(sid, {})
    calls = pred.get("tool_calls", [])
    
    count += 1
    print(f"\n=======================================================")
    print(f"Sample: {sid}")
    print(f"  [Real User Voice] (Gemini) : {gt_text}")
    print(f"  [Whisper ASR Input to Qwen]: {w_text}")
    print(f"  [Qwen v2 Tool Calls ({len(calls)})]:")
    for c in calls:
        print(f"     -> {c.get('name')}: {json.dumps(c.get('arguments', {}), ensure_ascii=False)}")
    
    if count >= 15:
        break
