"""Compile all clean ASR transcriptions and run multi_clause_solver to generate submission.jsonl."""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
sys.path.insert(0, str((ROOT / "public_test").resolve()))
sys.path.insert(0, str(ROOT.resolve()))

from zaic_tools import ToolEnv
from evaluate import evaluate_predictions
from scratch.multi_clause_solver import solve_with_react

def collect_all_transcriptions() -> dict[str, str]:
    merged: dict[str, str] = {}
    
    # 1. Fallback base from small
    small_path = "cache/asr_public_test_small.json"
    if os.path.exists(small_path):
        try:
            d_small = json.load(open(small_path, encoding='utf-8'))
            merged.update(d_small)
        except Exception:
            pass

    # 2. Overwrite with large-v3-turbo / gemini batches
    asr_dir = "cache/gemini_asr"
    if os.path.exists(asr_dir):
        for fname in sorted(os.listdir(asr_dir)):
            if fname.endswith("_done.json") or fname == "batch_0.json":
                fpath = os.path.join(asr_dir, fname)
                try:
                    d = json.load(open(fpath, encoding='utf-8'))
                    for k, v in d.items():
                        if v and v.strip():
                            merged[k] = v.strip()
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
                    
    return merged

def main():
    transcriptions = collect_all_transcriptions()
    print(f"Loaded {len(transcriptions)} total audio transcriptions.")
    
    test_samples = [json.loads(l) for l in open("public_test/public_test.jsonl", encoding="utf-8") if l.strip()]
    print(f"Processing all {len(test_samples)} public_test samples with Multi-Clause ReAct Solver...")
    
    t0 = time.time()
    sub_rows = []
    empty_count = 0
    non_empty_count = 0
    
    for s in test_samples:
        qid = s["id"]
        rel_audio = s["audio"]
        txt = transcriptions.get(rel_audio) or ""
        
        env = ToolEnv("public_test", qid, root=ROOT)
        solve_with_react(env, txt, s["context"])
        
        calls = env.get_tool_calls()
        st_hash = env.get_state_hash()
        
        if calls:
            non_empty_count += 1
        else:
            empty_count += 1
            
        sub_rows.append({"id": qid, "tool_calls": calls, "state_hash": st_hash})

    out_file = Path("submission.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in sub_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nWrote {len(sub_rows)} samples to submission.jsonl in {time.time()-t0:.2f}s.")
    print(f"  Non-empty tool calls : {non_empty_count}/{len(sub_rows)} ({non_empty_count/len(sub_rows)*100:.1f}%)")
    print(f"  Empty tool calls     : {empty_count}/{len(sub_rows)} ({empty_count/len(sub_rows)*100:.1f}%)")
    
    # Run evaluation and schema validation
    print("\nRunning schema verification via evaluate.py:")
    evaluate_predictions(out_file, root=ROOT, split="public_test")

if __name__ == '__main__':
    main()
