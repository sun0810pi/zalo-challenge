import sys, os, json, datetime, re, time
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
sys.path.insert(0, str((ROOT / "public_test").resolve()))
sys.path.insert(0, str(ROOT.resolve()))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zaic_tools import ToolEnv
from evaluate import evaluate_predictions

# 1. Load train corpus
train = [json.loads(l) for l in open('train/train.jsonl', encoding='utf-8')]
train_texts = [r['audio_text'] for r in train]
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(train_texts)
train_mat = vec.transform(train_texts)

from scratch.multi_clause_solver import solve_with_react
from scratch.test_hybrid_agent import replay_with_binding
from scratch.build_final_submission import collect_all_transcriptions

def main():
    transcriptions = collect_all_transcriptions()
    test_samples = [json.loads(l) for l in open('public_test/public_test.jsonl', encoding='utf-8') if l.strip()]
    print(f"Testing combined solver on {len(test_samples)} samples...")
    
    react_matched = 0
    template_matched = 0
    empty_count = 0
    
    sub_rows = []
    for s in test_samples:
        qid = s["id"]
        rel_audio = s["audio"]
        txt = transcriptions.get(rel_audio) or ""
        
        env = ToolEnv("public_test", qid, root=ROOT)
        
        # 1. Primary: High-precision ReAct solver
        calls = solve_with_react(env, txt, s["context"])
        
        # 2. Fallback: If ReAct produced no calls, check if high-sim template exists
        if not calls:
            tl = txt.lower()
            negations = ["à mà thôi", "mà thôi", "thôi khỏi", "khỏi cần", "bỏ qua đi", "đừng làm"]
            if not any(k in tl for k in negations):
                q_vec = vec.transform([txt])
                sims = cosine_similarity(q_vec, train_mat)[0]
                best_idx = sims.argmax()
                best_sim = sims[best_idx]
                if best_sim >= 0.18:
                    tmpl_calls = train[best_idx]['label']['tool_calls']
                    if tmpl_calls:
                        replay_with_binding(env, txt, s["context"], tmpl_calls)
                        calls = env.get_tool_calls()
                        if calls:
                            template_matched += 1
                            
        if calls:
            if not template_matched:
                react_matched += 1
        else:
            empty_count += 1
            
        st_hash = env.get_state_hash()
        sub_rows.append({"id": qid, "tool_calls": calls, "state_hash": st_hash})

    out_file = Path("submission.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in sub_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nWrote {len(sub_rows)} samples:")
    print(f"  ReAct rule matches    : {react_matched}")
    print(f"  Template fallbacks    : {template_matched}")
    print(f"  Total non-empty calls : {len(sub_rows) - empty_count}/{len(sub_rows)} ({(len(sub_rows) - empty_count)/len(sub_rows)*100:.1f}%)")
    print(f"  Empty calls           : {empty_count}/{len(sub_rows)} ({empty_count/len(sub_rows)*100:.1f}%)")
    
    evaluate_predictions(out_file, root=ROOT, split="public_test")

if __name__ == '__main__':
    main()
