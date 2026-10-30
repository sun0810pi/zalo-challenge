import sys, os, json, datetime, re, time
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
sys.path.insert(0, str((ROOT / "public_test").resolve()))
sys.path.insert(0, str(ROOT.resolve()))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zaic_tools import ToolEnv
from zaic_tools.validate import catalog_tools, schema_validate
from evaluate import evaluate_predictions

# 1. Load train corpus
train = [json.loads(l) for l in open('train/train.jsonl', encoding='utf-8')]
train_texts = [r['audio_text'] for r in train]
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(train_texts)
train_mat = vec.transform(train_texts)

# 2. Load public test data and clean ASR
test_samples = [json.loads(l) for l in open('public_test/public_test.jsonl', encoding='utf-8') if l.strip()]
asr_cache = json.load(open('cache/asr_public_test_small.json', encoding='utf-8'))

# Import binding and helpers from scratch/test_hybrid_agent.py
from scratch.test_hybrid_agent import replay_with_binding, parse_time, extract_money, parse_phone, extract_name

def main():
    print(f"Generating submission for all {len(test_samples)} public test samples...")
    t0 = time.time()
    
    sub_rows = []
    empty_count = 0
    non_empty_count = 0
    top_sims = []
    
    for s in test_samples:
        qid = s["id"]
        rel_audio = s["audio"]
        query_text = asr_cache.get(rel_audio) or ""
        
        env = ToolEnv("public_test", qid, root=ROOT)
        allowed = env.allowed
        
        # Self-cancellation check
        tl = query_text.lower()
        if any(k in tl for k in ["à mà thôi", "mà thôi", "thôi khỏi", "khỏi cần", "bỏ qua đi", "đừng làm"]):
            # Check if user still wants to check weather
            if "chỉ xem" in tl or "chỉ cần xem" in tl:
                if "mưa" in tl and "weather.get_forecast" in allowed:
                    now_str = s["context"].get("current_time", "")
                    parsed_t = parse_time(tl, now_str)
                    d_str = parsed_t[:10] if parsed_t else now_str[:10]
                    env.call("weather.get_forecast", {"date": d_str})
            calls = env.get_tool_calls()
            st_hash = env.get_state_hash()
            sub_rows.append({"id": qid, "tool_calls": calls, "state_hash": st_hash})
            if calls: non_empty_count += 1
            else: empty_count += 1
            continue
            
        # Match against train templates
        q_vec = vec.transform([query_text])
        sims = cosine_similarity(q_vec, train_mat)[0]
        best_idx = sims.argmax()
        best_sim = sims[best_idx]
        top_sims.append(best_sim)
        
        # If very low similarity and contains no actionable keywords, leave empty
        action_keywords = ["đặt", "gọi", "nhắn", "chuyển", "nạp", "báo thức", "nhắc", "hẹn", "lịch", "đèn", "điều hòa", "máy lạnh", "mưa", "thời tiết", "vé", "bay"]
        if best_sim < 0.12 and not any(k in tl for k in action_keywords):
            calls = []
            st_hash = env.get_state_hash()
            sub_rows.append({"id": qid, "tool_calls": calls, "state_hash": st_hash})
            empty_count += 1
            continue
            
        template_calls = train[best_idx]['label']['tool_calls']
        replay_with_binding(env, query_text, s.get('context', {}), template_calls)
        
        calls = env.get_tool_calls()
        st_hash = env.get_state_hash()
        sub_rows.append({"id": qid, "tool_calls": calls, "state_hash": st_hash})
        if calls: non_empty_count += 1
        else: empty_count += 1
        
    out_file = Path("submission.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in sub_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nGenerated {len(sub_rows)} submissions in {time.time()-t0:.2f}s.")
    print(f"Non-empty tool calls: {non_empty_count}/{len(sub_rows)} ({non_empty_count/len(sub_rows)*100:.1f}%)")
    print(f"Empty tool calls    : {empty_count}/{len(sub_rows)} ({empty_count/len(sub_rows)*100:.1f}%)")
    print(f"Average top-1 similarity: {sum(top_sims)/len(top_sims):.3f}")
    
    # Validate with evaluate.py
    print("\nRunning schema validation with evaluate.py:")
    evaluate_predictions(out_file, root=ROOT, split="public_test")

if __name__ == '__main__':
    main()
