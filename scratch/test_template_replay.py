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

train = [json.loads(l) for l in open('train/train.jsonl', encoding='utf-8')]
train_texts = [r['audio_text'] for r in train]
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(train_texts)
train_mat = vec.transform(train_texts)

def evaluate_on_train(sample_indices):
    name_f1s = []
    call_f1s = []
    state_matches = 0
    
    for idx in sample_indices:
        target = train[idx]
        qid = target['id']
        gt_calls = target['label']['tool_calls']
        
        # Query vector excluding self
        q_vec = vec.transform([target['audio_text']])
        sims = cosine_similarity(q_vec, train_mat)[0]
        sims[idx] = -1.0 # exclude self
        best_match_idx = sims.argmax()
        sim_score = sims[best_match_idx]
        best_match = train[best_match_idx]
        
        # Ground truth state hash
        gt_env = ToolEnv("train", qid, root=ROOT)
        for c in gt_calls:
            gt_env.call(c['name'], c['arguments'])
        gt_hash = gt_env.get_state_hash()
        
        # Predict calls using template from best match
        pred_env = ToolEnv("train", qid, root=ROOT)
        template_calls = best_match['label']['tool_calls']
        
        # Replay template adaptively
        for tc in template_calls:
            tname = tc['name']
            if tname not in pred_env.allowed:
                continue
            targs = dict(tc['arguments'])
            
            # Context-adaptive argument substitution
            if tname == 'calendar.list_events':
                res = pred_env.call(tname, targs)
            elif tname == 'weather.get_forecast':
                # Use current date + 1 for tomorrow
                now_str = target['context']['current_time']
                now_dt = datetime.datetime.fromisoformat(now_str)
                d_str = (now_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                res = pred_env.call(tname, {"date": d_str})
                # Check if rain condition matches
                if not any(f.get('rain') or 'mưa' in f.get('condition','').lower() for f in (res.get('forecast') or [])):
                    # No rain -> branch ends
                    break
            elif tname == 'bill_payment.list_bills':
                res = pred_env.call(tname, {})
            elif tname == 'smart_home.get_status':
                res = pred_env.call(tname, {})
            elif tname == 'banking.list_beneficiaries':
                res = pred_env.call(tname, {})
            elif tname == 'banking.get_balance':
                res = pred_env.call(tname, {})
            elif tname == 'contacts.search':
                res = pred_env.call(tname, targs)
            else:
                pred_env.call(tname, targs)
                
        pred_calls = pred_env.get_tool_calls()
        nf1, ff1 = compute_call_f1(pred_calls, gt_calls, FREE_TEXT_KEYS, _canon_scalar, _tokens)
        name_f1s.append(nf1)
        call_f1s.append(ff1)
        if pred_env.get_state_hash() == gt_hash:
            state_matches += 1
            
    n = len(sample_indices)
    print(f"Evaluated {n} samples:")
    print(f"  Tool Name F1   : {sum(name_f1s)/n*100:.2f}%")
    print(f"  Full Call F1   : {sum(call_f1s)/n*100:.2f}%")
    print(f"  State Match Acc: {state_matches/n*100:.2f}%")
    print(f"  Combined (50/50): {(0.5*state_matches/n + 0.5*sum(call_f1s)/n)*100:.2f}%")

if __name__ == '__main__':
    evaluate_on_train(list(range(50)))
