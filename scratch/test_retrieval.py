import sys, json
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train = [json.loads(l) for l in open('train/train.jsonl', encoding='utf-8')]
train_texts = [r['audio_text'] for r in train]
test = json.load(open('cache/asr_public_test_small.json', encoding='utf-8'))
test_keys = list(test.keys())[:15]
test_texts = [test[k] for k in test_keys]

vec = TfidfVectorizer().fit(train_texts + test_texts)
train_mat = vec.transform(train_texts)
test_mat = vec.transform(test_texts)
sims = cosine_similarity(test_mat, train_mat)

for i, k in enumerate(test_keys):
    best_idx = sims[i].argmax()
    best_score = sims[i][best_idx]
    gt_calls = [c['name'] for c in train[best_idx]['label']['tool_calls']]
    print(f"Test [{k}]:")
    print(f"  ASR: {test_texts[i]}")
    print(f"  Top match (sim={best_score:.3f}): {train_texts[best_idx]}")
    print(f"  GT calls: {gt_calls}\n")
