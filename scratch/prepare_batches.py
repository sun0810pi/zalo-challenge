import sys, json, os

samples = [json.loads(l) for l in open('public_test/public_test.jsonl', encoding='utf-8') if l.strip()]

# Load already transcribed
done = set()
if os.path.exists('cache/gemini_asr/batch_0.json'):
    d = json.load(open('cache/gemini_asr/batch_0.json', encoding='utf-8'))
    done.update(d.keys())

# Also check other batch files if any
for f in os.listdir('cache/gemini_asr'):
    if f.endswith('.json'):
        try:
            d = json.load(open(os.path.join('cache/gemini_asr', f), encoding='utf-8'))
            done.update(d.keys())
        except Exception:
            pass

remaining = [s for s in samples if s['audio'] not in done]
print(f"Total: {len(samples)} | Already done: {len(done)} | Remaining: {len(remaining)}")

batch_size = 25
batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]
print(f"Divided into {len(batches)} batches of up to {batch_size} files.")

os.makedirs('cache/gemini_asr/jobs', exist_ok=True)
for i, b in enumerate(batches, 1):
    job_file = f'cache/gemini_asr/jobs/job_{i}.json'
    items = [(s['id'], s['audio']) for s in b]
    with open(job_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Job {i}: {len(items)} files ({items[0][0]} -> {items[-1][0]})")
