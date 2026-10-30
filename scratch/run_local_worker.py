import sys, os, json, time, wave, numpy as np
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from faster_whisper import WhisperModel

def read_wav(p):
    with wave.open(p, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

def main():
    print("Loading large-v3-turbo from local directory...")
    t0 = time.time()
    model = WhisperModel('models/whisper-large-v3-turbo', device='cpu', compute_type='int8', cpu_threads=8)
    print(f"Model loaded in {time.time()-t0:.2f}s")
    
    # Process all jobs 1 to 25, skipping already completed ones
    for j_idx in range(1, 26):
        job_file = f"cache/gemini_asr/jobs/job_{j_idx}.json"
        done_file = f"cache/gemini_asr/job_{j_idx}_done.json"
        if not os.path.exists(job_file) or os.path.exists(done_file):
            continue
            
        items = json.load(open(job_file, encoding='utf-8'))
        print(f"\nProcessing Job {j_idx} ({len(items)} files)...")
        results = {}
        t_job = time.time()
        for idx, (qid, rel_audio) in enumerate(items, 1):
            wav_path = os.path.join("public_test", rel_audio)
            t_sample = time.time()
            try:
                segs, _ = model.transcribe(read_wav(wav_path), language='vi', beam_size=1)
                txt = " ".join(s.text.strip() for s in segs)
                results[rel_audio] = txt
                print(f"  [{idx}/{len(items)}] {qid} ({time.time()-t_sample:.1f}s): {txt[:50]}")
            except Exception as e:
                print(f"  [{idx}/{len(items)}] ERROR {qid}: {e}")
                results[rel_audio] = ""
                
        with open(done_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Job {j_idx} completed in {time.time()-t_job:.1f}s. Saved to {done_file}.")

if __name__ == '__main__':
    main()
