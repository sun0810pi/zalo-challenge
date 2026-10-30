import sys, os, json, time, wave, multiprocessing as mp, numpy as np
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

def read_wav(p):
    with wave.open(p, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

def process_job_list(job_indices, worker_id):
    from faster_whisper import WhisperModel
    print(f"[Worker {worker_id}] Loading large-v3-turbo...")
    t0 = time.time()
    model = WhisperModel('models/whisper-large-v3-turbo', device='cpu', compute_type='int8', cpu_threads=4)
    print(f"[Worker {worker_id}] Model loaded in {time.time()-t0:.1f}s. Handling {len(job_indices)} jobs: {job_indices}")
    
    for j_idx in job_indices:
        job_file = f"cache/gemini_asr/jobs/job_{j_idx}.json"
        done_file = f"cache/gemini_asr/job_{j_idx}_done.json"
        if not os.path.exists(job_file) or os.path.exists(done_file):
            continue
            
        items = json.load(open(job_file, encoding='utf-8'))
        results = {}
        t_job = time.time()
        for idx, (qid, rel_audio) in enumerate(items, 1):
            wav_path = os.path.join("public_test", rel_audio)
            t_s = time.time()
            try:
                segs, _ = model.transcribe(read_wav(wav_path), language='vi', beam_size=1)
                txt = " ".join(s.text.strip() for s in segs)
                results[rel_audio] = txt
                print(f"[W{worker_id}] J{j_idx} ({idx}/{len(items)}) {qid} ({time.time()-t_s:.1f}s): {txt[:50]}")
            except Exception as e:
                results[rel_audio] = ""
                print(f"[W{worker_id}] J{j_idx} ERROR {qid}: {e}")
                
        with open(done_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[Worker {worker_id}] Job {j_idx} done in {time.time()-t_job:.1f}s.")

def main():
    remaining_jobs = [j for j in range(1, 26) if not os.path.exists(f"cache/gemini_asr/job_{j}_done.json")]
    print(f"Remaining jobs to process: {remaining_jobs}")
    
    w1_jobs = [j for i, j in enumerate(remaining_jobs) if i % 3 == 0]
    w2_jobs = [j for i, j in enumerate(remaining_jobs) if i % 3 == 1]
    w3_jobs = [j for i, j in enumerate(remaining_jobs) if i % 3 == 2]
    
    p1 = mp.Process(target=process_job_list, args=(w1_jobs, 1))
    p2 = mp.Process(target=process_job_list, args=(w2_jobs, 2))
    p3 = mp.Process(target=process_job_list, args=(w3_jobs, 3))
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()
    print("All remaining jobs completed!")
    
    p1.join()
    p2.join()
    print("All jobs completed!")

if __name__ == '__main__':
    main()
