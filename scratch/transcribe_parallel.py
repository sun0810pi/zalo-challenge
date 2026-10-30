import sys
import os
import json
import wave
import time
import multiprocessing as mp
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROMPT = (
    "Trợ lý ảo LifeMate tiếng Việt: đặt xe 4 chỗ, 7 chỗ đi khám răng, công ty; "
    "chuyển khoản ngân hàng, ví điện tử, nạp tiền điện thoại, thanh toán hóa đơn tiền điện, tiền nước; "
    "dự báo thời tiết nếu mai mưa thì dời lịch, hủy lịch đá bóng, chạy bộ muộn 1 tiếng, 2 tiếng; "
    "bật tắt đèn phòng khách, phòng ngủ nhỏ, điều hòa, máy lạnh; tạo báo thức, nhắc nhở, gửi tin nhắn SMS."
)

CACHE_FILE = "cache/asr_public_test_small.json"

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

_worker_model = None

def init_worker():
    global _worker_model
    from faster_whisper import WhisperModel
    # Each of 4 workers uses 3 CPU threads (12 threads total out of 14)
    _worker_model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=3)

def transcribe_one(item):
    qid, rel_audio = item
    wav_path = os.path.join("public_test", rel_audio)
    try:
        audio = read_wav(wav_path)
        segs, _ = _worker_model.transcribe(audio, language='vi', initial_prompt=PROMPT, beam_size=1)
        text = " ".join(s.text.strip() for s in segs)
        return qid, rel_audio, text, None
    except Exception as e:
        return qid, rel_audio, "", str(e)

def main():
    os.makedirs("cache", exist_ok=True)
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE, "r", encoding="utf-8"))
        except Exception:
            cache = {}

    samples = [json.loads(line) for line in open("public_test/public_test.jsonl", encoding="utf-8") if line.strip()]
    to_process = []
    for s in samples:
        qid = s["id"]
        rel_audio = s["audio"]
        if rel_audio not in cache:
            to_process.append((qid, rel_audio))

    print(f"Total samples: {len(samples)} | Already cached: {len(cache)} | To transcribe: {len(to_process)}")
    if not to_process:
        print("All samples already transcribed!")
        return

    num_workers = 4
    print(f"Starting {num_workers} parallel workers on 14-core CPU...")
    t0 = time.time()
    completed = 0

    with mp.Pool(num_workers, initializer=init_worker) as pool:
        for qid, rel_audio, text, err in pool.imap_unordered(transcribe_one, to_process, chunksize=1):
            completed += 1
            if err:
                print(f"[{completed}/{len(to_process)}] ERROR on {qid}: {err}")
            else:
                cache[rel_audio] = text
                if completed % 10 == 0 or completed == len(to_process):
                    elapsed = time.time() - t0
                    speed = completed / elapsed
                    remaining = (len(to_process) - completed) / speed if speed > 0 else 0
                    print(f"[{completed:3d}/{len(to_process)}] ({speed:.2f} items/s, ~{remaining/60:.1f}m left) {qid}: {text[:60]}")
                    with open(CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\nSuccessfully transcribed {len(to_process)} samples in {time.time()-t0:.1f}s. Saved to {CACHE_FILE}.")

if __name__ == '__main__':
    main()
