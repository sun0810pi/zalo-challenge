import sys, json, os, wave, time, numpy as np
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from faster_whisper import WhisperModel

PROMPT = ('Trợ lý ảo LifeMate tiếng Việt: đặt xe 4 chỗ, 7 chỗ đi khám răng, công ty, chuyển khoản ngân hàng, ví điện tử, thời tiết nếu mai mưa thì dời lịch, hủy lịch đá bóng, chạy bộ, báo thức, nhắc nhở, đèn, điều hòa.')
def read_wav(p):
    with wave.open(p, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=8)
for row in [json.loads(l) for l in open('public_test/public_test.jsonl', encoding='utf-8')][:6]:
    t0 = time.time()
    audio = read_wav(os.path.join('public_test', row['audio']))
    segs, _ = model.transcribe(audio, language='vi', initial_prompt=PROMPT, beam_size=1)
    print(f"[{row['id']}] ({time.time()-t0:.2f}s): " + " ".join(s.text.strip() for s in segs))
