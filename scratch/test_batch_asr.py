import sys
import os
import json
import wave
import time
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from faster_whisper import WhisperModel

PROMPT = (
    "Trợ lý ảo LifeMate tiếng Việt: đặt xe 4 chỗ, 7 chỗ đi khám răng, họp phụ huynh, công ty; "
    "chuyển khoản ngân hàng, ví điện tử nạp tiền điện thoại, thanh toán hóa đơn điện nước internet; "
    "dự báo thời tiết nếu mai mưa thì dời lịch, hủy lịch đá bóng, chạy bộ; "
    "bật tắt đèn phòng ngủ, điều hòa, máy lạnh; tạo báo thức, nhắc nhở, gửi tin nhắn SMS."
)

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

def main():
    print("Loading faster-whisper small on CPU (8 threads)...")
    model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=8)
    
    test_lines = [json.loads(l) for l in open('public_test/public_test.jsonl', encoding='utf-8')][:10]
    for row in test_lines:
        qid = row['id']
        wav_path = os.path.join('public_test', row['audio'])
        t0 = time.time()
        audio = read_wav(wav_path)
        segs, _ = model.transcribe(audio, language='vi', initial_prompt=PROMPT, beam_size=2)
        text = " ".join(s.text.strip() for s in segs)
        dt = time.time() - t0
        print(f"[{qid}] ({dt:.2f}s): {text}")

if __name__ == '__main__':
    main()
