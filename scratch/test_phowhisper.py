import sys, time, wave, numpy as np
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from transformers import pipeline

def load_audio(p):
    with wave.open(p, 'rb') as w:
        frames = w.readframes(w.getnframes())
        return {'raw': np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, 'sampling_rate': w.getframerate()}

print('Loading PhoWhisper-base from local directory...')
t0 = time.time()
pipe = pipeline('automatic-speech-recognition', model='models/phowhisper-base', device='cpu')
print(f'Loaded in {time.time()-t0:.2f}s')

test_files = [
    'public_test/audio/public_0002/public_0002.wav',
    'public_test/audio/public_0003/public_0003.wav',
    'public_test/audio/public_0004/public_0004.wav',
    'public_test/audio/public_0005/public_0005.wav',
    'public_test/audio/public_0007/public_0007.wav',
    'public_test/audio/public_0008/public_0008.wav',
    'public_test/audio/public_0010/public_0010.wav',
    'public_test/audio/public_0012/public_0012.wav',
    'public_test/audio/public_0213/public_0213.wav',
]

for p in test_files:
    t1 = time.time()
    res = pipe(load_audio(p))
    print(f"{p} ({time.time()-t1:.2f}s): {res['text']}")
