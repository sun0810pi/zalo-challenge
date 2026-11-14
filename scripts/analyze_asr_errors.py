import sys
import json
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
cache_path = Path(__file__).resolve().parents[1] / "cache" / "asr_public_test.json"
with cache_path.open("r", encoding="utf-8") as f:
    data = json.load(f)
print(f'Total transcripts: {len(data)}')

patterns = {
    'phim/rap': re.compile(r'\b(?:nạp|lạp)\s+(?:chiếu\s+)?phim\b', re.I),
    'trien lam': re.compile(r'\b(?:chuyện|tiển|tiễn|chuyển)\s+(?:làm|năm|lãm)\b', re.I),
    'huy': re.compile(r'\b(?:huyết|hủ)\s+(?:họp|cuộc\s+họp|đơn|vé|báo\s+thức)\b', re.I),
    'xang 95': re.compile(r'\bxăng\s+(?:chín\s+lắm|chín\s+năm)\b', re.I),
    'danh ba': re.compile(r'\bdanh\s+(?:bạc|mạng)\b', re.I),
    'doi tac': re.compile(r'\bđói\s+tá\b', re.I),
    'ha canh': re.compile(r'\bhạ\s+canh\b', re.I),
    'danh sach': re.compile(r'\b(?:cho\s+anh\s+sách|danh\s+sắt)\b', re.I),
    'chuyen bay/tau': re.compile(r'\b(?:chuyện|chuyện\s+bay)\s+(?:rẻ|nhất|sớm|nào)\b', re.I),
    'tien nuoc/dien': re.compile(r'\b(?:tiền\s+nửa|trường\s+điện|đọng\s+hồ)\b', re.I),
    'xe 4 cho': re.compile(r'\bxe\s+(?:vốn|vốn\s+chỗ)\b', re.I),
    'mien trung (ni, mo, hu, gio)': re.compile(r'\b(?:chiều\s+ni|hôm\s+ni|tập\s+gym\s+mô|hủ\s+đơn|giờ\s+muộn)\b', re.I)
}

for name, pat in patterns.items():
    matches = [(k, v) for k, v in data.items() if pat.search(v)]
    print(f'Pattern "{name}": {len(matches)} matches')
    for k, v in matches[:3]:
        print(f'   {k}: {v}')
