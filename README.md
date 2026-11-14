
# ZAC 2026 — LifeMate

(Lưu ý: Bộ dữ liệu này chỉ cung cấp cho các đội thi đã đăng ký thi trong khuôn khổ Zalo AI Challenge 2026, các đội thi không chia sẽ phát tán/phân phối bộ dữ liệu này.)

Mô tả về gói dữ liệu, cấu trúc thư mục, định dạng dữ liệu, cách chạy tool và tạo file nộp bài.

## 1. Cấu trúc thư mục

```text
README.md                           tài liệu này
VERSION                             phiên bản gói dữ liệu, ngày build, số mẫu

train/
├── train.jsonl                     mẫu có nhãn: context, history, audio_text, label (tool_calls, tool_results, final_state)
├── train_tools_catalog.json        schema các tool của train (kèm sys)
├── train_environment/<id>/state.json   dữ liệu app của từng mẫu tại thời điểm người dùng gửi yêu cầu
├── audio/<id>/                     <id>.wav: âm thanh lượt cuối; <id>_u1.wav, <id>_u2.wav...: các lượt user trong history
└── zaic_tools/                     thư viện chạy tool local (các app của train)

public_test/
├── public_test.jsonl               chỉ input (không label, không audio_text)
├── public_test_tools_catalog.json  schema các tool của các app của public test
├── public_test_environment/<id>/state.json   dữ liệu app của từng mẫu tại thời điểm người dùng gửi yêu cầu
├── audio/<id>/                     <id>.wav: âm thanh lượt cuối; <id>_u1.wav...: các lượt user trong history
├── sample_submission.jsonl         định dạng file nộp bài với 3 ví dụ mẫu
└── zaic_tools/                     thư viện chạy tool local (các app của public test)

private_test/                       release sau khi nộp final solution (docker)
```

## 2. Định dạng dữ liệu

### 2.1. `<split>.jsonl`

Mỗi dòng là một mẫu (JSON):

- `id`: mã mẫu, vd. `train_0001`, `public_0001`. Dùng làm `id` của dòng nộp và tên thư mục `<split>_environment/<id>/`.
- `audio`: đường dẫn âm thanh yêu cầu cuối của người dùng, vd. `audio/<id>/<id>.wav`. Đây là yêu cầu cần xử lý và được chấm.
- `context`: bối cảnh người dùng tại thời điểm gửi yêu cầu.
  + `current_time`, `timezone`: thời điểm hiện tại, dùng để hiểu "mai", "tối nay"...
  + `location`: vị trí hiện tại (`lat`, `lng`) của người dùng.
  + `user_profile`: tên, địa chỉ nhà, nơi làm việc, thành phố hiện tại của người dùng.
  + `installed_apps`: các app đã cài và danh sách tool của mỗi app (`{app: {"tools": [...]}}`); mỗi mẫu chỉ được gọi các tool này.
- `history`: các lượt hội thoại trước đó (có thể rỗng) của người và agent. Chỉ chấm điểm hành vi của agent với yêu cầu cuối. Mỗi lượt gồm:
  + lượt `user`: `audio` (`audio/<id>/<id>_u1.wav`...).
  + lượt `assistant`: `text`; nếu có gọi tool thì có `tool_calls` (thường kèm `tool_results`). Các thao tác này đã được thực hiện sẵn trong `<split>_environment/<id>/state.json`.

Riêng train có thêm:

- `audio_text`: lời của yêu cầu cuối (lượt `user` trong `history` cũng có `audio_text`).
- `label`: đáp án.
  + `tool_calls`: chuỗi tool call đúng cho yêu cầu cuối (`[]` nếu không cần gọi tool).
  + `tool_results`: kết quả trả về của từng call, `tool_results[i]` ứng với `tool_calls[i]`.
  + `final_state`: state của các app bị thay đổi sau khi chạy xong `tool_calls` (toàn bộ state của app đó); những app không thay đổi dữ liệu thì không có trong `final_state`.

### 2.2. `<split>_tools_catalog.json`

Schema mọi app và tool của split (JSON):

- `schema_version`: phiên bản catalog.
- `note`: chú thích giá trị `kind` của tool.
- `apps`: danh sách app.
  + `id`: mã app, vd. `sms`; `name_vi`: tên tiếng Việt; `group`: nhóm (`home`, `finance`...); `description`: mô tả.
  + `tools`: danh sách tool của app.
    - `name`: tên tool `<app>.<tool>`, vd. `sms.send`; dùng làm `name` trong tool call.
    - `kind`: `read` (chỉ đọc), `write` (thay đổi dữ liệu), `meta` (`sys.ask_missing_info`).
    - `description`: tool làm gì, tham số nào bắt buộc.
    - `parameters`: JSON Schema của `arguments`: `properties` (kiểu, mô tả, `enum`, `minimum`...), `required` (bắt buộc),
      `x-one-of` (bắt buộc một trong các nhóm, vd. `contact_id` hoặc `phone_number`).
      Mở rộng: `x-ref` (giá trị là id lấy từ dữ liệu app khác, vd. `contacts.contacts`), `x-match: fuzzy` (văn bản tự do, chấm gần đúng).
    - `returns`: tóm tắt dữ liệu trả về (có thể rỗng).
    - `app`: app chứa tool.

### 2.3. `<split>_environment/<id>/state.json`

Mỗi mẫu có một file `state.json`: dữ liệu của người dùng trong từng app đã cài, tại thời điểm gửi yêu cầu cuối (các thao tác trong `history` đã được áp dụng sẵn).

- `_meta`: `question_id`, `current_time`, `tools` (các tool được phép gọi).
- `<app>`: dữ liệu của app, vd. `calendar.events`, `contacts.contacts`, `sms.messages`.

Agent không đọc trực tiếp file này mà xem dữ liệu qua tool (vd. `contacts.search`). `zaic_tools` nạp file này khi tạo phiên cho một mẫu; mọi thay đổi chỉ diễn ra trong bộ nhớ, file gốc không bị sửa.

## 3. Chạy tool: `<split>/zaic_tools`

Thư viện Python (chỉ dùng thư viện chuẩn) chạy tool trên `state.json` của từng mẫu. Bài nộp được chấm bằng đúng thư viện này, nên kết quả chạy local giống khi chấm.

```python
env = ToolEnv("public_test", "public_0001", root=".")     # một phiên mới cho một mẫu
env.call("smart_home.get_status", {})                     # chạy tool -> {"ok": true, ...} hoặc {"ok": false, "error": ...}
env.validate("smart_home.light_set", {"power": "off"})   # chỉ kiểm tra schema, không chạy -> {"ok": false, "error": "thiếu tham số bắt buộc: device_id"}
env.get_tool_calls()                                      # mọi call đã gọi (kể cả call bị từ chối)
env.get_state(), env.get_state_hash()                     # state hiện tại và hash của nó (state_hash của dòng nộp)
```

- Mỗi mẫu dùng một `ToolEnv` riêng và chỉ được tạo một lần: mọi call đã gọi đều tính vào bài nộp, không được tạo lại `ToolEnv` để thử rồi làm lại từ state ban đầu.
- Call sai schema (tool không có / chưa cài, thiếu tham số, sai kiểu...) bị từ chối (`{"ok": false, "error": ...}`) và không đổi state.
- Dòng lệnh (chạy trong thư mục `public_test/`): `python -m zaic_tools --split public_test --root .. --apps` (liệt kê app/tool), `--validate submission.jsonl` thay cho `--apps` (kiểm tra schema cả file nộp).

## 4. Tạo file kết quả dự đoán của agent

File `submission.jsonl` chứa kết quả dự đoán của agent trên một split:

- Là file nộp bài: BTC dùng file này để chấm điểm và hiển thị kết quả trên leaderboard.
- Khi BTC chạy Docker của đội (vòng private), Docker cũng phải tạo ra đúng file này. Hướng dẫn đóng gói Docker sẽ có tài liệu riêng.

Định dạng: UTF-8 JSON Lines, đúng một dòng cho mỗi mẫu của split, mỗi dòng gồm:

- `id`: `id` của mẫu.
- `tool_calls`: mọi tool call agent đã gọi cho yêu cầu cuối, `[{"name": ..., "arguments": {...}}, ...]` (`env.get_tool_calls()`); `[]` nếu không gọi tool nào.
- `state_hash`: hash của state cuối (`env.get_state_hash()`).

Xem `sample_submission.jsonl` (3 dòng ví dụ: làm được / thiếu thông tin / ngoài phạm vi).

Ví dụ chạy agent trên public_test và tạo `submission.jsonl` (chạy từ thư mục chứa `public_test/`):

```python
import json
import sys
from pathlib import Path

ROOT = Path(".")                                      # thư mục chứa public_test/ (chạy từ thư mục này)
SPLIT = "public_test"
sys.path.insert(0, str(ROOT / SPLIT))                  # zaic_tools của split
from zaic_tools import ToolEnv  # noqa: E402


def infer(item):
    env = ToolEnv(SPLIT, item["id"], root=ROOT)        # một phiên độc lập cho mỗi mẫu, nạp <split>_environment/<id>/state.json + catalog
    # Pipeline của bạn: ASR (ROOT / SPLIT / item["audio"]) -> LLM -> env.call(tool, arguments) -> LLM -> ...
    # env.validate(tool, arguments)                    # chỉ kiểm schema, không chạy -> {"ok": true/false, "error": ...}
    # env.call("smart_home.get_status", {})            # -> {"ok": true, ...} hoặc {"ok": false, "error": "..."}; state cập nhật trong bộ nhớ
    # env.call("smart_home.light_set", {"device_id": "light_kitchen", "power": "off"})
    return env.get_tool_calls(), env.get_state_hash()  # mọi call đã gọi (kể cả call bị từ chối) và hash của state cuối


def main():
    n = 0
    with open(ROOT / SPLIT / f"{SPLIT}.jsonl", encoding="utf-8") as fin, open("submission.jsonl", "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            tool_calls, state_hash = infer(item)
            fout.write(json.dumps({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} dòng -> submission.jsonl")


if __name__ == "__main__":
    main()
```


## 5. Cấu trúc Source Code & Đóng gói Docker (Chuẩn BTC ZAC 2026)

### 5.1. Phân cấp Thư mục (`/code/` trong Docker)
- `predict.py`: Entrypoint inference offline chuẩn BTC — tự nhận diện split dưới `/data/<split>/`, nạp `zaic_tools` trực tiếp từ `/data/<split>/`, cố định seed (`42`), đo thời gian từng mẫu và xuất `/result/submission.jsonl` (gồm `id`, `tool_calls`, `state_hash`, `time`).
- `predict.sh`: Script chạy inference (`/code/predict.sh`) với cờ offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- `start_jupyter.sh`: Khởi động JupyterLab tại port `9777` (token/password `zac2026`).
- `predict_notebook.ipynb`: Notebook 4 bước chuẩn Phụ lục Mục 7 để BTC đo thời gian inference và xuất `/result/jupyter_submission.jsonl` (khớp 100% logic với `predict.py`).
- `saved_models/`: Thư mục chứa trọng số đã fine-tune/đóng gói sẵn cho inference offline:
  - `saved_models/llm/` (hoặc `models/qwen3-4b-merged/`): Model LLM đã gộp trọng số (`< 5B` params).
  - `saved_models/asr/` (hoặc `models/whisper-large-v3-turbo/`): Model ASR (`whisper-large-v3-turbo`).
- `training_code/`: Toàn bộ mã nguồn huấn luyện (`train_sft.py`, `train_grpo_rlvr.py`, `merge_whisper_lora.py`, `configs/`, `scripts/`, `README.md`).
- `src/`: Thư viện lõi chứa logic ReAct agent, semantic tool router, pure-schema state guardrail, entity normalizer, hybrid engine, action triage và ASR cleaner.
- `run_infer.py`, `run_infer_batched.py`, `evaluate.py`: Công cụ phát triển và chấm điểm local trên `train/` và `public_test/`.

### 5.2. Mô tả giải pháp (Theo mục 6.1 Docker Guide)
- **ASR Model**: `deepdml/faster-whisper-large-v3-turbo-ct2` / `openai/whisper-large-v3-turbo` (~0.81B tham số), decode deterministic (`temperature=0.0`, `beam_size=5`) kết hợp hotwords theo ngữ cảnh `user_profile` + `installed_apps` và bộ lọc chuẩn hóa lỗi ngữ âm tiếng Việt (`src/asr.py`, `SafeContextPreprocessor`).
- **LLM Model**: `Qwen/Qwen3-4B` (~4.0B tham số, `< 5B`), `max_model_len=16384` tokens (`<= 16K`), tổng tham số toàn hệ thống ~4.81B (`< 6B`), inference deterministic (`temperature=0.0`, greedy decoding) qua vLLM / Transformers.
- **Vòng lặp Agent & Tool Calling**:
  - Multi-turn ReAct (tối đa 8 bước/mẫu) chỉ tương tác qua `zaic_tools.ToolEnv` (`validate`, `call`, `get_tool_calls`, `get_state_hash`), không đọc trực tiếp `state.json`.
  - Tiền xử lý an toàn (`SafeContextPreprocessor`), định tuyến công cụ hai tầng (`SemanticToolRouter`), bảo vệ schema & trạng thái (`PureSchemaStateTracker`), chuẩn hóa thực thể mờ (`EntityNormalizer`), kiểm tra logic điều kiện/tính toán (`HybridProgramAidedEngine`), và chắn ảo giác/ngoài phạm vi (`ActionTriageBarrier`).
- **Dữ liệu & Huấn luyện**: SFT LoRA (có weighted assistant-only loss trên `<think>` + `<tool_call>`) và tùy chọn RLVR (GRPO trực tiếp trên `ToolEnv`) từ tập `train/` của BTC và dữ liệu tổng hợp (`scripts/data/`). Chi tiết tái lập xem tại `training_code/README.md`.

### 5.3. Lệnh Build, Kiểm tra & Xuất nộp Docker
```bash
# 1. Đặt weights đã merge (chỉ weights inference, không chứa base model train thừa) vào saved_models/llm/ và saved_models/asr/
# 2. Build Docker image (CUDA 13.2)
docker build -t zac2026-teamname:v1 .

# 3. Kiểm tra chạy offline giống server BTC (thay đường dẫn data/result tương ứng)
docker run --rm --gpus '"device=0"' --network none \
  -v /path/to/data:/data:ro \
  -v /path/to/result:/result \
  zac2026-teamname:v1 /bin/bash /code/predict.sh

# 4. Kiểm tra JupyterLab (http://localhost:9777, nhập token/mật khẩu: zac2026, Run All Cells predict_notebook.ipynb)
docker run --rm -it --gpus '"device=0"' --network none -p 9777:9777 \
  -v /path/to/data:/data:ro \
  -v /path/to/result:/result \
  zac2026-teamname:v1 /bin/bash /code/start_jupyter.sh

# 5. Xuất file .tar.gz và lấy MD5 checksum để nộp Google Drive + Portal
docker save zac2026-teamname:v1 | gzip > zac2026_TeamName.tar.gz
md5sum zac2026_TeamName.tar.gz
```

