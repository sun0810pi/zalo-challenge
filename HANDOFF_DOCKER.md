# Hướng dẫn Handoff: Thêm Model, Build Docker & Nộp bài ZAC 2026 LifeMate

Tài liệu này dành cho thành viên phụ trách **đóng gói model + build Docker + test offline + nộp bài** cho track **LifeMate — Zalo AI Challenge 2026**.

Toàn bộ code inference, notebook, script Docker và cấu trúc `/code/` theo đúng tài liệu `Docker_Guide_ZAC2026_LifeMate.pdf` đã được chuẩn bị sẵn trong repo. Em chỉ cần bỏ model LM & Voice vào thư mục `saved_models/`, build image, test `--network none`, xuất `.tar.gz` và nộp.

---

## 0. Quy định & Giới hạn bắt buộc của BTC

- **Deadline nộp**: `23:59 GMT+7 ngày 16/11/2026` (tối đa **2 lần nộp** trên portal `https://challenge.zalo.ai/portal/lifemate`).
- **Phần cứng chấm bài**: 1x NVIDIA RTX 5090 (32GB VRAM), driver `595.58.03`, base image `nvidia/cuda:13.2.0-devel-ubuntu22.04`.
- **Giới hạn model**:
  - LLM `< 5B` tham số (mặc định `Qwen/Qwen3-4B` + LoRA đã merge).
  - Tổng tham số toàn pipeline `< 6B` (mặc định `Qwen3-4B` ~4.0B + `whisper-large-v3-turbo` ~0.8B = ~4.8B).
  - Context length tối đa `16384`.
- **Môi trường chấm bài**: **100% Offline (`--network none`)**, thời gian chạy `<= 30s/mẫu`.
- **Cấu trúc trong container (`/code/`)**:
  - `/code/predict.py`, `/code/predict.sh` -> chạy suy luận trên `/data/<split>`, ghi ra `/result/submission.jsonl` (mỗi dòng có `id`, `tool_calls`, `state_hash`, `time` tính bằng mili-giây).
  - `/code/start_jupyter.sh`, `/code/predict_notebook.ipynb` -> BTC có thể chạy đối chiếu qua JupyterLab (port `9777`, token `zac2026`), ghi ra `/result/jupyter_submission.jsonl`.
  - `/code/training_code/` -> chứa sẵn toàn bộ code train SFT / GRPO / merge LoRA + `training_code/README.md`.

---

## 1. Cấu trúc sườn đã dựng sẵn

```text
zalo-challenge/
├── saved_models/
│   ├── llm/                  # <-- BỎ MODEL LLM (Qwen3-4B / LoRA đã merge) VÀO ĐÂY
│   └── asr/                  # <-- BỎ MODEL VOICE (Whisper / LoRA đã merge) VÀO ĐÂY
├── scripts/
│   └── prepare_saved_models.py   # Script hỗ trợ tải / merge LoRA / kiểm tra model
├── src/                      # Core agent, ASR (Whisper), guardrail, fallback
├── training_code/            # Code train + configs (Dockerfile sẽ copy vào /code/training_code/)
├── predict.py                # Entrypoint suy luận offline
├── predict.sh                # Script BTC gọi khi chấm tự động
├── start_jupyter.sh          # Script BTC gọi để bật JupyterLab (token zac2026)
├── predict_notebook.ipynb    # Notebook 4 cell ghi /result/jupyter_submission.jsonl
├── Dockerfile                # Base nvidia/cuda:13.2.0-devel-ubuntu22.04
├── .dockerignore             # Loại trừ rác / dataset khi docker build
└── requirements.txt          # Dependencies (bao gồm jupyterlab>=4.0.0)
```

> **Lưu ý**: `predict.py` tự động dò tìm model theo thứ tự ưu tiên: tham số CLI / biến môi trường (`MODEL_PATH`, `ASR_PATH`) -> `saved_models/llm` & `saved_models/asr` (bỏ qua file `.gitkeep`). Nên chỉ cần đặt đúng weights vào 2 thư mục trên là không cần sửa code.

---

## Bước 1: Chuẩn bị máy build Linux + GPU + Docker

Kiểm tra máy build có đủ NVIDIA Driver, Docker và NVIDIA Container Toolkit:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:13.2.0-devel-ubuntu22.04 nvidia-smi
```

Clone repo và cài Python dependencies (nếu cần chạy `scripts/prepare_saved_models.py` hoặc unit test trên host):

```bash
git clone <URL_REPO_CUA_TEAM>
cd zalo-challenge
pip install -r requirements.txt
```

---

## Bước 2: Thêm Model LM & Voice vào `saved_models/`

Em chọn **1 trong 2 cách** dưới đây để đưa đủ weights vào `saved_models/llm/` và `saved_models/asr/`:

### Cách A: Dùng script chuẩn bị sẵn (`scripts/prepare_saved_models.py`)

- Nếu dùng **base model** (chưa fine-tune):
  ```bash
  python scripts/prepare_saved_models.py \
    --llm-source Qwen/Qwen3-4B \
    --asr-source deepdml/faster-whisper-large-v3-turbo-ct2
  ```
- Nếu đã có **LoRA checkpoint** cho LLM và/hoặc Whisper, script tự merge rồi xuất ra `saved_models/`:
  ```bash
  python scripts/prepare_saved_models.py \
    --llm-source Qwen/Qwen3-4B \
    --llm-lora /duong/dan/toi/qwen_lora_checkpoint \
    --asr-source openai/whisper-large-v3-turbo \
    --asr-lora /duong/dan/toi/whisper_lora_checkpoint
  ```

### Cách B: Copy trực tiếp weights đã merge sẵn

Copy file model vào đúng 2 thư mục sau:
- **`saved_models/llm/`**: cần có `config.json`, tokenizer (`tokenizer.json` / `tokenizer_config.json`) và weights (`*.safetensors` hoặc `*.bin`).
- **`saved_models/asr/`**:
  - Nếu dùng format **CTranslate2 (`faster-whisper`)**: cần có `model.bin` (cùng `config.json`, `tokenizer.json`, `vocabulary.*`).
  - Nếu dùng format **HuggingFace Whisper** (đã merge LoRA): cần có `config.json` và `*.safetensors` / `*.bin` (code `src/asr.py` đã hỗ trợ load offline từ thư mục local).

### Kiểm tra model trước khi build Docker

```bash
python scripts/prepare_saved_models.py --check-only
```

Nếu cả 2 mục `[OK]` là đạt.

> **Nhắc nhỏ**: Nếu team upload checkpoint hoặc dữ liệu train lên HuggingFace / Google Drive, nhớ điền link vào `training_code/README.md` trước khi build để đáp ứng yêu cầu tái lập của BTC.

---

## Bước 3: Chạy Unit Test nhanh & Build Docker Image

Chạy unit test để chắc chắn không có lỗi code hoặc config:

```bash
python -m unittest discover -s tests -v
```

Build Docker image (đặt tag theo tên team, ví dụ `zac2026-lifemate:v1`):

```bash
docker build -t zac2026-lifemate:v1 .
```

> Lưu ý: Nhờ `.dockerignore`, các thư mục `train/`, `public_test/`, `checkpoints/` sẽ không bị đóng gói nhầm vào image, nhưng `saved_models/` **được giữ lại** để chạy offline trong container.

---

## Bước 4: Test chạy Offline (`--network none`) giống hệt server BTC

Chuẩn bị một thư mục dataset test (ví dụ `/abs/path/to/public_test`, bên trong có `samples.jsonl`, `audio/` hoặc `audios/`, và `zaic_tools/` nếu có) và thư mục output `/abs/path/to/result_test`.

### 4.1. Test chạy tự động `/code/predict.sh`

```bash
mkdir -p /abs/path/to/result_test

docker run --rm --gpus all --network none \
  -v /abs/path/to/public_test:/data/public_test:ro \
  -v /abs/path/to/result_test:/result \
  zac2026-lifemate:v1 /bin/bash /code/predict.sh
```

Kiểm tra kết quả tại `/abs/path/to/result_test/submission.jsonl`:
- File phải tồn tại, không rỗng.
- Mỗi dòng JSON hợp lệ gồm đủ: `id`, `tool_calls`, `state_hash`, `time` (số mili-giây).
- Không có lỗi kết nối mạng / thiếu file model trong log.

### 4.2. Test chạy đối chiếu JupyterLab `/code/start_jupyter.sh`

```bash
docker run --rm -d --name zac-jupyter --gpus all --network none \
  -p 9777:9777 \
  -v /abs/path/to/public_test:/data/public_test:ro \
  -v /abs/path/to/result_test:/result \
  zac2026-lifemate:v1 /bin/bash /code/start_jupyter.sh
```

1. Mở trình duyệt vào `http://localhost:9777` và nhập token: `zac2026`.
2. Mở notebook `predict_notebook.ipynb`, chọn **Run -> Run All Cells**.
3. Kiểm tra file `/abs/path/to/result_test/jupyter_submission.jsonl` được tạo thành công và khớp format với `submission.jsonl`.
4. Dừng container test:
   ```bash
   docker stop zac-jupyter
   ```

---

## Bước 5: Đóng gói `.tar.gz`, lấy MD5 & Nộp bài

1. **Xuất Docker image thành `.tar.gz`** (thay `<TeamName>` bằng tên team của mình):
   ```bash
   docker save zac2026-lifemate:v1 | gzip > zac2026_<TeamName>.tar.gz
   ```

2. **Tính mã MD5**:
   ```bash
   md5sum zac2026_<TeamName>.tar.gz
   ```
   Lưu lại chuỗi MD5 vừa in ra.

3. **Upload lên Google Drive**:
   - Upload file `zac2026_<TeamName>.tar.gz` lên Google Drive.
   - Bật quyền chia sẻ: **Anyone with the link** (Viewer).

4. **Nộp bài trên Portal BTC**:
   - Vào `https://challenge.zalo.ai/portal/lifemate`.
   - Dán link Google Drive và mã MD5 tương ứng trước **`23:59 GMT+7 ngày 16/11/2026`**.
