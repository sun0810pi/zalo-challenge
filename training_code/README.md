# Training Code & Reproducibility Guide (ZAC 2026 — LifeMate)

Thư mục này chứa toàn bộ mã nguồn huấn luyện, cấu hình siêu tham số, và script tạo dữ liệu để tái lập (reproduce) quá trình huấn luyện mô hình cho cuộc thi **Zalo AI Challenge 2026 — LifeMate**.

> Lưu ý theo hướng dẫn BTC: Base model dùng trong quá trình training **không** được đóng gói bên trong Docker image (chỉ đưa weights đã fine-tune dùng cho inference vào `saved_models/`). BTC chạy reproduce có kết nối internet.

## 1. Cấu trúc mã nguồn training

- `train_sft.py`: Huấn luyện LoRA/QLoRA SFT cho `Qwen/Qwen3-4B` với loss masking chỉ tính trên lượt assistant (`<think>` + `<tool_call>`), cố định seed (`42`).
- `train_grpo_rlvr.py`: Tinh chỉnh tăng cường RLVR bằng thuật toán GRPO trực tiếp trên `zaic_tools.ToolEnv` (`R = 0.5 * StateMatch + 0.5 * FullToolCallF1`).
- `merge_whisper_lora.py`: Gộp adapter Whisper LoRA vào base Whisper thành standalone model.
- `scripts/merge_qwen_lora.py`: Gộp adapter Qwen3-4B LoRA vào base `Qwen/Qwen3-4B` để phục vụ inference offline trong Docker.
- `scripts/data/`: Các script xây dựng dữ liệu SFT từ tập `train/` của BTC (`generate_reasoning_sft.py`, `build_v5_sft_dataset.py`, `build_rsft_dataset.py`, v.v.).
- `configs/`: Các file JSON cấu hình siêu tham số cho SFT và GRPO (`train_sft_v5_expanded.json`, `train_grpo_rlvr.json`, v.v.).
- `src/`: Thư viện lõi (prompt builder, schema guardrail, semantic router, entity normalizer, hybrid engine, ASR cleaner).

## 2. Nguồn tải Base Models & Dữ liệu

### Base Models (Public Open-Source, phát hành trước 26/10/2026)
- **LLM Base (`< 5B`, 4.0B params)**: `Qwen/Qwen3-4B` — https://huggingface.co/Qwen/Qwen3-4B
- **ASR Base (`~0.81B` params)**:
  - HuggingFace Transformers / LoRA base: `openai/whisper-large-v3-turbo` — https://huggingface.co/openai/whisper-large-v3-turbo
  - CTranslate2 inference base: `deepdml/faster-whisper-large-v3-turbo-ct2` — https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2

### Dữ liệu huấn luyện & Weights đã fine-tune
- Dữ liệu gốc: bộ `train/` do BTC Zalo AI Challenge 2026 cung cấp.
- Synthetic / processed SFT data & fine-tuned weights (nếu upload lên HuggingFace trước hạn nộp 16/11/2026, điền URL cố định tại đây trước khi đóng gói Docker):
  - Training Data URL: `<Điền URL HuggingFace / Google Drive của đội nếu có>`
  - Fine-tuned LLM / LoRA URL: `<Điền URL HuggingFace / Google Drive của đội nếu có>`
  - Fine-tuned ASR / LoRA URL: `<Điền URL HuggingFace / Google Drive của đội nếu có>`

## 3. Quy trình tái lập (Reproducibility)

Cố định seed `42` trong toàn bộ quá trình chuẩn bị dữ liệu, training và inference.

```bash
# Bước 1: Tạo dữ liệu SFT từ tập train/ của BTC
python3 scripts/data/generate_reasoning_sft.py
python3 scripts/data/build_v5_sft_dataset.py

# Bước 2: Huấn luyện SFT LoRA trên Qwen/Qwen3-4B (seed=42)
python3 train_sft.py \
    --model Qwen/Qwen3-4B \
    --data-dir data/sft/candidates/v5_expanded \
    --output-dir checkpoints/qwen3-4b-zac-lora-v5-expanded \
    --epochs 3 --batch-size 2 --grad-accum 8 --lr 2e-4 \
    --max-seq-len 8192 --lora-r 32 --lora-alpha 64 --weighted-loss --optimized

# Bước 3 (Tùy chọn): Tinh chỉnh GRPO-RLVR trên ToolEnv
python3 train_grpo_rlvr.py \
    --model Qwen/Qwen3-4B \
    --lora-path checkpoints/qwen3-4b-zac-lora-v5-expanded \
    --output-dir checkpoints/qwen3-4b-zac-lora-grpo-rlvr

# Bước 4: Gộp LoRA thành standalone model để đưa vào /code/saved_models/llm/
python3 scripts/merge_qwen_lora.py \
    --base-model Qwen/Qwen3-4B \
    --lora-path checkpoints/qwen3-4b-zac-lora-v5-expanded \
    --output-dir saved_models/llm
```
