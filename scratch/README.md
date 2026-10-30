# Scratch / Exploratory Archive

Thư mục này lưu trữ các script thử nghiệm đơn lẻ, kiểm tra nhanh (ad-hoc tests) và benchmark tạm thời trong các vòng phát triển ban đầu của cuộc thi ZAC 2026.

- Các script kiểm thử ASR nhanh: `test_phowhisper.py`, `test_batch_asr.py`, `test_small_beam1.py`, `transcribe_parallel.py`.
- Các solver thử nghiệm: `multi_clause_solver.py`, `test_combined_solver.py`, `test_hybrid_agent.py`, `test_template_replay.py`.
- Các helper xử lý batch: `prepare_batches.py`, `run_local_worker.py`, `run_parallel_large.py`.

Các pipeline chính thức để huấn luyện và suy luận được duy trì tại:
- `train_sft.py`: Huấn luyện SFT LoRA v3 cho Qwen.
- `run_infer.py`: Suy luận Agent trên public_test / private_test.
- `evaluate.py`: Đánh giá State Hash & Tool Call F1.
- `scripts/`: Các công cụ sinh dữ liệu (`data/`), đo lường (`benchmarks/`), và baseline (`baselines/`).
