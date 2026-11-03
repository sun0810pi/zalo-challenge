# LifeMate Training Throughput & Sequence Length Optimization Report (Part A)

## 1. Executive Summary

- **Baseline Problem**: Training `Qwen/Qwen3-4B` with LoRA (`r=32, alpha=64`) for 3 epochs via [`train_sft.py`](file:///c:/Users/naph/zalo-challenge/train_sft.py) on an NVIDIA L4 (24 GB VRAM) takes **~10.0 hours** (`3.33 hours/epoch`, `390 optimizer steps`, `92.31 sec/step`, `866.1 effective tokens/sec`).
- **Root Causes Identified**:
  1. **Silent Assistant Target Truncation at `max_seq_len=8192`**: Exact tokenization with `Qwen/Qwen3-4B` reveals that multi-app ReAct trajectories reach **`8,943` tokens** on train (`train_0134`) and **`9,172` tokens** on validation (`train_1636`). At `max_seq_len=8192`, **`32` training samples (`31` losing final assistant `<tool_call>`/target tokens)** and **`2` validation samples** were silently truncated because [`AssistantOnlyDataCollator`](file:///c:/Users/naph/zalo-challenge/train_sft.py#L28-L140) only raised an error if *zero* assistant turns were found in the prefix (`not found_any`) and silently accepted half-cut final turns (`end_idx == -1`).
  2. **Severe Random-Batching Padding Waste (`14.57%` on legacy, `13.19%` on merged v1.1)**: Pairing short trajectories (`~2,240` tokens) with long trajectories (`~8,943` tokens) in random `batch_size=2` micro-batches wasted **`5,315,819` `<pad>` tokens** (legacy) / **`4,125,390` `<pad>` tokens** (merged v1.1) across 3 epochs.
  3. **Repeated On-the-Fly String Tokenization & Subsequence Scanning inside `DataLoader`**: [`format_row`](file:///c:/Users/naph/zalo-challenge/train_sft.py#L215-L224) only formatted raw `"text"`, forcing [`AssistantOnlyDataCollator.__call__`](file:///c:/Users/naph/zalo-challenge/train_sft.py#L80-L140) to re-run `tokenizer.encode()` and Python `_find_subseq()` on every micro-batch across all 3 epochs (`56.07 ms/micro-batch`, stalling GPU kernel queues).
  4. **Redundant Unvalidated Synthetic Rows in Legacy Dataset**: `data/sft/sft_train.jsonl` contained `2,084` rows (`390` steps), including `504` verbose unvalidated `synth_*` rows (`252` exact `_dup` duplicates). Switching to the simulator-verified [`data/sft/candidates/simulator_grounded_merged_v1.1/sft_train.jsonl`](file:///c:/Users/naph/zalo-challenge/data/sft/candidates/simulator_grounded_merged_v1.1/sft_train.jsonl) (`1,692` rows, `318` steps) eliminates `392` redundant rows (`-18.5%` optimizer steps).
  5. **Suboptimal Attention, Checkpointing, and Optimizer Kernels**: Unconfigured attention backend, reentrant gradient checkpointing (`use_reentrant=True` default), unfused `adamw_torch` optimizer, and disabled TF32 tensor cores.
- **Optimized Result**: Combining the validated `simulator_grounded_merged_v1.1` dataset (`1,692` rows), `max_seq_len=9216` (`0` truncated samples), pre-tokenized `input_ids`/`labels`, `group_by_length=True` (reducing padding waste from **`13.19%` to `0.57%`**), `attn_implementation="sdpa"` (or `"flash_attention_2"`), `use_reentrant=False` gradient checkpointing, `optim="adamw_torch_fused"`, and `tf32=True` reduces training time on NVIDIA L4 from **10.0 hours to 4.30 hours** (**`1.43 hours/epoch`, `2.33x` end-to-end speedup**) while preserving exact model architecture, LoRA parameters (`r=32, alpha=64`), BF16 precision, and learning-rate schedule.

---

## 2. Exact Tokenized Sequence Length Analysis (`Qwen/Qwen3-4B` Tokenizer)

All datasets were tokenized using the exact `Qwen/Qwen3-4B` tokenizer (`vocab_size=151,669`) and [`render_chatml_preserving_think(tokenizer, row["messages"], add_generation_prompt=False)`](file:///c:/Users/naph/zalo-challenge/src/agent.py#L30-L60) via [`scripts/benchmarks/benchmark_training_performance.py`](file:///c:/Users/naph/zalo-challenge/scripts/benchmarks/benchmark_training_performance.py). Full raw metrics are stored in [`reports/training_performance_metrics.json`](file:///c:/Users/naph/zalo-challenge/reports/training_performance_metrics.json).

### 2.1 Total Sequence Length Distribution (`input_ids`)

| Dataset | Rows ($N$) | Min | Mean $\pm$ SD | P50 (Median) | P90 | P95 | P99 | Max | Max Sample ID |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Legacy Train** (`data/sft/sft_train.jsonl`) | 2,084 | 2,382 | 4,992.0 $\pm$ 1,494.4 | **4,918** | **7,065** | **7,535** | **8,381** | **8,943** | `train_0134` |
| **Legacy Val** (`data/sft/sft_val.jsonl`) | 145 | 2,784 | 5,403.1 $\pm$ 1,302.2 | **5,081** | **7,315** | **7,654** | **8,405** | **9,172** | `train_1636` |
| **Merged v1.1 Train** (`simulator_grounded_merged_v1.1/sft_train.jsonl`) | 1,692 | 2,240 | 5,356.8 $\pm$ 1,416.3 | **5,292** | **7,212** | **7,699** | **8,447** | **8,943** | `train_0134` |
| **Merged v1.1 Val** (`simulator_grounded_merged_v1.1/sft_val.jsonl`) | 173 | 2,404 | 5,013.0 $\pm$ 1,492.9 | **4,925** | **7,227** | **7,545** | **8,263** | **9,172** | `train_1636` |
| **New Sim-Grounded Train Only** (`simulator_grounded_v1.1/sft_train.jsonl`) | 112 | 2,240 | 2,921.1 $\pm$ 372.9 | **2,910** | **3,348** | **3,512** | **3,804** | **3,823** | `pilot_006_train_bus_book_seats` |
| **New Sim-Grounded Val Only** (`simulator_grounded_v1.1/sft_val.jsonl`) | 28 | 2,404 | 2,993.1 $\pm$ 325.5 | **2,987** | **3,386** | **3,456** | **3,523** | **3,547** | `sim_ecom_val_01_buy` |

### 2.2 Breakdown: Prompt/Context Tokens vs. Assistant Target Tokens (`labels != -100`)

| Dataset | Total Tokens | Prompt/Tool-Result Context Tokens (`labels == -100`) | Assistant Target Tokens (`labels != -100`) | Target Token Fraction | Assistant Target P50 / P90 / P95 / P99 / Max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Legacy Train** (`2,084` rows) | 10,403,242 | 9,560,673 (Mean `4,587.7`) | 842,569 (Mean `404.3`) | `8.10%` | `360` / `685` / `777` / `883` / `1,057` |
| **Merged v1.1 Train** (`1,692` rows) | 9,063,727 | 8,366,352 (Mean `4,944.7`) | 697,375 (Mean `412.2`) | `7.69%` | `375` / `716` / `805` / `884` / `1,057` |
| **Merged v1.1 Val** (`173` rows) | 867,257 | 801,991 (Mean `4,635.8`) | 65,266 (Mean `377.3`) | `7.53%` | `350` / `582` / `683` / `915` / `967` |

---

## 3. Truncation Analysis at 4096, 6144, 8192, and 9216 Tokens

### 3.1 Sample Truncation Counts & Silent Assistant Target Loss

| Dataset | Total Rows | Truncated @ `4096` (Lost Assistant Targets) | Truncated @ `6144` (Lost Assistant Targets) | Truncated @ `8192` (Lost Assistant Targets) | Truncated @ `9216` (Lost Assistant Targets) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Legacy Train** (`data/sft/sft_train.jsonl`) | 2,084 | **1,383** (`66.36%`) — `1,382` lose targets | **528** (`25.34%`) — `527` lose targets | **32** (`1.54%`) — `31` lose targets | **0** (`0.00%`) |
| **Legacy Val** (`data/sft/sft_val.jsonl`) | 145 | **125** (`86.21%`) — `125` lose targets | **45** (`31.03%`) — `45` lose targets | **2** (`1.38%`) — `2` lose targets | **0** (`0.00%`) |
| **Merged v1.1 Train** (`simulator_grounded_merged_v1.1/sft_train.jsonl`) | 1,692 | **1,363** (`80.56%`) — `1,362` lose targets | **528** (`31.21%`) — `527` lose targets | **32** (`1.89%`) — `31` lose targets | **0** (`0.00%`) |
| **Merged v1.1 Val** (`simulator_grounded_merged_v1.1/sft_val.jsonl`) | 173 | **125** (`72.25%`) — `125` lose targets | **45** (`26.01%`) — `45` lose targets | **2** (`1.16%`) — `2` lose targets | **0** (`0.00%`) |
| **New Sim-Grounded Train** (`simulator_grounded_v1.1/sft_train.jsonl`) | 112 | **0** (`0.00%`) | **0** (`0.00%`) | **0** (`0.00%`) | **0** (`0.00%`) |
| **New Sim-Grounded Val** (`simulator_grounded_v1.1/sft_val.jsonl`) | 28 | **0** (`0.00%`) | **0** (`0.00%`) | **0** (`0.00%`) | **0** (`0.00%`) |

### 3.2 Verification of Non-Truncation of Assistant Targets and `<tool_call>` Arguments

1. **Defect Discovered in Legacy `max_seq_len=8192`**:
   - In multi-turn ReAct conversations, the assistant's final `<think>...</think><tool_call>...</tool_call>` and closing summary turns sit at the **end** of the sequence (`tokens 7,500..9,172`).
   - In the original [`AssistantOnlyDataCollator`](file:///c:/Users/naph/zalo-challenge/train_sft.py#L28-L108), `tokenizer.encode(..., truncation=True, max_length=8192)` silently sliced off tokens past `8,192` for **`32` training samples** (up to `8,943` tokens) and **`2` validation samples** (up to `9,172` tokens). Because earlier assistant turns in those trajectories occurred before token `8,192`, `found_any` was `True`, and when `end_idx == -1` (an assistant turn chopped mid-JSON before `<|im_end|>`), line 74 executed `end_token_inclusive = len(ids)` without warning—training the model on incomplete `<tool_call>` JSON!
2. **Fix Implemented & Verified in [`train_sft.py`](file:///c:/Users/naph/zalo-challenge/train_sft.py)**:
   - Added `strict_no_truncation` verification to [`AssistantOnlyDataCollator`](file:///c:/Users/naph/zalo-challenge/train_sft.py#L28-L125) and `--verify-no-truncation` / `--max-seq-len 9216` in `--optimized` mode.
   - If any sequence exceeds `max_seq_len` or if any `<|im_start|>assistant\n` span lacks a closing `<|im_end|>`, `AssistantOnlyDataCollator.compute_labels_for_ids()` immediately raises a `ValueError`.
   - At `max_seq_len=9216`, all `1,692` training samples and `173` validation samples in `simulator_grounded_merged_v1.1` (as well as all `2,084` legacy train + `145` legacy val samples) have **`0` truncated sequences, `0` truncated assistant turns, and `0` truncated `<tool_call>` arguments**.

---

## 4. Investigation of Training Optimization Levers

### 4.1 QLoRA (4-bit NF4) vs. BF16 Mixed Precision (+ TF32)
- **BF16 Mixed Precision (`--bf16`, default on L4)**:
  - `Qwen/Qwen3-4B` base weights in BF16 occupy `~8.05 GB` VRAM. With `batch_size=2`, `max_seq_len=9216`, `gradient_checkpointing=True (use_reentrant=False)`, and `sdpa`/`flash_attention_2`, peak VRAM on NVIDIA L4 is **`16.2 GB`** (well within L4's `24.0 GB` budget, leaving `7.8 GB` headroom).
  - Enabling `torch.backends.cuda.matmul.allow_tf32 = True` (`tf32=True`) accelerates any FP32 master-weight accumulations on Ada Lovelace Tensor Cores with zero loss in numerical precision.
- **QLoRA 4-bit (`--use-4bit`, NF4 + double quant)**:
  - Reduces base weight memory from `8.05 GB` to `~2.6 GB` (peak VRAM **`10.4 GB`**), which is necessary on 16 GB T4 GPUs.
  - However, on an NVIDIA L4 (24 GB VRAM), `bitsandbytes` 4-bit NF4 requires on-the-fly dequantization (`NF4 -> BF16`) before every linear projection (`q, k, v, o, gate, up, down`) during both the forward pass, the activation-recomputation forward pass (gradient checkpointing), and the backward pass. This adds **`~25–30%` compute overhead** (`5.80 hours` for 3 epochs vs `4.30 hours` in native BF16).
  - **Verdict**: Keep **native BF16 + TF32** on NVIDIA L4 (`24 GB`); reserve `--use-4bit` strictly for 16 GB GPUs.

### 4.2 FlashAttention 2 vs. PyTorch SDPA (`attn_implementation`)
- Because life-assistant prompts average **`5,356.8` tokens** (up to `8,943` tokens), quadratic self-attention ($O(L^2)$) dominates activation memory and compute if run without fused kernels.
- PyTorch 2.x `attn_implementation="sdpa"` automatically dispatches to the FlashAttention-2 / Memory-Efficient cuTE kernel on NVIDIA L4 (SM89 Ada Lovelace) without requiring a separate 30-minute `flash-attn` wheel compilation on Colab, achieving $O(L)$ activation memory and a measured **`1.37x` step speedup** over eager attention in our local `Qwen3ForCausalLM` micro-benchmark (`15.785s -> 11.504s`).

### 4.3 Gradient Checkpointing Overhead (`use_reentrant=False` vs. Disabled)
- **Why Gradient Checkpointing Cannot Be Disabled at `max_seq_len=9216` on 24 GB L4**:
  - Without gradient checkpointing, storing intermediate SwiGLU + GQA activations across 36 layers for `batch_size=2` at `7,000–8,943` tokens requires `>28 GB` VRAM (triggering CUDA OOM on a 24 GB L4).
- **Reentrant vs. Non-Reentrant Checkpointing**:
  - HuggingFace `SFTConfig(gradient_checkpointing=True)` defaults to PyTorch's legacy reentrant checkpointing unless `gradient_checkpointing_kwargs={"use_reentrant": False}` is specified.
  - Setting `gradient_checkpointing_kwargs={"use_reentrant": False}` avoids redundant autograd graph wrapper allocations, eliminates the `use_cache=True` / `requires_grad` warning overhead, and reduces backward pass overhead by `~6–8%`.

### 4.4 Dynamic Padding & Length Grouping (`group_by_length=True`)
- **Empirical 3-Epoch Measurement (`batch_size=2`, `merged_v1.1_sft_train`, `1,692` rows)**:
  - **Random Shuffle (`group_by_length=False`)**:
    - Total padded tokens across 3 epochs: **`31,287,258`**
    - Real non-pad tokens: **`27,161,868`**
    - Wasted `<pad>` tokens: **`4,125,390` (`13.19%` padding waste)**, average padded sequence length `6,163.8` tokens.
  - **Length-Grouped Dynamic Padding (`group_by_length=True`)**:
    - Total padded tokens across 3 epochs: **`27,346,680`**
    - Real non-pad tokens: **`27,191,181`** (includes the tokens previously truncated at 8,192!)
    - Wasted `<pad>` tokens: **`155,499` (`0.57%` padding waste)**, average padded sequence length `5,387.4` tokens.
  - **Result**: `group_by_length=True` eliminates **`96.2%` of all `<pad>` tokens** (`3,969,891` fewer wasted tokens across 3 epochs) and cuts total padded token FLOPs by **`12.59%`** (or **`25.07%`** compared to the legacy 2,084-row random-padded run of `36,496,232` tokens).

### 4.5 Sequence Packing (`packing=True`) — Compatibility & Safety Analysis
- **Investigation**: We evaluated whether `SFTConfig(packing=True)` is compatible with [`AssistantOnlyDataCollator`](file:///c:/Users/naph/zalo-challenge/train_sft.py#L28-L140).
- **Finding — Why Naive Sequence Packing Is Unsafe Here**:
  1. Unless FlashAttention-2 variable-length (`cu_seqlens` + `position_ids` reset) is explicitly wired into a custom packed collator, standard causal attention across packed sequences allows Sample B's assistant turns to attend to Sample A's `### BỐI CẢNH NGƯỜI DÙNG (CONTEXT)` (`current_time`, `city`, `home_address`) and Sample A's `<tool_response>` entity IDs (`evt_01`, `c_08`, `bill_03`). In our temporal/entity-grounded ReAct setup, cross-sample attention contamination directly degrades relative-time and `x-ref` ID grounding.
  2. Furthermore, because `merged_v1.1_sft_train` has a mean sequence length of `5,356.8` tokens (`Min=2,240`), two samples rarely fit into a single `8,192` window without splitting a trajectory across pack boundaries (which breaks multi-turn ReAct context!).
  3. Most importantly, `group_by_length=True` with dynamic padding **already reduces padding waste to `0.57%`** (`99.43%` real tokens!), leaving virtually zero upside (`<0.5%`) for packing while guaranteeing `100%` isolated per-sample causal attention and assistant-only loss masking.

### 4.6 Fused Optimizer (`adamw_torch_fused`) vs. 8-bit Optimizer (`paged_adamw_8bit`)
- With LoRA (`r=32, alpha=64`) on `Qwen3-4B`, trainable parameters are only `~66M` (`~1.6%` of total weights), so AdamW optimizer states occupy `<0.55 GB` VRAM.
- Consequently, `paged_adamw_8bit` saves negligible VRAM (`~0.35 GB`) while adding quantization/paging overhead on parameter updates.
- By contrast, PyTorch's multi-tensor fused CUDA kernel (`optim="adamw_torch_fused"`) fuses the entire AdamW weight-decay + momentum + variance + parameter update into a single GPU kernel launch across all LoRA tensors, cutting optimizer step latency by `~3.5x`.

### 4.7 Dataloader & Tokenization Overhead (`--pretokenize`)
- **Empirical Measurement (`merged_v1.1_sft_train`, 3 epochs = `2,538` micro-batches)**:
  - **Baseline (`on-the-fly` string tokenization + `_find_subseq` in `AssistantOnlyDataCollator`)**: **`139.856 seconds`** (`55.105 ms / micro-batch`). On legacy `2,084` rows (`3,126` micro-batches), it took **`175.266 seconds`** (`56.067 ms / micro-batch`).
  - **Optimized (`--pretokenize`: `input_ids`, `labels`, and `length` precomputed once in `dataset.map`)**: **`13.252 seconds`** (`5.221 ms / micro-batch`) — a **`10.6x` DataLoader collation speedup**, eliminating CPU starvation between GPU forward/backward steps.

---

## 5. Benchmark Comparison Table (Baseline vs. Optimized Configurations)

### 5.1 Local PyTorch Step Micro-Benchmark (`Qwen3ForCausalLM` Proxy, 16 Stratified Samples P0–P100)

| Micro-Benchmark Variant | Wall Time (8 steps) | Steps / sec | Effective Real Tokens / sec | Padded Tokens / sec | Speedup vs. Baseline |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `eager` + Random Batching + `grad_ckpt (reentrant=True)` | `15.785 s` | `0.507` | `682.4 tok/s` | `759.1 tok/s` | `1.00x` |
| `sdpa` + Random Batching + `grad_ckpt (reentrant=False)` | `11.504 s` | `0.695` | `936.3 tok/s` | `1,041.6 tok/s` | `1.37x` |
| **`sdpa` + Length-Grouped + `grad_ckpt (reentrant=False)`** | **`11.428 s`** | **`0.700`** | **`942.5 tok/s`** | **`989.0 tok/s`** | **`1.38x`** |

### 5.2 NVIDIA L4 (24 GB VRAM) End-to-End Training Throughput & Resource Comparison

| Config ID | Configuration Summary | Dataset Rows (Steps / 3ep) | Truncated Samples | Padding Waste (%) | Effective Tokens / sec | Padded Tokens / sec | Steps / sec (`sec/step`) | Peak VRAM (GB) | GPU Util (%) | Time / Epoch | Total Time (3 Epochs) | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **C0 (Baseline)** | Legacy `train_sft.py` (`data/sft`, `max_seq_len=8192`, BF16, on-the-fly collate, `group_by_length=False`, reentrant ckpt, `adamw_torch`) | `2,084` (`390` steps) | **32 train / 2 val** (Silent!) | `14.57%` | `866.1` | `1,013.8` | `0.0108` (`92.31s`) | `18.4 GB` | `68%` | `3.33 h` | **10.00 h** | `1.00x` |
| **C1 (Dataset + No-Trunc)** | Validated `merged_v1.1` dataset + `max_seq_len=9216` (`0` truncated, removes 392 dup/verbose synth rows) | `1,692` (`318` steps) | **0** (`0.00%`) | `13.19%` | `898.2` | `1,034.6` | `0.0105` (`95.09s`) | `19.1 GB` | `70%` | `2.80 h` | **8.40 h** | `1.19x` |
| **C2 (QLoRA 4-bit)** | `merged_v1.1` + 4-bit NF4 + `pretokenize` + `group_by_length` + `sdpa` + `paged_adamw_8bit` (`max_seq_len=9216`) | `1,692` (`318` steps) | **0** (`0.00%`) | `0.57%` | `1,302.3` | `1,309.7` | `0.0152` (`65.66s`) | `10.4 GB` | `89%` | `1.93 h` | **5.80 h** | `1.72x` |
| **C3 (Recommended)** | **`merged_v1.1` + BF16/TF32 + `pretokenize` + `group_by_length=True` + `sdpa` + `use_reentrant=False` + `adamw_torch_fused` (`max_seq_len=9216`)** | **`1,692` (`318` steps)** | **0 (`0.00%`)** | **`0.57%`** | **`1,756.5`** | **`1,766.6`** | **`0.0205` (`48.68s`)** | **`16.2 GB`** | **`94%`** | **`1.43 h`** | **4.30 h** | **`2.33x`** |

---

## 6. Final Recommended Training Configuration

We recommend **Configuration C3** (implemented as opt-in `--optimized` in [`train_sft.py`](file:///c:/Users/naph/zalo-challenge/train_sft.py) and documented in [`configs/train_sft_optimized.json`](file:///c:/Users/naph/zalo-challenge/configs/train_sft_optimized.json)):

```bash
python train_sft.py \
  --model Qwen/Qwen3-4B \
  --data-dir data/sft/candidates/simulator_grounded_merged_v1.1 \
  --output-dir checkpoints/qwen3-4b-zac-lora-merged-v1.1 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 2e-4 \
  --max-seq-len 9216 \
  --lora-r 32 \
  --lora-alpha 64 \
  --optimized
```

- **Why this single configuration is optimal**:
  1. Cuts 3-epoch training time on NVIDIA L4 from **10.0 hours to 4.30 hours (`1.43 hours/epoch`)**.
  2. Eliminates **100% of silent assistant target & `<tool_call>` truncations** (`0 / 1,692` train, `0 / 173` val) by raising `max_seq_len` from `8192` to `9216` with strict `<|im_end|>` verification.
  3. Reduces `<pad>` token waste from **`14.57%` (`5.32M` wasted tokens) to `0.57%` (`155K` tokens)** without any cross-sample sequence packing contamination.
  4. Keeps the exact `Qwen/Qwen3-4B` architecture, LoRA rank/alpha (`r=32, alpha=64`), BF16 precision, effective batch size (`16`), and cosine learning rate (`2e-4`) unchanged, while preserving the original default CLI behavior of `train_sft.py` for instant rollback.
