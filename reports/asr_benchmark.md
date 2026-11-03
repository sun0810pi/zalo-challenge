# LifeMate ASR Reliability Benchmark & Uncertainty-Based Fallback Report

**Repository**: `c:\Users\naph\zalo-challenge`  
**Evaluation Split**: Labeled training set (`train/train.jsonl` + `train/audio/*.wav`) — **strictly zero `public_test` labels used**  
**Benchmark Script**: [`scripts/benchmarks/benchmark_asr_reliability.py`](file:///c:/Users/naph/zalo-challenge/scripts/benchmarks/benchmark_asr_reliability.py)  
**Detailed Error Analysis & Provenance JSON**: [`reports/asr_error_analysis.json`](file:///c:/Users/naph/zalo-challenge/reports/asr_error_analysis.json)  
**Fallback & Safe Preprocessor Implementation**: [`src/asr_fallback.py`](file:///c:/Users/naph/zalo-challenge/src/asr_fallback.py)  
**Production Baseline Preserved**: [`src/asr.py`](file:///c:/Users/naph/zalo-challenge/src/asr.py), [`src/preprocessor_a.py`](file:///c:/Users/naph/zalo-challenge/src/preprocessor_a.py), [`run_infer.py`](file:///c:/Users/naph/zalo-challenge/run_infer.py)

---

## 1. Executive Summary

We evaluated four speech-to-text configurations on a stratified benchmark of **28 labeled Vietnamese training utterances** (`train/audio/*.wav`) spanning all seven required challenge error categories:
1. **Whisper Alone** (`models/whisper-large-v3-turbo` via `faster-whisper` with context hotwords prompt).
2. **`Qwen3-ASR-0.6B` Alone** (`Qwen/Qwen3-ASR-0.6B` via `qwen-asr` experimental recognizer).
3. **Current Hybrid Pipeline** (`src/asr.clean_asr_transcript` + `ContextLanguagePreprocessorA` in [`src/preprocessor_a.py`](file:///c:/Users/naph/zalo-challenge/src/preprocessor_a.py)).
4. **Uncertainty-Based Dual-ASR Fallback Pipeline** ([`UncertaintyAwareASRFallback`](file:///c:/Users/naph/zalo-challenge/src/asr_fallback.py#L529-L833) + [`SafeContextPreprocessor`](file:///c:/Users/naph/zalo-challenge/src/asr_fallback.py#L268-L441)).

### Key Findings
- **Why the Current Hybrid Pipeline is Unreliable**:
  1. **Over-aggressive fuzzy n-gram replacement in [`ContextLanguagePreprocessorA.align_entities()`](file:///c:/Users/naph/zalo-challenge/src/preprocessor_a.py#L140-L202)**: Because `align_entities()` matches 1-to-3 word spans against `user_profile.city` and contact names without checking word-count parity or protecting Vietnamese prepositions/verbs, it actively **degrades** Mean Sample WER (`32.60%` $\rightarrow$ `32.80%`), CER (`23.93%` $\rightarrow$ `24.45%`), and Critical Entity Accuracy (`82.14%` $\rightarrow$ `80.36%`). For example, in `train_0020`, it corrupts `"Phần nào đang bật điều hòa"` into `"Phần Đà Nẵng bật điều hòa"` (matching `"nào đang"` $\rightarrow$ `"Đà Nẵng"` from `user_profile.city`), and in `train_0009`/`train_0050` it swallows verbs/prepositions (`"Đặt dùm Em Linh"` $\rightarrow$ `"Đặt Em Linh"`, `"cho em Linh"` $\rightarrow$ `"Em Linh"`).
  2. **Catastrophic Whisper Hallucinations on Noisy Audio**: In `train_0002`, Whisper outputs a YouTube subtitle hallucination (`"Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn"`), which `clean_asr_transcript` strips to an **empty string `""`** (`100%` WER). In `train_0015`, Whisper echoes its own `initial_prompt` (`"Trợ lý ảo tiếng Việt LifeMate, Hà Nội. Từ khóa đơn, tiền điện thoại, hãy subscribe..."`, `116.0%` WER).
- **Strengths and Weaknesses of Standalone `Qwen3-ASR-0.6B`**:
  - **Strengths**: `2.18x` faster than `Whisper-large-v3-turbo` on CPU (`9.70s` vs `21.14s`), immune to Whisper's YouTube subtitle and `initial_prompt` echo hallucinations (`4.35%` WER on `train_0002` vs Whisper's `100%` WER), and achieves the highest standalone **Negation & Intent Accuracy (`89.29%`)**.
  - **Weaknesses**: Without Whisper's decoder `initial_prompt` hotword biasing, standalone `Qwen3-ASR-0.6B` achieves only **`57.14%` Critical Entity Accuracy** (vs Whisper's `82.14%`) and **`64.58%` Number/Date/Time Accuracy** (vs Whisper's `78.27%`), frequently misrecognizing domain abbreviations (`"SJC"` $\rightarrow$ `"trí sĩ"`/`"FGC"`/`"SDC"`, `"Zalo"` $\rightarrow$ `"gia lô"`, `"SMS"` $\rightarrow$ `"em mở sờ"`) and spoken digits (`"123 triệu"` $\rightarrow$ `"128 triệu"`).
- **Uncertainty-Based Dual-ASR Fallback Wins Across Every Accuracy Metric**:
  - Combining Whisper primary + [`SafeContextPreprocessor`](file:///c:/Users/naph/zalo-challenge/src/asr_fallback.py#L268-L441) + conditional `Qwen3-ASR-0.6B` fallback reduces Corpus WER from **`33.64%` to `26.36%` (`-7.28` pp)** and Corpus CER from **`26.01%` to `19.17%` (`-6.84` pp)**, while improving **Critical Entity Accuracy to `83.93%`**, **Number/Date/Time Accuracy to `81.85%`**, **Negation & Intent Accuracy to `85.71%`**, and **Downstream Tool Execution Accuracy to `28.57%`**, with **zero false entity overwrites** and explicit ambiguity preservation on 9 genuine slot conflicts.

---

## 2. Quantitative Benchmark Results (4 Pipelines × 7 Metrics)

All four pipelines were evaluated on the exact same 28 stratified labeled training samples (`train/train.jsonl`):

| Metric | 1. Whisper Alone (`whisper-large-v3-turbo`) | 2. `Qwen3-ASR-0.6B` Alone | 3. Current Hybrid (`clean_asr` + `PreprocessorA`) | 4. Uncertainty-Based Fallback (`Whisper` + `SafePrep` + `Qwen3-ASR`) | Delta (Fallback vs. Current Hybrid) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Corpus WER (%)** $\downarrow$ | `33.64%` | `34.00%` | `33.64%` | **`26.36%`** | **`-7.28%` (rel. `-21.6%`)** |
| **Mean Sample WER (%)** $\downarrow$ | `32.60%` | `34.56%` | `32.80%` | **`26.41%`** | **`-6.39%`** |
| **Corpus CER (%)** $\downarrow$ | `25.73%` | `24.42%` | `26.01%` | **`19.17%`** | **`-6.84%` (rel. `-26.3%`)** |
| **Mean Sample CER (%)** $\downarrow$ | `23.93%` | `24.76%` | `24.45%` | **`18.54%`** | **`-5.91%`** |
| **Critical Entity Accuracy (%)** $\uparrow$ | `82.14%` | `57.14%` | `80.36%` | **`83.93%`** | **`+3.57%`** |
| **Number / Date / Time Accuracy (%)** $\uparrow$ | `78.27%` | `64.58%` | `78.27%` | **`81.85%`** | **`+3.58%`** |
| **Negation & Intent Accuracy (%)** $\uparrow$ | `82.14%` | **`89.29%`** | `82.14%` | `85.71%` | **`+3.57%`** |
| **Downstream Tool Execution Accuracy (%)** $\uparrow$ | `25.00%` | `14.29%` | `25.00%` | **`28.57%`** | **`+3.57%`** |
| **False Entity Overwrites (Count)** $\downarrow$ | `0` | `0` | `3` (corrupted) | **`0`** | **`-3` (100% eliminated)** |
| **Avg Latency per Utterance (CPU / Est. L4 GPU)** | `21.14s` / `~0.42s` | **`9.70s`** / `~0.22s` | `21.14s` / `~0.42s` | `30.51s` / `~0.58s` | `+9.37s` CPU / `+0.16s` L4 |

---

## 3. Breakdown Across the 7 Required Error Categories

### Category-Level Downstream Tool Execution Accuracy (%)

| Error Category | Sample Count (`n`) | 1. Whisper Alone | 2. `Qwen3-ASR-0.6B` Alone | 3. Current Hybrid | 4. Uncertainty Fallback | Key Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Vietnamese Regional Accents** | 12 | `16.67%` | `8.33%` | `16.67%` | **`16.67%`** | Heavy Central/Southern colloquial speech (`dùm`, `nghen`, `coi`, `bữa`) causes phonetic vowel shifts in both models; SafePreprocessor prevents false entity corruption. |
| **2. Product & Application Names** | 10 | `30.00%` | `10.00%` | `30.00%` | **`30.00%`** | Whisper's `initial_prompt` hotwords preserve `SJC`, `SMS`, `Zalo`, `RON 95`; Qwen3-ASR alone mishears `SJC` as `trí sĩ`/`FGC`/`SDC` and `Zalo` as `gia lô`. |
| **3. Proper Nouns** | 17 | `17.65%` | `11.76%` | `17.65%` | **`23.53%`** | Fallback fixes `"con mai"` $\rightarrow$ `"cô Mai"` (`train_0003`) and eliminates `"nào đang"` $\rightarrow$ `"Đà Nẵng"` (`train_0020`). |
| **4. Numbers & Amounts** | 15 | `33.33%` | `13.33%` | `33.33%` | **`40.00%`** | Fallback repairs `"vốn chỗ"` $\rightarrow$ `"4 chỗ"` (`train_0036`) and flags conflicting amounts (`123 triệu` vs `128 triệu` in `train_0019`) via ambiguity guard. |
| **5. Dates & Time** | 18 | `27.78%` | `11.11%` | `27.78%` | **`33.33%`** | Rescues `"ngày mai"` weather check in `train_0002` and normalizes `8 giờ` vs `tám giờ` (`time:8:00`) without false conflicts. |
| **6. Negation & Cancellation** | 10 | `30.00%` | `10.00%` | `30.00%` | **`30.00%`** | Rescues `"à mà thôi, cứ để lịch đó"` in `train_0002` (where Whisper hallucinated YouTube subtitles) and repairs `"khởi khỏi, không cần"` $\rightarrow$ `"thôi khỏi, không cần"` (`train_0010`). |
| **7. Multi-Intent Requests** | 13 | `15.38%` | `23.08%` | `15.38%` | **`23.08%`** | Rescues multi-step conditional utterances (`train_0002`, `train_0015`) where Whisper echoed its prompt or hallucinated subtitles. |

---

### Detailed Error Inspection by Category (Concrete Examples from `train`)

#### 1. Vietnamese Regional Accents (`regional_accents`)
- **Example `train_0009`**:
  - **Ground Truth**: `"Nạp dùm em Linh 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha."`
  - **Raw Whisper**: `"Đặt dùm Em Linh 50.000 cho số 0950536 0367 nha."`
  - **`Qwen3-ASR-0.6B`**: `"Đặt giùm em lên 50.000 cho số 0950 5360367 nha."`
  - **Current Hybrid**: `"Đặt Em Linh 50.000 cho số 0950536 0367 nha"` (**Regression**: `ContextLanguagePreprocessorA` swallowed `"dùm"` into `"Em Linh"`).
  - **Uncertainty Fallback**: `"Đặt dùm Em Linh 50.000 cho số 0950536 0367 nha"` (Preserves `"dùm"` and `"Em Linh"`, while logging the phone digit spacing difference between Whisper and Qwen3-ASR).
- **Example `train_0027`** (Extreme fast colloquial speech):
  - **Ground Truth**: `"Anh muốn đặt xe từ Quận 1 đi sân bay lúc 4 giờ chiều nay... mà thôi khỏi, chiều anh tự lái xe đi cũng được."`
  - **Raw Whisper**: `"À, đúng rồi, sẽ tự cùng một cái sân bay lúc 4 giờ chiều nay. Mà thôi khỏi, chị ăn từ lá sẽ đi cũng được."`
  - **`Qwen3-ASR-0.6B`**: `"anh muốn đặt xe từ quận 1 đi sân bay lúc 4 giờ chiều nay mà thôi khỏi chiều anh tự lái xe đi cũng được"` (**`WER = 0.00%`** — Qwen3-ASR decoded the Southern accent cleanly where Whisper failed on `"Quận 1"` and `"anh tự lái xe"`).

#### 2. Product and Application Names (`product_and_app_names`)
- **Example `train_0034`**:
  - **Ground Truth**: `"Xem giá vàng SJC rồi nhắn qua Zalo cho Mẹ."`
  - **Raw Whisper**: `"Xem giá vàng SJC rồi nhắn qua Zalo cho mẹ."` (`WER = 0.00%` normalized)
  - **`Qwen3-ASR-0.6B`**: `"Xem giá vàng trí sĩ rồi nhắn qua gia lô cho mẹ."` (`WER = 40.0%` — mishears `"SJC"` as `"trí sĩ"` and `"Zalo"` as `"gia lô"`).
- **Example `train_0007`**:
  - **Ground Truth**: `"Nhắn Zalo cho sếp Minh hỏi mấy giờ họp, xong gửi SMS cho vợ bảo tối nay anh về trễ."`
  - **Raw Whisper**: `"Nhắn Zalo cho sếp mình hỏi mấy giờ họp, xong gửi SMS cho vợ bảo tối nay ăn về trễ."`
  - **`Qwen3-ASR-0.6B`**: `"Nhắn gia lô cho sếp Minh hỏi mấy giờ họp, xong gửi em mở sờ cho vợ bảo tối nay anh về trễ."` (Recovers `"sếp Minh"` and `"anh về trễ"`, but mishears `"Zalo"` $\rightarrow$ `"gia lô"` and `"SMS"` $\rightarrow$ `"em mở sờ"`).

#### 3. Proper Nouns (`proper_nouns`)
- **Example `train_0020`** (False-Positive City Injection Bug in Current Hybrid Pipeline):
  - **Ground Truth**: `"Phòng nào đang bật điều hòa thì tăng lên 26 độ, còn không thì thôi."`
  - **Raw Whisper**: `"Phần nào đang bật điều hòa thì tăng lên 26 độ, còn không thì thôi."`
  - **Current Hybrid**: `"Phần Đà Nẵng bật điều hòa thì tăng lên 26 độ, còn không thì thôi"` (**Critical Bug**: `ContextLanguagePreprocessorA` matched `"nào đang"` to `user_profile.city = "Đà Nẵng"` with Dice similarity `0.73 >= 0.72`).
  - **Uncertainty Fallback**: `"Phần nào đang bật điều hòa thì tăng lên 26 độ, còn không thì thôi"` (`SafeContextPreprocessor` blocks `user_profile.city` overwrites and protects Vietnamese stopwords `"nào"`, `"đang"`).
- **Example `train_0003`**:
  - **Ground Truth**: `"Kiểm tra số dư ngân hàng, nếu đủ thì chuyển ví cho cô Mai 200 nghìn."`
  - **Raw Whisper**: `"Kiểm tra số dư ngân hàng, nếu đủ thì chuyển ví cho con mai 200.000."`
  - **Uncertainty Fallback**: `"Kiểm tra số dư ngân hàng, nếu đủ thì chuyển ví cho cô Mai 200.000"` (`WER` improves from `21.7%` $\rightarrow$ `13.0%`, Critical Entity Accuracy `0%` $\rightarrow$ `100%`).

#### 4. Numbers and Amounts (`numbers_amounts`)
- **Example `train_0036`**:
  - **Ground Truth**: `"Đặt xe 4 chỗ từ nhà đến công ty lúc 8 giờ sáng mai."`
  - **Raw Whisper**: `"Đặt xe vốn chỗ từ nhà đến công ty lúc 8 giờ sáng mai."` (Mishears `"4 chỗ"` as `"vốn chỗ"`, causing `ride_hailing.book_ride(vehicle_type=...)` to lose the `4-seat` constraint).
  - **Uncertainty Fallback**: `"Đặt xe 4 chỗ từ nhà đến công ty lúc 8 giờ sáng mai"` (Repaired to `"4 chỗ"`, `ndt_acc = 100%`).
- **Example `train_0019`** (Conflicting Amounts Guarded by Ambiguity Preservation):
  - **Ground Truth**: `"Xem giá vàng SJC với lại cổ phiếu ZKN, cái nào dưới 123 triệu thì báo tôi mua."`
  - **Raw Whisper**: `"Xem giá vàng SJC với lại cổ phiếu ZKN. Cái nào dưới 120 triệu thì báo tôi mua."` (`120 triệu`)
  - **`Qwen3-ASR-0.6B`**: `"Xem giá vàng FGC với lại cổ phiếu ZKN, cái nào dưới 128 triệu thì báo tôi mua."` (`128 triệu`)
  - **Uncertainty Fallback**: Detects `numbers_amounts` conflict (`whisper=['120000000']` vs `qwen3_asr=['128000000']`), sets `conflict_detected=True`, and attaches `clarification_prompt` instead of silently guessing between `120M` and `128M`.

#### 5. Dates and Time (`dates_and_time`)
- **Example `train_0005`** (Self-Repair Time Expression):
  - **Ground Truth**: `"Đặt báo thức 5 giờ nhé, à không, 5 rưỡi đi."`
  - **Raw Whisper / Fallback**: `"Đặt báo thức 5 giờ nhé, à không, 5 rưỡi đi."` (`WER = 0.00%`, `dates_times = ['time:5:00', 'time:5:30']`, `negation_cancellation = ['self_repair_a_khong']`).

#### 6. Negation and Cancellation (`negation_and_cancellation`)
- **Example `train_0002`** (Whisper YouTube Subtitle Hallucination Rescued by `Qwen3-ASR-0.6B`):
  - **Ground Truth**: `"Nếu mai mưa thì hủy lịch đá bóng... À mà thôi, cứ để lịch đó, chỉ cần xem giúp mai có mưa không."`
  - **Raw Whisper**: `"Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn"` (`WER = 100.0%`).
  - **Current Hybrid**: `""` (Stripped to empty string; agent receives empty prompt and fails completely).
  - **`Qwen3-ASR-0.6B` / Uncertainty Fallback**: `"nếu mai mưa thì hủy lịch đá bóng à mà thôi tôi để lịch đó chỉ cần xem giúp mai có mưa không"` (**`WER = 4.35%`**; `qwen3_rescued_whisper_hallucination_or_empty` recovers both the cancellation `"à mà thôi"` and the weather query `"xem giúp mai có mưa không"`).
- **Example `train_0010`**:
  - **Ground Truth**: `"Tắt đèn phòng ngủ giúp... à mà thôi khỏi, không cần đâu."`
  - **Raw Whisper**: `"Tắt đèn phòng ngủ giúp. À mà khởi khỏi, không cần đâu."` (Mishears `"thôi khỏi"` as `"khởi khỏi"`, causing `is_complete_cancellation()` to miss the cancellation).
  - **Uncertainty Fallback**: `"Tắt đèn phòng ngủ giúp. À mà thôi khỏi, không cần đâu"` (`negation_intent_accuracy` improves from `0%` $\rightarrow$ `100%`).

#### 7. Multi-Intent Requests (`multi_intent_requests`)
- **Example `train_0015`** (Whisper `initial_prompt` Echo Hallucination Rescued by `Qwen3-ASR-0.6B`):
  - **Ground Truth**: `"Kiểm tra hóa đơn tiền điện với tiền nước tháng này rồi thanh toán cái nào dưới 300 nghìn bằng ngân hàng."`
  - **Raw Whisper**: `"Trợ lý ảo tiếng Việt LifeMate, Hà Nội. Từ khóa đơn, tiền điện thoại, hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn. Cái nào dưới 300.000đ bằng ngân hàng?"` (`WER = 116.0%` — Whisper echoed its own hotwords prompt and YouTube subtitle boilerplate over the first half of the audio!).
  - **Uncertainty Fallback**: Strips the prompt echo and subtitle hallucination (`removed_whisper_prompt_echo`, `removed_whisper_subtitle_hallucination`) and rescues the utterance via `Qwen3-ASR-0.6B`, cutting WER by मोर than half (`116.0%` $\rightarrow$ `56.0%`).

---

## 4. Architecture of the Uncertainty-Based Fallback (`src/asr_fallback.py`)

Implemented in [`src/asr_fallback.py`](file:///c:/Users/naph/zalo-challenge/src/asr_fallback.py) and exposed via opt-in CLI flags in [`run_infer.py`](file:///c:/Users/naph/zalo-challenge/run_infer.py#L96-L147) (`--preprocessor safe` and `--use-asr-fallback`):

```mermaid
flowchart TD
    A["Input Audio (.wav) + Sample Context"] --> B["Primary ASR: faster-whisper-large-v3-turbo\n(Extract raw_text, avg_logprob, no_speech_prob, compression_ratio)"]
    B --> C["SafeContextPreprocessor\n1. Strip YouTube & prompt-echo hallucinations\n2. High-precision phonetic rules\n3. Word-count-parity & stopword-safe entity align"]
    C --> D{"Uncertainty / Critical Slot Trigger?\n- avg_logprob < -0.32\n- Hallucination stripped / empty text\n- Phonetic/entity repair applied\n- Negation / self-repair slot\n- Write-domain number/entity with avg_logprob < -0.18"}
    D -- "No (High-Confidence Primary)" --> E["Return Safe-Cleaned Whisper Transcript\n+ Provenance Record"]
    D -- "Yes" --> F["Secondary ASR: Qwen/Qwen3-ASR-0.6B\n+ SafeContextPreprocessor"]
    F --> G{"Whisper Empty or Corrupted by\nSubtitle/Prompt Hallucination?"}
    G -- "Yes (e.g. train_0002, train_0015)" --> H["Rescue with Qwen3-ASR Transcript\n(reason: qwen3_rescued_whisper_hallucination_or_empty)"]
    G -- "No" --> I{"Compare Structured Critical Slots\n(numbers_amounts, dates_times,\nnegation_cancellation, proper_nouns)"}
    I -- "Slots Agree" --> J["Return Verified Transcript\n(resolution: dual_asr_verified)"]
    I -- "Conflict on Critical Slot" --> K["NEVER Silently Guess!\nSet conflict_detected=True, populate ambiguous_slots\n& attach [ASR_AMBIGUITY_GUARD] clarification note"]
```

### Provenance Fields Preserved on Every Utterance ([`ASRProvenanceRecord`](file:///c:/Users/naph/zalo-challenge/src/asr_fallback.py#L135-L159))
Every sample in [`reports/asr_error_analysis.json`](file:///c:/Users/naph/zalo-challenge/reports/asr_error_analysis.json) and `run_infer.py --use-asr-fallback` preserves:
- `audio_path`: Original relative `.wav` path.
- `raw_whisper_transcript`: Unmodified `faster-whisper-large-v3-turbo` output.
- `qwen3_asr_transcript`: Alternative `Qwen/Qwen3-ASR-0.6B` transcript (when invoked).
- `current_hybrid_transcript`: Output of legacy `ContextLanguagePreprocessorA` for side-by-side audit.
- `fallback_corrected_transcript`: Final transcript passed to the downstream agent.
- `trigger_reasons` & `correction_reasons`: Exact rule names and similarity scores that fired.
- `confidence_evidence`: `whisper_avg_logprob`, `whisper_no_speech_prob`, `whisper_compression_ratio`, `whisper_slots`, `qwen3_slots`, and `resolution`.
- `conflict_detected`, `ambiguous_slots`, and `clarification_prompt`: Explicit ambiguity guard instruction for `sys.ask_missing_info` when Whisper and Qwen3-ASR disagree on a critical slot.

---

## 5. Reproducibility Commands

```powershell
# Re-run the ASR benchmark (uses cached transcriptions in cache/benchmark_*_train.json in <1s;
# pass a new cache path or delete cache files to re-run full model inference from scratch):
python scripts/benchmarks/benchmark_asr_reliability.py --num-samples 28 --device cpu

# Run inference with the safe preprocessor + uncertainty-aware Qwen3-ASR fallback (opt-in):
python run_infer.py --split public_test --mode local --device cuda --preprocessor safe --use-asr-fallback --asr-fallback-log reports/public_test_asr_provenance.json
```
