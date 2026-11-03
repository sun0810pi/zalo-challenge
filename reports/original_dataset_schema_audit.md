# Phase 1 Audit: Official Challenge Dataset Schema & Semantic Roles

**Repository**: `c:\Users\phuon\zalo-challenge`  
**Dataset Versions & Manifest**: `train/train.jsonl` (1,545 samples), `public_test/public_test.jsonl` (634 samples), `VERSION` (v3.2.0)  
**Tools Catalogs**: `train/train_tools_catalog.json` (30 apps, 95 tools), `public_test/public_test_tools_catalog.json` (27 apps, 80 tools)  
**Audit Status**: CONFIRMED via direct inspection of Python runtime, datasets, and simulator execution code.

---

## 1. Executive Summary

This audit establishes the ground-truth contract of the Zalo AI Challenge 2026 (LifeMate track). We verified every schema field across the 1,545 labeled training samples (`train/train.jsonl`) and the 634 unlabeled test samples (`public_test/public_test.jsonl`).

### Key Confirmed Findings
1. **Audio vs. Text Disconnect**: In `public_test/public_test.jsonl`, `audio_text` is **100% absent** (`0 / 634` samples), and past user turns in `history` also lack transcripts (`0 / 1,299` turns have `audio_text`). However, the agent LLM (`Qwen/Qwen3-4B`) is trained **strictly on text** (`audio_text`). It processes zero audio features. The entire system is thus strictly reliant on upstream ASR transcriptions (`models/whisper-large-v3-turbo` or `faster-whisper`), making ASR phonetic corruption the primary source of downstream intent/entity failures.
2. **Deterministic Positional Alignment**: Across all 1,545 training samples, `label.tool_calls` and `label.tool_results` have **zero positional or length mismatches** (`len(calls) == len(results)` in 100% of samples).
3. **Intentional Zero-Action & Read-Only Samples**:
   - Exactly **111 samples (7.18%)** require **zero tool calls** (`label.tool_calls = []`), representing self-cancellations, out-of-scope tasks (uninstalled apps), or conversational small talk.
   - Exactly **373 samples (24.14%)** have an **empty `final_state` (`{}`)**, of which 111 have zero calls and 262 perform only non-mutating `read` operations (e.g. weather forecasts, gold price checks) or call `sys.ask_missing_info` when information is ambiguous.
4. **Disambiguation as a First-Class Action**: Exactly **217 training samples** call `sys.ask_missing_info`. The competition treats asking for clarification on missing or ambiguous parameters as an explicit executable action, not a natural language turn.
5. **Absence of Negative/Failed Executions**: In the entire official training set of 1,545 samples, **zero tool calls return `ok: false`**, and only **1 sample (`train_1696`)** contains a tool result with `done: false`. The ground-truth data contains exclusively positive, successful demonstrations, which severely limits naive offline RL and failure recovery learning.

---

## 2. Input Fields Deep Audit

### 2.1 Field-by-Field Semantic Role & Lifecycle

| Original Field | Data Type | Semantic Role | Runtime Availability | Current SFT Usage | Evaluation Usage | RL Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `string` | Unique sample identifier; links to `<split>_environment/<id>/state.json`. | Train & Test (Always) | Metadata only; not passed to model prompt. | Key in `submission.jsonl`; loads initial environment state. | Unique environment seed / episode ID. |
| `audio` | `string` | Filepath to user request speech (`audio/<id>/<id>.wav`). | Train & Test (Always) | None by LLM. Used exclusively to train/evaluate Whisper ASR. | Raw evaluation input for end-to-end benchmark. | Real environment audio observation. |
| `audio_text` | `string` | Ground-truth Vietnamese transcript of the final user request. | **Train ONLY** (`1,545/1,545`). Absent in test (`0/634`). | **Sole primary user input** to `Qwen3-4B` in `messages[1]`. | Debugging / oracle upper-bound baseline. | Oracle observation text for teacher rollouts. |
| `context.current_time` | `string` (ISO-8601) | Timestamp of utterance (e.g., `2026-11-28T07:10:00+07:00`). | Train & Test (Always) | Formatted into 14-day relative calendar table by `build_calendar_helper()`. | Required by simulator to resolve dates/times (`session.now`). | State feature $s_0$ for temporal reasoning. |
| `context.timezone` | `string` | Timezone identifier (`Asia/Ho_Chi_Minh`). | Train & Test (Always) | Extracted as UTC offset suffix (e.g. `+07:00`) for date-time arguments. | Timezone calculation in calendar/alarm. | Static environment configuration. |
| `context.location` | `dict` (`lat`, `lng`) | Current GPS coordinates of user. | Train & Test (Always) | Rendered as text in `### BỐI CẢNH NGƯỜI DÙNG`: `lat=..., lng=...`. | Used by `ride_hailing`, `maps`, `weather`. | Spatial state feature. |
| `context.user_profile` | `dict` | User entity info: `name`, `home_address`, `work_address`, `city`. | Train & Test (Always) | Injected into system context block. Used to ground "nhà", "công ty". | Grounding target for place search and contacts. | Agent profile / personalization state. |
| `context.installed_apps`| `dict` | Catalog of installed apps & permitted tools `{app: {tools: [...]}}`. | Train & Test (Always) | Formatted into `ALLOWED TOOLS` block. Determines legal action space. | Schema validator enforces that uninstalled calls are rejected. | **Masked Action Space $\mathcal{A}(s)$**. |
| `history` | `list[dict]` | Prior multi-turn conversation turns (user audio + assistant text/calls). | Train (1,018/1,545), Test (419/634). | Formatted into read-only markdown log. Entity IDs are reused without re-read. | Prior state mutations are pre-applied to `state.json`. | Multi-turn dialogue state / working memory. |

### 2.2 Deep Dive: Relationship Between `audio` and `audio_text`
- **Observed Behavior (CONFIRMED)**: In `train_sft.py`, `prepare_sft_data.py`, and `generate_reasoning_sft.py`, `Qwen3-4B` is trained purely on `audio_text`. No audio representations, spectrograms, or speech embeddings are fed into the LLM.
- **Inference Mismatch**: At test time, `public_test.jsonl` contains zero transcripts. In `run_infer.py`:
  ```python
  final_text, hist_transcripts = transcriber.transcribe_sample(item, split_dir)
  tool_calls, st_hash = agent.run_sample(..., final_user_text=final_text)
  ```
  The pipeline is strictly cascaded (`Whisper -> Preprocessor -> Router -> Qwen3-4B`).
- **ASR Noise Sensitivity**: When Whisper commits an error (e.g. `train_0036`: `"4 chỗ"` heard as `"vốn chỗ"`; `train_0003`: `"cô Mai"` heard as `"con mai"`), the LLM receives corrupted text. In the SFT training set, only 180 synthetic ASR noisy samples were generated via rule-based word substitutions (`_ASR_NOISE_PATTERNS`). The LLM has not been trained on real Whisper distribution outputs.

---

## 3. Ground-Truth Fields Deep Audit

### 3.1 Distribution of `label.tool_calls`
Analysis of all 1,545 training samples in `train/train.jsonl`:

| Number of Tool Calls | Sample Count | Percentage | Archetype Description | Representative Sample |
| :---: | :---: | :---: | :--- | :--- |
| **0** | **111** | **7.18%** | Cancellation, negation ("thôi khỏi"), out-of-scope, small talk. | `train_0019`, `train_0036`, `train_0038` |
| **1** | **369** | **23.88%** | Direct single action (e.g. simple alarm, weather check, gold price check). | `train_0002`, `train_0006`, `train_0007` |
| **2** | **428** | **27.70%** | Read-before-write dependency (e.g. `smart_home.get_status` -> `light_set`). | `train_0004`, `train_0047` |
| **3** | **359** | **23.24%** | Multi-hop information lookup + execution (search contact -> get bill -> pay). | `train_0021`, `train_1696` |
| **4** | **161** | **10.42%** | Multi-app workflow (weather -> calendar check -> calendar cancel -> SMS). | `train_0015` |
| **5** | **107** | **6.93%** | Complex conditional pipelines + multi-recipient messaging. | `train_0001`, `train_0003`, `train_0025` |
| **6** | **10** | **0.65%** | Maximum trajectory length in challenge dataset. | `train_0134`, `train_0872` |

### 3.2 Positional Alignment & Execution Dependencies
- **Strict 1:1 Correspondence (CONFIRMED)**: Every tool call in `label.tool_calls[i]` has its exact return payload in `label.tool_results[i]`. Length mismatch across all 1,545 samples is exactly `0`.
- **Sequential Data Flow**: In multi-step samples, `arguments` of call $i$ depend directly on scalar values or IDs returned in `tool_results[i-1]` or earlier.
  - *Example `train_0001`*:
    1. Call 1: `weather.get_forecast(date="2026-11-29")` $\rightarrow$ Result 1: `{"rain": true, "condition": "mưa rào"}`.
    2. Call 2: `calendar.list_events()` $\rightarrow$ Result 2: `[{"id": "evt_01", "title": "chạy bộ", "start": "2026-11-29T08:00:00+07:00"}]`.
    3. Call 3: `calendar.update_event(event_id="evt_01", start="2026-11-29T09:00:00+07:00")` (postponed 1 hour based on rain condition).
    4. Call 4: `contacts.search(query="Sếp Dũng")` $\rightarrow$ Result 4: `[{"id": "c_08", "name": "Sếp Dũng"}]`.
    5. Call 5: `sms.send(contact_id="c_08", content="mai chạy bộ dời sang 09:00 nhé")`.
  - In this trajectory, Call 3 cannot be generated without Result 2 (`evt_01`), and Call 5 cannot be generated without Result 4 (`c_08`).

### 3.3 Semantic Meaning of `label.final_state`
- `final_state` records the **complete final state dictionary** of any application whose data was mutated by a write tool.
- If an app was only queried (read tool) or was not changed, it is **omitted** from `final_state`.
- **373 Samples with Empty `final_state` `{}`**:
  - `111` samples: `tool_calls = []` (no action required).
  - `138` samples: Terminated with `sys.ask_missing_info` before any write tool executed (e.g. `train_0048`).
  - `124` samples: Called only informational read tools (`weather`, `gold_price`, `exchange_rate`) where user condition was not met (e.g. `train_0006`: gold price condition unsatisfied).

### 3.4 Handling of Ambiguity and Disambiguation (`sys.ask_missing_info`)
- The system tool `sys.ask_missing_info` is an official built-in tool that must be invoked when required arguments cannot be determined.
- *Sample `train_0025`*:
  User request: `"Nếu mai mưa thì huỷ lịch đá bóng sáng mai với gọi cho bạn Minh nhé..."`
  1. Calls `weather.get_forecast` $\rightarrow$ returns rain.
  2. Calls `calendar.list_events` $\rightarrow$ cancels `evt_01`.
  3. Calls `contacts.search(query="Minh")` $\rightarrow$ returns **two contacts**: `c_05` (Minh - đồng nghiệp) and `c_08` (Minh - khách hàng).
  4. Call 4: `sys.ask_missing_info(tool_name="phone.call", params=["contact_id"])`.
  5. The episode terminates immediately. Calling `phone.call` with an arbitrary guess results in an immediate evaluation penalty.

### 3.5 Ground-Truth Label Defects Discovered
- *Sample `train_1696` (CONFIRMED DEFECT)*:
  User request asks to message "bà Hồng" via Zalo after opening savings.
  1. Call 1: `banking.open_savings` $\rightarrow$ succeeds.
  2. Call 2: `chat_ott.list_chats()` $\rightarrow$ returns 5 chats (`Bố`, `Tuấn`, `Gia đình`, `Hương`, `Bác Ba`). "Bà Hồng" is NOT in the chat list.
  3. Call 3 in GT label: `chat_ott.send_message(chat_id="chat_6142", content="Tui gửi tiết kiệm rồi nha bà")`.
  4. Result 3: `{"ok": true, "done": false, "message": "không tìm thấy chat_id"}`.
  5. The ground-truth label called a fabricated ID (`chat_6142`) that never existed in the environment, resulting in a soft execution failure (`done: false`).
  - This demonstrates that even official challenge labels contain rare hallucinated/fabricated targets.

---

## 4. Comprehensive Schema Mapping Matrix

| Category | Field Path | Required In Input | Mutable By Agent | Present At Test Time | Ground-Truth Supervision Target |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Identifier** | `id` | Yes | No | Yes | No |
| **Audio** | `audio` | Yes | No | Yes | No |
| **Transcript** | `audio_text` | No (Train only) | No | **No** | No |
| **Temporal Context** | `context.current_time` | Yes | No | Yes | Indirect (via args) |
| **Temporal Context** | `context.timezone` | Yes | No | Yes | Indirect (via args) |
| **Spatial Context** | `context.location.lat` | Yes | No | Yes | Indirect (via args) |
| **Spatial Context** | `context.location.lng` | Yes | No | Yes | Indirect (via args) |
| **User Profile** | `context.user_profile.name` | Yes | No | Yes | Indirect |
| **User Profile** | `context.user_profile.city` | Yes | No | Yes | Indirect |
| **User Profile** | `context.user_profile.home_address` | Yes | No | Yes | Indirect |
| **User Profile** | `context.user_profile.work_address` | Yes | No | Yes | Indirect |
| **App Governance** | `context.installed_apps` | Yes | No | Yes | Legal Action Mask |
| **Prior State** | `history[].role` | Yes | No | Yes | No |
| **Prior State** | `history[].audio` | Yes | No | Yes | No |
| **Prior State** | `history[].audio_text` | No (Train only) | No | **No** | No |
| **Prior State** | `history[].text` | Yes | No | Yes | No |
| **Prior State** | `history[].tool_calls` | Yes | No | Yes | No |
| **Prior State** | `history[].tool_results` | Yes | No | Yes | No |
| **Actions** | `label.tool_calls[].name` | Label only | Yes | No | **Primary Target** |
| **Actions** | `label.tool_calls[].arguments`| Label only | Yes | No | **Primary Target** |
| **Observations** | `label.tool_results[]` | Label only | No | No | Interactive Feedback |
| **Terminal State** | `label.final_state` | Label only | Yes (in memory) | No | Evaluation Hash Target |

---

## 5. Conclusions on Dataset Readiness

1. **For Behavior Cloning (SFT)**: The original dataset provides 1,545 rich, positionally aligned, multi-turn trajectories with complex conditional logic and disambiguation.
2. **For Offline RL**: The original dataset cannot support standard offline RL (e.g. CQL, IQL) directly because it lacks negative trajectories, failed attempts, and dense reward labels.
3. **For Rejection Sampling / Best-of-N**: The dataset is ideal. The combination of `train_environment/<id>/state.json`, `label.tool_calls`, and `label.final_state` allows instantaneous, deterministic reward scoring ($R \in \{0.0, 1.0\}$) by replaying candidate trajectories in `zaic_tools.ToolEnv`.
