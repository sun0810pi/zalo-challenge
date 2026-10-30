"""End-to-end Gemini ReAct agent with reasoning for ZAC 2026 LifeMate.
Tests the upper bound of Gemini 3.8 Flash / Gemini 4 Argon with step-by-step thinking
to benchmark and guide local model training.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate_predictions
from src.guardrail import normalize_arguments, parse_tool_calls


def read_wav_array(wav_path: Path) -> np.ndarray:
    """Read 16kHz mono WAV file using Python standard library without requiring ffmpeg."""
    with wave.open(str(wav_path), "rb") as w:
        frames = w.getnframes()
        raw = w.readframes(frames)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class FastWhisperTranscriber:
    """Fast local Whisper transcriber with disk caching."""

    def __init__(self, model_name: str = "openai/whisper-tiny", cache_file: Path | None = None):
        self.model_name = model_name
        self.cache_file = cache_file
        self.cache: dict[str, str] = {}
        if self.cache_file and self.cache_file.is_file():
            try:
                self.cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.processor = None
        self.model = None

    def _ensure_loaded(self):
        if self.processor is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

    def transcribe(self, wav_path: Path, prompt: str = "") -> str:
        key = str(wav_path.as_posix())
        if key in self.cache:
            return self.cache[key]

        self._ensure_loaded()
        audio = read_wav_array(wav_path)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        predicted_ids = self.model.generate(
            inputs.input_features,
            language="vi",
            task="transcribe",
        )
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        self.cache[key] = text
        return text

    def save_cache(self):
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")


SYSTEM_PROMPT_TEMPLATE = """Bạn là LifeMate — Trợ lý ảo AI thông minh, hỗ trợ người dùng thực hiện các tác vụ hàng ngày qua các ứng dụng đã cài đặt.

BỐI CẢNH HIỆN TẠI:
- Thời gian hiện tại: {current_time}
- Múi giờ: {timezone}
- Vị trí: {location}
- Thông tin người dùng: {user_profile}

DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP SỬ DỤNG:
Chỉ được phép gọi các tool dưới đây (không gọi tool của app chưa cài đặt):
{tools_schema_str}

CÁC NGUYÊN TẮC SUY LUẬN QUAN TRỌNG (BẮT BUỘC TUÂN THỦ):
1. **Lập luận từng bước (Thinking)**: Trước khi quyết định hành động, hãy suy nghĩ cặn kẽ trong khối `<think>...</think>`:
   - Phân tích ý định của người dùng và thời gian (ví dụ: "mai", "tối nay", "chiều thứ 6" tính từ current_time).
   - Kiểm tra hủy bỏ / phủ định ("à mà thôi", "thôi khỏi", "bỏ qua đi", "chỉ cần xem"): nếu người dùng đổi ý hoặc hủy lệnh, KHÔNG thực hiện thay đổi dữ liệu (write tool), chỉ xem hoặc không làm gì.
   - Kiểm tra điều kiện ("nếu mai mưa thì...", "nếu giá vàng dưới..."): phải gọi tool đọc điều kiện trước (weather, gold_price), đợi kết quả ở bước tiếp theo rồi mới quyết định có gọi lệnh tiếp hay không.
   - Read-before-write: Để cập nhật hoặc xóa lịch, báo thức, ghi chú, cần tìm/liệt kê để lấy ID chính xác trước khi thực hiện sửa/xóa.
   - Thiếu thông tin bắt buộc: Nếu thông tin cần thiết không có trong câu nói hoặc lịch sử và không thể tự suy luận, gọi `sys.ask_missing_info` với `tool_name` và danh sách `params` còn thiếu.
   - Không gọi dư thừa: Nếu câu nói chỉ là câu chào hỏi, hỏi thông tin chung ngoài phạm vi tool, hoặc người dùng từ chối, KHÔNG gọi tool nào cả.

2. **Định dạng Tool Call**:
Khi cần gọi công cụ, viết đúng cấu trúc sau:
<tool_call>
{{"name": "<app>.<tool>", "arguments": {{...}}}}
</tool_call>

Hệ thống sẽ thực thi và trả lời kết quả trong khối `<tool_response>...</tool_response>`.
Khi đã hoàn tất mọi thao tác (hoặc không cần gọi tool), đưa ra câu trả lời ngắn gọn cho người dùng và không gọi thêm tool nào nữa.
"""


def format_tools_schema(installed_apps: dict[str, Any], catalog_tools_map: dict[str, Any]) -> str:
    lines = []
    allowed_tools = set()
    for app, data in installed_apps.items():
        for t in data.get("tools", []):
            allowed_tools.add(t)
    allowed_tools.add("sys.ask_missing_info")

    for t_name in sorted(allowed_tools):
        t_meta = catalog_tools_map.get(t_name, {})
        desc = t_meta.get("description", "")
        params = t_meta.get("parameters", {})
        props = params.get("properties", {})
        req = params.get("required", [])
        p_desc = []
        for k, v in props.items():
            req_marker = "(bắt buộc)" if k in req else "(tùy chọn)"
            p_desc.append(f"{k} [{v.get('type', 'any')}] {req_marker}: {v.get('description', '')}")
        lines.append(f"- Tool `{t_name}`: {desc}\n  Tham số: {', '.join(p_desc) if p_desc else 'không có'}")
    return "\n".join(lines)


def run_gemini_react(
    client: Any,
    model_name: str,
    sample: dict[str, Any],
    final_user_text: str,
    history_transcripts: dict[str, str],
    catalog_tools_map: dict[str, Any],
    env: Any,
    final_audio_path: Path | None = None,
    audio_mode: str = "multimodal",
    max_steps: int = 6,
    enable_thinking: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Run interactive ReAct tool-calling loop using Gemini with reasoning."""
    from google.genai import types

    ctx = sample.get("context") or {}
    installed_apps = ctx.get("installed_apps") or {}
    tools_str = format_tools_schema(installed_apps, catalog_tools_map)

    system_instruction = SYSTEM_PROMPT_TEMPLATE.format(
        current_time=ctx.get("current_time", ""),
        timezone=ctx.get("timezone", "Asia/Ho_Chi_Minh"),
        location=json.dumps(ctx.get("location", {}), ensure_ascii=False),
        user_profile=json.dumps(ctx.get("user_profile", {}), ensure_ascii=False),
        tools_schema_str=tools_str,
    )

    contents: list[types.Content] = []

    # Inject history turns
    for turn in sample.get("history") or []:
        role = turn.get("role")
        if role == "user":
            rel_audio = turn.get("audio", "")
            u_text = history_transcripts.get(rel_audio) or turn.get("audio_text", "")
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"[Lời nói người dùng]: {u_text}")]))
        elif role == "assistant":
            parts = []
            a_text = turn.get("text", "")
            if a_text:
                parts.append(types.Part.from_text(text=a_text))
            for tc in turn.get("tool_calls") or []:
                tc_str = f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"
                parts.append(types.Part.from_text(text=tc_str))
            contents.append(types.Content(role="model", parts=parts))

            # If there were tool results, add them as user turn
            tr_parts = []
            for tr in turn.get("tool_results") or []:
                tr_str = f"<tool_response>\n{json.dumps(tr, ensure_ascii=False)}\n</tool_response>"
                tr_parts.append(types.Part.from_text(text=tr_str))
            if tr_parts:
                contents.append(types.Content(role="user", parts=tr_parts))

    # Add final user request (Multimodal audio + text guidance)
    final_parts: list[types.Part] = []
    if audio_mode == "multimodal" and final_audio_path and final_audio_path.is_file():
        audio_bytes = final_audio_path.read_bytes()
        final_parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
        user_msg = "[Yêu cầu của người dùng trong file âm thanh đính kèm]"
        if final_user_text:
            user_msg += f" (Phiên âm tham khảo: {final_user_text})"
        final_parts.append(types.Part.from_text(text=user_msg))
    else:
        final_parts.append(types.Part.from_text(text=f"[Yêu cầu hiện tại của người dùng]: {final_user_text}"))

    contents.append(types.Content(role="user", parts=final_parts))

    config_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "system_instruction": system_instruction,
    }
    if enable_thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)

    config = types.GenerateContentConfig(**config_kwargs)

    # ReAct execution loop
    for step in range(max_steps):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            resp_text = response.text or ""
        except Exception as e:
            print(f"  [Gemini Error step {step}]: {e}")
            break

        parsed_calls = parse_tool_calls(resp_text)
        if not parsed_calls:
            break

        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=resp_text)]))

        call = parsed_calls[0]
        c_name = call.get("name", "")
        c_args = call.get("arguments") or {}

        norm_args = normalize_arguments(c_name, c_args, catalog_tools_map)
        res = env.call(c_name, norm_args)

        tool_resp_str = f"<tool_response>\n{json.dumps(res, ensure_ascii=False)}\n</tool_response>"
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=tool_resp_str)]))

    return env.get_tool_calls(), env.get_state_hash()


def main():
    ap = argparse.ArgumentParser(description="ZAC 2026 LifeMate Gemini Reasoning Agent")
    ap.add_argument("--split", default="public_test", choices=["public_test", "train"])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--output", default="submission.jsonl")
    ap.add_argument("--model", default="gemini-3.8-flash", help="Model: gemini-3.8-flash, gemini-3.5-flash-lite, gemini-4-argon")
    ap.add_argument("--audio-mode", default="multimodal", choices=["multimodal", "asr"], help="multimodal = gửi file wav trực tiếp tới Gemini; asr = dùng whisper trước")
    ap.add_argument("--whisper-model", default="openai/whisper-tiny")
    ap.add_argument("--use-gt-asr", action="store_true", help="Dùng audio_text có sẵn trên train")
    ap.add_argument("--limit", type=int, default=0, help="Số mẫu chạy thử (0 = toàn bộ)")
    ap.add_argument("--resume", action="store_true", default=True, help="Tiếp tục từ output đã có")
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n" + "=" * 60)
        print("[LỖI] Chưa tìm thấy GEMINI_API_KEY trong môi trường hoặc file .env!")
        print("Vui lòng cấu hình GEMINI_API_KEY theo hướng dẫn để chạy inference.")
        print("=" * 60 + "\n")
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=api_key)

    root = Path(args.root)
    split_dir = root / args.split

    if str(split_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(split_dir.resolve()))
    from zaic_tools import ToolEnv
    from zaic_tools.validate import catalog_tools

    catalog_path = split_dir / f"{args.split}_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_tools_map = catalog_tools(catalog)

    samples = [
        json.loads(line)
        for line in (split_dir / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        samples = samples[: args.limit]

    out_path = Path(args.output)
    existing_preds: dict[str, dict[str, Any]] = {}
    if args.resume and out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    existing_preds[row["id"]] = row
                except Exception:
                    pass
        print(f"[Resume] Đã có {len(existing_preds)}/{len(samples)} mẫu trong {args.output}")

    transcriber = None
    if args.audio_mode == "asr" or args.split == "train":
        transcriber = FastWhisperTranscriber(
            model_name=args.whisper_model,
            cache_file=root / "cache" / f"asr_{args.split}.json",
        )

    print(f"\n[Gemini Agent] Model={args.model} | Mode={args.audio_mode} | Split={args.split} | Mẫu={len(samples)}")

    t0 = time.perf_counter()
    with open(out_path, "a" if existing_preds else "w", encoding="utf-8") as fout:
        for idx, item in enumerate(samples, 1):
            qid = item["id"]
            if qid in existing_preds:
                continue

            s_t0 = time.perf_counter()
            env = ToolEnv(args.split, qid, root=root)

            # Transcripts / audio paths
            hist_transcripts: dict[str, str] = {}
            for turn in item.get("history") or []:
                if turn.get("role") == "user":
                    rel = turn.get("audio", "")
                    if args.use_gt_asr and turn.get("audio_text"):
                        hist_transcripts[rel] = turn["audio_text"]
                    elif rel and transcriber:
                        hist_transcripts[rel] = transcriber.transcribe(split_dir / rel)

            rel_final = item.get("audio", "")
            final_audio_path = (split_dir / rel_final) if rel_final else None
            final_text = ""
            if args.use_gt_asr and item.get("audio_text"):
                final_text = item["audio_text"]
            elif transcriber and final_audio_path:
                final_text = transcriber.transcribe(final_audio_path)

            # Run Gemini reasoning loop
            tool_calls, st_hash = run_gemini_react(
                client=client,
                model_name=args.model,
                sample=item,
                final_user_text=final_text,
                history_transcripts=hist_transcripts,
                catalog_tools_map=catalog_tools_map,
                env=env,
                final_audio_path=final_audio_path,
                audio_mode=args.audio_mode,
                max_steps=args.max_steps,
            )

            row = {"id": qid, "tool_calls": tool_calls, "state_hash": st_hash}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            elapsed = time.perf_counter() - s_t0
            print(f"[{idx}/{len(samples)}] {qid} ({elapsed:.2f}s) -> {len(tool_calls)} calls")

    if transcriber:
        transcriber.save_cache()
    print(f"\nHoàn tất inference -> {args.output}")

    # Evaluate
    evaluate_predictions(args.output, root=root, split=args.split)


if __name__ == "__main__":
    main()
