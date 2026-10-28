"""Tool call parser, argument auto-normalizer, and zero-cost schema guardrail using env.validate()."""
from __future__ import annotations

import copy
import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_UNCLOSED_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*\})\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_TIME_IN_STR_RE = re.compile(r"(\d{2}:\d{2}(?::\d{2})?)")


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract all tool calls `[{"name": ..., "arguments": {...}}, ...]` from LLM output text."""
    if not text:
        return []
    # Strip <think> blocks first in case the model drafted a tool call inside <think>
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    calls: list[dict[str, Any]] = []

    matches = _TOOL_CALL_RE.findall(cleaned)
    for raw in matches:
        parsed = _try_parse_json_call(raw)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    # Fallback 1: unclosed <tool_call> tag
    for raw in _UNCLOSED_TOOL_CALL_RE.findall(cleaned):
        parsed = _try_parse_json_call(raw)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    # Fallback 2: check fenced ```json blocks
    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL):
        parsed = _try_parse_json_call(block)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    # Fallback 3: bare JSON {"name": ..., "arguments": ...} (e.g. if <tool_call> special tokens were stripped)
    for line in cleaned.splitlines():
        line_s = line.strip()
        if line_s.startswith("{") and line_s.endswith("}") and '"name"' in line_s:
            parsed = _try_parse_json_call(line_s)
            if parsed is not None:
                calls.append(parsed)

    return calls


def _try_parse_json_call(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try minor trailing comma or single quote repair
        try:
            repaired = re.sub(r",\s*([}\]])", r"\1", raw)
            obj = json.loads(repaired)
        except Exception:
            return None

    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return {"name": obj["name"].strip(), "arguments": args}
    return None


def normalize_arguments(
    name: str,
    arguments: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    tz_offset: str = "+07:00",
) -> dict[str, Any]:
    """Auto-repair common type/format issues against the catalog schema before calling env.validate()."""
    args = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
    spec = (catalog_tools_map.get(name) or {}).get("parameters") or {}
    props = spec.get("properties") or {}
    required = set(spec.get("required") or [])

    cleaned: dict[str, Any] = {}
    for k, v in args.items():
        # Drop unknown keys if properties are defined
        if props and k not in props:
            continue
        # Drop empty values if not required
        if v in (None, "", [], {}) and k not in required:
            continue

        p_spec = props.get(k) or {}
        p_type = p_spec.get("type")
        p_fmt = p_spec.get("format")

        # Type coercion
        if p_type == "integer" and isinstance(v, str):
            v_str = v.strip().replace(",", "").replace(".", "")
            if v_str.lstrip("-").isdigit():
                v = int(v_str)
        elif p_type == "integer" and isinstance(v, float) and v.is_integer():
            v = int(v)
        elif p_type == "number" and isinstance(v, str):
            try:
                v = float(v.strip())
                if v.is_integer():
                    v = int(v)
            except ValueError:
                pass
        elif p_type == "boolean" and isinstance(v, str):
            if v.strip().lower() in ("true", "1", "yes", "bật", "on"):
                v = True
            elif v.strip().lower() in ("false", "0", "no", "tắt", "off"):
                v = False

        # Format normalization
        if isinstance(v, str):
            v = v.strip()
            if p_fmt == "date":
                m = _DATE_PREFIX_RE.match(v)
                if m:
                    v = m.group(1)
            elif p_fmt == "time":
                if "T" in v:
                    m = _TIME_IN_STR_RE.search(v.split("T", 1)[1])
                    if m:
                        v = m.group(1)[:5]
                elif len(v) == 8 and v.count(":") == 2 and v.endswith(":00"):
                    v = v[:5]
            elif p_fmt == "date-time":
                # Convert "YYYY-MM-DD HH:MM" -> "YYYY-MM-DDTHH:MM:00+07:00"
                if " " in v and "T" not in v:
                    v = v.replace(" ", "T", 1)
                if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", v):
                    v = f"{v}:00{tz_offset}"
                elif re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", v):
                    v = f"{v}{tz_offset}"
                elif re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                    v = f"{v}T00:00:00{tz_offset}"

        # Enum case-insensitive repair
        if "enum" in p_spec and isinstance(v, str) and v not in p_spec["enum"]:
            for enum_val in p_spec["enum"]:
                if isinstance(enum_val, str) and enum_val.lower() == v.lower():
                    v = enum_val
                    break

        cleaned[k] = v

    return cleaned
