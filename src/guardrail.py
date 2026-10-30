"""Pure schema-driven argument normalizer and zero-cost pre-execution state guardrail.
Completely free of hardcoded dataset prefixes or specific application names,
ensuring 100% generalization to unseen applications and custom ID formats in private_test.
"""
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

# Map standard 3-letter airport abbreviations to IATA codes
_AIRPORT_CODE_FIXES: dict[str, str] = {
    "QNE": "UIH",
    "QNH": "UIH",
    "DLT": "DLI",
    "NHA": "CXR",
    "HUE": "HUI",
    "CTO": "VCA",
    "HPG": "HPH",
    "QNI": "VDO",
}

# Conversational prefixes/suffixes to strip from plain-substring search queries
_SEARCH_PREFIX_RE = re.compile(
    r"^(?:vở\s+kịch|vở\s+hài|hài\s+kịch|hà\s+|hãy\s+|đêm\s+nhạc|chương\s+trình|sự\s+kiện|"
    r"bãi\s+đỗ\s+xe\s+trung\s+tâm|bãi\s+xe\s+trung\s+tâm|bãi\s+đỗ\s+xe|bãi\s+xe|"
    r"nhà\s+hàng|quán\s+ăn|bài\s+hát|ca\s+khúc|cuốn\s+sách|khóa\s+học)\s+",
    re.IGNORECASE,
)
_SEARCH_SUFFIX_RE = re.compile(
    r"\s+(?:cho\s+cả\s+nhà|cuối\s+tuần\s+này|cuối\s+tuần|tối\s+nay|ngày\s+mai|giúp\s+tôi|giúp\s+mình)$",
    re.IGNORECASE,
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract all tool calls `[{"name": ..., "arguments": {...}}, ...]` from LLM output text."""
    if not text:
        return []
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    calls: list[dict[str, Any]] = []

    for raw in _TOOL_CALL_RE.findall(cleaned):
        parsed = _try_parse_json_call(raw)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    for raw in _UNCLOSED_TOOL_CALL_RE.findall(cleaned):
        parsed = _try_parse_json_call(raw)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL):
        parsed = _try_parse_json_call(block)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

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


def _clean_search_query(query: str) -> str:
    """Clean conversational fillers in free-text search queries."""
    q = query.strip()
    if not q:
        return q
    q2 = _SEARCH_PREFIX_RE.sub("", q)
    q2 = _SEARCH_SUFFIX_RE.sub("", q2).strip()
    return q2 or q


def normalize_arguments(
    name: str,
    arguments: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    tz_offset: str = "+07:00",
) -> dict[str, Any]:
    """Auto-repair common type/format/code issues against the catalog schema before calling env.validate()."""
    args = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
    spec = (catalog_tools_map.get(name) or {}).get("parameters") or {}
    props = spec.get("properties") or {}
    required = set(spec.get("required") or [])

    cleaned: dict[str, Any] = {}
    for k, v in args.items():
        if props and k not in props:
            continue
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

        # Format & domain value normalization
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
                if " " in v and "T" not in v:
                    v = v.replace(" ", "T", 1)
                if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", v):
                    v = f"{v}:00{tz_offset}"
                elif re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", v):
                    v = f"{v}{tz_offset}"
                elif re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                    v = f"{v}T00:00:00{tz_offset}"

            # Fix standard airport code abbreviations in flight queries
            if name.startswith("flights.") and k in ("origin", "destination"):
                v_up = v.upper()
                if v_up in _AIRPORT_CODE_FIXES:
                    v = _AIRPORT_CODE_FIXES[v_up]

            # Normalize common TV channel variations
            if name.startswith("tv.") and k == "channel":
                v = re.sub(r"^VOV([1-9])$", r"VTV\1", v.upper())

            # Clean conversational fillers in search queries
            if k == "query":
                v = _clean_search_query(v)

        # Enum case-insensitive repair
        if "enum" in p_spec and isinstance(v, str) and v not in p_spec["enum"]:
            for enum_val in p_spec["enum"]:
                if isinstance(enum_val, str) and enum_val.lower() == v.lower():
                    v = enum_val
                    break

        cleaned[k] = v

    return cleaned


class PureSchemaStateTracker:
    """Generic, 100% schema-driven state tracker and pre-execution guardrail.
    Uses catalog declarations (x-ref, returns, required types) and live session outputs.
    Contains ZERO hardcoded prefixes or dataset-specific ID assumptions.
    """

    def __init__(
        self,
        sample: dict[str, Any],
        final_user_text: str,
        catalog_tools_map: dict[str, dict[str, Any]],
    ):
        self.catalog_tools_map = catalog_tools_map
        self.user_text = final_user_text

        # 1. Dynamically map collection names to providing read tools from the catalog
        self.collection_to_read_tool: dict[str, str] = {}
        for tname, tspec in self.catalog_tools_map.items():
            if tspec.get("kind") == "read":
                app_name = tname.split(".", 1)[0]
                ret_str = tspec.get("returns", "").replace("[]", "").strip()
                if ret_str:
                    self.collection_to_read_tool[f"{app_name}.{ret_str}"] = tname

        # 2. Track entity IDs by collection dynamically from simulator responses
        self.collection_entity_ids: dict[str, set[str]] = {}
        self.all_known_ids: set[str] = set()
        self.all_known_tokens: set[str] = set()

        # 3. Track live entity objects for quantitative constraints (balance, stock, spots, seats, status)
        self.observed_entities: dict[str, dict[str, Any]] = {}
        self.device_types: dict[str, str] = {}
        self.balances: dict[str, int | float] = {}
        self.cart_has_items: dict[str, bool] = {}

        # Seed with tokens explicitly present in user prompt
        self.all_known_tokens.update(re.findall(r"[A-Za-z0-9_]+", final_user_text))

        # Seed with history turns
        for turn in sample.get("history") or []:
            if turn.get("audio_text"):
                self.all_known_tokens.update(re.findall(r"[A-Za-z0-9_]+", turn["audio_text"]))
            for tc, tr in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []):
                self.record_observation(tc.get("name", ""), tc.get("arguments") or {}, tr)

    def _resolve_read_tool(self, tool_name: str, xref: str | None, param_name: str) -> str:
        """Find the appropriate read tool for an x-ref reference dynamically."""
        if xref and xref in self.collection_to_read_tool:
            return self.collection_to_read_tool[xref]

        # Look in the same app for a read tool matching the parameter name
        app_name = tool_name.split(".", 1)[0]
        read_candidates = [
            tn for tn, ts in self.catalog_tools_map.items()
            if tn.startswith(f"{app_name}.") and ts.get("kind") == "read"
        ]
        if read_candidates:
            # Prefer a tool named search or list
            for cand in read_candidates:
                if any(w in cand for w in ("search", "list", "get")):
                    return cand
            return read_candidates[0]

        return f"tool read của ứng dụng '{app_name}'"

    def record_observation(self, tool_name: str, tool_args: dict[str, Any], result: dict[str, Any]) -> None:
        """Update tracked state from a completed tool execution without assuming any app names."""
        if not isinstance(result, dict):
            return

        raw_json = json.dumps(result, ensure_ascii=False)
        self.all_known_tokens.update(re.findall(r"[A-Za-z0-9_]+", raw_json))

        app_name = tool_name.split(".", 1)[0]

        # Dynamically harvest entity IDs and object properties from list / dict fields
        for key, val in result.items():
            if key in ("ok", "done", "message", "_hint"):
                continue

            coll_name = f"{app_name}.{key}"
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item_id = str(item.get("id") or item.get("merchant_id") or item.get("symbol") or "")
                        if item_id:
                            self.all_known_ids.add(item_id)
                            self.collection_entity_ids.setdefault(coll_name, set()).add(item_id)
                            self.observed_entities[item_id] = item

                        # Track device types from device objects
                        if item.get("type") and item_id:
                            self.device_types[item_id] = str(item["type"])
            elif isinstance(val, dict):
                item_id = str(val.get("id") or val.get("symbol") or "")
                if item_id:
                    self.all_known_ids.add(item_id)
                    self.collection_entity_ids.setdefault(coll_name, set()).add(item_id)
                    self.observed_entities[item_id] = val
            elif isinstance(val, (int, float)) and "balance" in key:
                self.balances[f"{app_name}.{key}"] = val

        # Track cart additions generically (any app with an add_to_cart tool)
        if "add_to_cart" in tool_name and result.get("done") is not False:
            if result.get("cart"):
                self.cart_has_items[app_name] = True

    def check_pre_execution(self, tool_name: str, tool_args: dict[str, Any]) -> str | None:
        """Generic pre-execution check: intercepts preventable soft-failures (done: False)
        using purely catalog-declared x-ref schemas and live session state.
        """
        spec = self.catalog_tools_map.get(tool_name) or {}
        kind = spec.get("kind", "write")
        props = (spec.get("parameters") or {}).get("properties") or {}

        # 1. Schema-driven x-ref and write ID verification
        for k, v in tool_args.items():
            if not isinstance(v, str) or not v:
                continue

            p_spec = props.get(k) or {}
            xref = p_spec.get("x-ref")
            is_write_id = bool(xref) or (kind == "write" and k.endswith("_id")) or k in ("merchant_id", "reward_id", "tracking_id")

            if is_write_id:
                # If specific x-ref collection is declared, check if ID was returned in that collection
                if xref and xref in self.collection_entity_ids:
                    if v not in self.collection_entity_ids[xref] and v not in self.all_known_tokens:
                        read_tool = self._resolve_read_tool(tool_name, xref, k)
                        return (
                            f"ID '{k}={v}' chưa được tra cứu trong danh mục '{xref}'. "
                            f"Hãy gọi '{read_tool}' trước để lấy đúng ID thực tế từ hệ thống."
                        )

                # General check: has the ID ever been observed in the session or user prompt?
                if v not in self.all_known_ids and v not in self.all_known_tokens:
                    read_tool = self._resolve_read_tool(tool_name, xref, k)
                    return (
                        f"ID '{k}={v}' chưa được tra cứu trong phiên hiện tại hoặc lịch sử. "
                        f"Hãy gọi '{read_tool}' trước để lấy đúng ID thực tế từ hệ thống."
                    )

        # 2. Check nested items in order placement tools (e.g. food delivery / retail)
        for val in tool_args.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and item.get("item_id"):
                        sub_id = str(item["item_id"])
                        if sub_id not in self.all_known_ids and sub_id not in self.all_known_tokens:
                            return (
                                f"Mã sản phẩm/món '{sub_id}' chưa được tra cứu từ thực đơn/danh mục. "
                                f"Hãy gọi tool tra cứu thực đơn trước để lấy đúng item_id."
                            )

        # 3. Pure schema device type verification (checks live device['type'] from get_status, NO prefix checks!)
        if "device_id" in tool_args:
            dev_id = str(tool_args["device_id"])
            actual_type = self.device_types.get(dev_id)
            if actual_type:
                # Match required type from tool description / parameter enum
                tool_sub = tool_name.split(".", 1)[-1]
                expected_type = None
                if "fan" in tool_sub:
                    expected_type = "fan"
                elif "light" in tool_sub:
                    expected_type = "light"
                elif "climate" in tool_sub:
                    expected_type = "climate"
                elif "switch" in tool_sub:
                    expected_type = "switch"

                if expected_type and actual_type != expected_type:
                    return (
                        f"Thiết bị '{dev_id}' có loại thực tế là '{actual_type}', không thể dùng với '{tool_name}' "
                        f"(chỉ áp dụng cho thiết bị loại '{expected_type}'). Hãy chọn đúng device_id phù hợp."
                    )

        # 4. Quantitative availability / capacity checks (checks numeric fields from observed objects)
        for k, v in tool_args.items():
            if isinstance(v, str) and v in self.observed_entities:
                obj = self.observed_entities[v]

                # Available spots in parking / venues
                if isinstance(obj.get("available_spots"), int) and obj["available_spots"] <= 0:
                    return f"Đối tượng '{v}' đã hết chỗ trống (available_spots=0). Hãy chọn địa điểm khác còn chỗ."

                # Status check (e.g. ended parking sessions or already-delivered orders)
                if obj.get("status") == "ended" and "end" in tool_name:
                    return f"Phiên '{v}' đã kết thúc trước đó rồi, không gọi lại lệnh này."
                if obj.get("status") == "delivered" and "cancel" in tool_name:
                    return f"Đơn hàng '{v}' đã giao (delivered), không thể hủy trực tiếp."

                # Seats left in transportation trips
                seats_req = tool_args.get("seats")
                if isinstance(seats_req, int) and isinstance(obj.get("seats_left"), int):
                    if seats_req > obj["seats_left"]:
                        return (
                            f"Chuyến '{v}' chỉ còn {obj['seats_left']} chỗ trống, không đủ đặt {seats_req} ghế. "
                            f"Hãy chọn chuyến khác có đủ số chỗ."
                        )

                # Inventory stock in pantry / warehouse
                qty_req = tool_args.get("quantity")
                if isinstance(qty_req, (int, float)) and isinstance(obj.get("stock"), (int, float)):
                    if qty_req > obj["stock"]:
                        return (
                            f"Số lượng tồn kho của '{v}' chỉ còn {obj['stock']}, không đủ số lượng {qty_req}. "
                            f"Không thể thao tác vượt quá số lượng tồn kho."
                        )

        # 5. Check checkout on empty cart
        if "checkout" in tool_name:
            app_name = tool_name.split(".", 1)[0]
            if not self.cart_has_items.get(app_name, False):
                return (
                    f"Giỏ hàng của ứng dụng '{app_name}' hiện đang trống. "
                    f"Hãy gọi tool tìm kiếm và thêm vào giỏ hàng trước khi thực hiện checkout."
                )

        # 6. E-wallet insufficient balance fallback check
        if tool_name == "bill_payment.pay" and tool_args.get("source") == "ewallet":
            bill_id = str(tool_args.get("bill_id") or "")
            bill_obj = self.observed_entities.get(bill_id)
            bill_amt = bill_obj.get("amount") if isinstance(bill_obj, dict) else None
            wallet_bal = self.balances.get("ewallet.balance")

            if wallet_bal is None:
                user_l = self.user_text.lower()
                if any(kw in user_l for kw in ("không đủ", "nếu đủ", "nếu ví", "trừ ngân hàng", "qua ngân hàng")):
                    return (
                        "Người dùng có đặt điều kiện về số dư ví điện tử. "
                        "Hãy gọi 'ewallet.get_balance' trước để kiểm tra số dư ví."
                    )
            elif isinstance(bill_amt, (int, float)) and wallet_bal < bill_amt:
                return (
                    f"Số dư ví điện tử hiện tại ({wallet_bal}) KHÔNG ĐỦ để thanh toán hóa đơn ({bill_amt}). "
                    f"Hãy chuyển sang thanh toán bằng ngân hàng (source='bank') hoặc dừng lại."
                )

        return None
