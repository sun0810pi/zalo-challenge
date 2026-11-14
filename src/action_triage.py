"""Fast-Action Triage Barrier & When2Call Guard for ZAC 2026 LifeMate.
Solution 4: Categorizes user requests into DOABLE / ASK_INFO / REJECT,
prevents out-of-scope hallucinations, and blocks premature sys.ask_missing_info calls.
"""
from __future__ import annotations

from typing import Any


def _resolve_xref_read_tools(
    xref_collection: str,
    target_tool: str,
    catalog_tools_map: dict[str, dict[str, Any]],
) -> list[str]:
    """Find the exact prerequisite read tool(s) returning `xref_collection` (e.g. `ecommerce.orders` -> `ecommerce.order_status`)."""
    app_prefix = xref_collection.split(".", 1)[0] if "." in xref_collection else target_tool.split(".", 1)[0]
    coll_suffix = xref_collection.split(".", 1)[1] if "." in xref_collection else ""

    exact_matches: list[str] = []
    app_reads: list[str] = []
    for tname, tspec in catalog_tools_map.items():
        if not tname.startswith(f"{app_prefix}.") or tspec.get("kind") != "read":
            continue
        app_reads.append(tname)
        ret_str = str(tspec.get("returns") or "").replace("[]", "").strip()
        ret_suffix = ret_str.split(".", 1)[-1] if ret_str else ""
        if ret_str == xref_collection or (coll_suffix and ret_suffix == coll_suffix):
            exact_matches.append(tname)

    return exact_matches or app_reads


class ActionTriageBarrier:
    """Action triage and ask_missing_info validator."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check_out_of_scope(self, sample: dict[str, Any], user_text: str) -> tuple[bool, str]:
        """Reserved triage hook. Disabled for keyword heuristics to avoid false rejects on cross-app nouns."""
        return False, ""

    def guard_ask_missing_info(
        self,
        tool_call_args: dict[str, Any],
        sample: dict[str, Any],
        seen_calls: set[str],
        catalog_tools_map: dict[str, dict[str, Any]],
        observed_entities: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validate whether calling `sys.ask_missing_info` is strictly justified or premature.
        Returns: (is_allowed, reason_or_guidance)
        """
        if not self.enabled:
            return True, "Triage barrier disabled"

        target_tool = tool_call_args.get("tool_name")
        params = tool_call_args.get("params") or []

        if not target_tool:
            return False, "Thiếu tham số 'tool_name' trong sys.ask_missing_info."

        spec = catalog_tools_map.get(target_tool) or {}
        props = (spec.get("parameters") or {}).get("properties") or {}
        seen_tool_names = {c.split(":", 1)[0] for c in seen_calls}

        # Include history read tools as already observed
        for turn in sample.get("history") or []:
            for tc in turn.get("tool_calls") or []:
                if tc.get("name"):
                    seen_tool_names.add(tc["name"])

        # Prerequisite 1: Exact x-ref read tool must have been called before asking user for a system ID
        for p in params:
            p_info = props.get(p) or {}
            raw_xref = p_info.get("x-ref") or ""
            xref = raw_xref.split(" ", 1)[0].strip()
            if xref:
                required_reads = _resolve_xref_read_tools(xref, target_tool, catalog_tools_map)
                if required_reads and not any(rt in seen_tool_names for rt in required_reads):
                    return False, (
                        f"Tham số '{p}' của công cụ '{target_tool}' cần được tra cứu trước qua "
                        f"{'/'.join(required_reads)} (x-ref: '{xref}') thay vì hỏi người dùng."
                    )

        # Prerequisite 2: Parameter already present in user context profile or timestamp
        ctx = sample.get("context") or {}
        prof = ctx.get("user_profile") or {}
        for p in params:
            if p in ("home_address", "work_address", "city") and prof.get(p):
                return False, f"Tham số '{p}' đã có sẵn trong hồ sơ người dùng ({prof[p]})."
            if p in ("current_time", "date") and ctx.get("current_time"):
                return False, f"Tham số thời gian đã có trong bối cảnh ({ctx['current_time']})."

        return True, "Hợp lệ"
