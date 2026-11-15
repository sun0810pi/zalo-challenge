"""Build `data/sft/data_v7_final` from `data/sft/data_v6_adapted` for final L4 fine-tuning.

Strategy:
1. Preserve 100% of the clean, simulator-verified `data/sft/data_v6_adapted` base
   (1,906 train / 201 val) with its unified 1-line concise `<think>` format and exact
   triage distribution (79.1% DOABLE / 14.1% ASK_INFO / 6.8% REJECT).
2. Extract all 181 tool schemas across 57 apps from `data_v6_adapted` system prompts.
3. Select 140 unique non-ASR base samples targeting the 14 lowest-coverage weak apps
   (`translator`, `calculator`, `lottery`, `exchange_rate`, `event_tickets`,
   `public_transit`, `habit_tracker`, `package_tracking`, `home_energy`, `pantry`,
   `fuel_price`, `recipes`, `gold_price`, `parking`) in the locked triage ratio:
   111 DOABLE (79.3%) / 20 ASK_INFO (14.3%) / 9 REJECT (6.4%).
4. Generate semantic-preserving spoken / ASR-realistic Vietnamese user request variants
   (`v7_weak_<base_id>`) by modifying ONLY the `Người dùng:` utterance outside quotes
   while keeping `messages[2:]` (simulator-verified `<think>`, `<tool_call>`,
   `<tool_response>`, and closing reply) intact.
5. Run structural, schema parameter, Read-Before-Write ID grounding, and leakage
   validation across all 2,046 train and 201 val records.
"""
from __future__ import annotations

import copy
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.trajectory_validator import (  # noqa: E402
    _extract_tokens_and_ids_from_result,
    validate_sft_messages_record,
)
from src.guardrail import parse_tool_calls  # noqa: E402

SRC_DIR = ROOT / "data" / "sft" / "data_v6_adapted"
OUT_DIR = ROOT / "data" / "sft" / "data_v7_final"

WEAK_APPS = [
    "translator",
    "calculator",
    "lottery",
    "exchange_rate",
    "event_tickets",
    "public_transit",
    "habit_tracker",
    "package_tracking",
    "home_energy",
    "pantry",
    "fuel_price",
    "recipes",
    "gold_price",
    "parking",
]

SPOKEN_PREFIXES = [
    "À ừm, ",
    "Alo trợ lý ơi, ",
    "Này em ơi, ",
    "Cho mình hỏi chút, ",
    "Ủa ",
    "À này, ",
    "Này ",
    "Nhờ bạn chút nha, ",
]

SPOKEN_SUFFIXES = [
    " giúp mình luôn nhé",
    " dùm mình nha",
    " giúp anh với ạ",
    " nhé bạn ơi",
    " giùm em với",
    " nha trợ lý",
]

# Word-boundary-safe lexical / ASR spoken replacements applied outside quoted literals
LEXICAL_SWAPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bUSD\b"), "u ét đê"),
    (re.compile(r"\bEUR\b"), "ơ rô"),
    (re.compile(r"\bJPY\b"), "yên Nhật"),
    (re.compile(r"\bSJC\b"), "ét gi xi"),
    (re.compile(r"\bPNJ\b"), "pi en gi"),
    (re.compile(r"\bRON\s*95\b", re.IGNORECASE), "ron chín lăm"),
    (re.compile(r"\bE5\b"), "e năm"),
    (re.compile(r"\bkWh\b", re.IGNORECASE), "kí lô oát giờ"),
    (re.compile(r"\bcm\b"), "xăng ti mét"),
    (re.compile(r"\bgiúp tôi\b", re.IGNORECASE), "hộ mình"),
    (re.compile(r"\bgiúp mình\b", re.IGNORECASE), "dùm mình"),
    (re.compile(r"\bgiúp anh\b", re.IGNORECASE), "hộ anh"),
    (re.compile(r"\bgiúp chị\b", re.IGNORECASE), "hộ chị"),
    (re.compile(r"\bgiúp em\b", re.IGNORECASE), "hộ em"),
    (re.compile(r"\bkiểm tra\b", re.IGNORECASE), "coi thử"),
    (re.compile(r"\bbao nhiêu\b", re.IGNORECASE), "nhiêu"),
    (re.compile(r"(?<!ngày\s)\bhôm nay\b", re.IGNORECASE), "bữa nay"),
    (re.compile(r"\bngày mai\b", re.IGNORECASE), "mai"),
    (re.compile(r"\bứng dụng\b", re.IGNORECASE), "app"),
]


def _extract_catalog_tools_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Extract normalized tool schemas from system prompt JSON lines in `data_v6_adapted`."""
    tools: dict[str, dict[str, Any]] = {}
    for row in rows:
        sys_txt = row["messages"][0]["content"]
        for line in sys_txt.splitlines():
            s = line.strip()
            if s.startswith('{"name":"'):
                obj = json.loads(s)
                t_name = obj["name"]
                if t_name not in tools:
                    tools[t_name] = {
                        "name": t_name,
                        "kind": obj.get("kind", "write"),
                        "description": obj.get("desc", ""),
                        "parameters": {
                            "type": "object",
                            "properties": obj.get("params") or {},
                            "required": obj.get("required") or [],
                        },
                    }
    return tools


def _classify_row(row: dict[str, Any]) -> tuple[str, set[str], set[str]]:
    """Return `(triage_label, called_apps, installed_apps)` for one SFT row."""
    called_apps: set[str] = set()
    ask_apps: set[str] = set()
    has_tc = False

    for msg in row["messages"][2:]:
        if msg["role"] == "assistant":
            for tc in parse_tool_calls(msg["content"]):
                has_tc = True
                t_name = tc.get("name", "")
                if t_name == "sys.ask_missing_info":
                    target = (tc.get("arguments") or {}).get("tool_name", "")
                    if "." in target:
                        ask_apps.add(target.split(".", 1)[0])
                elif "." in t_name:
                    called_apps.add(t_name.split(".", 1)[0])

    installed_apps: set[str] = set()
    for line in row["messages"][0]["content"].splitlines():
        s = line.strip()
        if s.startswith('{"name":"'):
            t_name = json.loads(s)["name"]
            if "." in t_name:
                installed_apps.add(t_name.split(".", 1)[0])

    if ask_apps or "sys.ask_missing_info" in row["messages"][-2]["content"] if len(row["messages"]) >= 2 else False:
        pass

    has_ask = any(
        tc.get("name") == "sys.ask_missing_info"
        for msg in row["messages"][2:]
        if msg["role"] == "assistant"
        for tc in parse_tool_calls(msg["content"])
    )
    triage = "ASK_INFO" if has_ask else ("DOABLE" if has_tc else "REJECT")
    return triage, (called_apps | ask_apps), installed_apps


def _extract_user_request(user_msg_content: str) -> str:
    marker = "Người dùng:"
    if marker in user_msg_content:
        return user_msg_content.rsplit(marker, 1)[-1].strip()
    return user_msg_content.strip()


def _replace_user_request(user_msg_content: str, new_request: str) -> str:
    marker = "Người dùng:"
    head, _ = user_msg_content.rsplit(marker, 1)
    return f"{head}{marker} {new_request}"


def _transform_unquoted(text: str, rng: random.Random) -> str:
    """Apply safe spoken/ASR variations only outside single/double quoted segments."""
    segments = re.split(r"('[^']*'|\"[^\"]*\")", text)
    swapped_count = 0
    out_segments: list[str] = []
    for seg in segments:
        if (seg.startswith("'") and seg.endswith("'")) or (seg.startswith('"') and seg.endswith('"')):
            out_segments.append(seg)
            continue
        cur = seg
        # Pick up to 2 lexical swaps deterministically so intent stays 100% clear
        shuffled = list(LEXICAL_SWAPS)
        rng.shuffle(shuffled)
        for pat, rep in shuffled:
            if swapped_count >= 2:
                break
            cur2, n = pat.subn(rep, cur, count=1)
            if n > 0:
                cur = cur2
                swapped_count += 1
        out_segments.append(cur)

    transformed = "".join(out_segments).strip()

    # If no lexical swap matched or we want conversational ASR prosody, add prefix/suffix
    has_cancel_tail = any(
        kw in transformed.lower()
        for kw in ("à mà thôi", "à thôi", "thôi khỏi", "khoan đã", "đừng ", "khỏi cần")
    )

    if swapped_count == 0 or rng.random() < 0.7:
        prefix = rng.choice(SPOKEN_PREFIXES)
        # Avoid doubling greetings or discourse particles
        if not re.match(
            r"^(?:à|alo|này|ủa|cho\s+mình|cho\s+em|cho\s+anh|cho\s+chị|nhờ|em\s+ơi|ừ|rứa|thế|vậy|dạ|vâng|thưa)",
            transformed,
            re.IGNORECASE,
        ):
            first = transformed[0].lower() + transformed[1:] if transformed else ""
            transformed = prefix + first

    if not has_cancel_tail and (transformed == text or rng.random() < 0.5):
        core = transformed.rstrip(".!? ")
        if not re.search(
            r"(?:nhé|nha|ạ|với|nghe|vậy|ha|hả|không|nhỉ|chưa|đi|luôn|(?:giúp|hộ|dùm|giùm)\s+\S+(?:\s+\S+)?)$",
            core,
            re.IGNORECASE,
        ):
            transformed = core + rng.choice(SPOKEN_SUFFIXES) + "."
        else:
            # Strip trailing punctuation occasionally to mimic raw ASR transcript
            transformed = core if rng.random() < 0.5 else core + "."
    elif has_cancel_tail and transformed == text:
        transformed = transformed.rstrip(".!?")

    # Normalize spacing
    transformed = re.sub(r"\s+", " ", transformed).strip()
    return transformed


def _validate_record_deep(
    row: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate structure, schema params/types, and Read-Before-Write ID grounding offline."""
    errs = validate_sft_messages_record(row, catalog_tools_map)
    if errs:
        return errs

    sid = row["id"]
    user_text = _extract_user_request(row["messages"][1]["content"])
    history_block = row["messages"][1]["content"].split("### YÊU CẦU HIỆN TẠI")[0]
    grounded_ids: set[str] = set()
    grounded_tokens: set[str] = set(re.findall(r"[A-Za-z0-9_]+", user_text + "\n" + history_block))

    msgs = row["messages"]
    for idx in range(2, len(msgs) - 1, 2):
        asst_content = msgs[idx]["content"]
        calls = parse_tool_calls(asst_content)
        if not calls:
            continue
        tc = calls[0]
        t_name = tc["name"]
        t_args = tc.get("arguments") or {}
        spec = catalog_tools_map.get(t_name) or {}
        params_spec = spec.get("parameters") or {}
        props = params_spec.get("properties") or {}
        required = params_spec.get("required") or []

        for req_k in required:
            if req_k not in t_args:
                errs.append(f"{sid}: Step {idx} ({t_name}) missing required param {req_k!r}.")

        for arg_k, arg_v in t_args.items():
            if arg_k not in props:
                errs.append(f"{sid}: Step {idx} ({t_name}) unknown param {arg_k!r}.")
                continue
            p_spec = props[arg_k]
            if "enum" in p_spec and arg_v not in p_spec["enum"]:
                errs.append(f"{sid}: Step {idx} ({t_name}) invalid enum {arg_k}={arg_v!r}.")

            # Check ID grounding on x-ref params
            if p_spec.get("x-ref") and isinstance(arg_v, str) and arg_v:
                if arg_v not in grounded_ids and arg_v not in grounded_tokens:
                    errs.append(
                        f"{sid}: Step {idx} ({t_name}) ungrounded x-ref ID {arg_k}={arg_v!r}."
                    )

        # Harvest grounded IDs from paired tool_response
        if idx + 1 < len(msgs):
            resp_txt = msgs[idx + 1]["content"]
            m_json = re.search(r"<tool_response>\s*(\{.*\})\s*</tool_response>", resp_txt, re.DOTALL)
            if m_json:
                try:
                    res_obj = json.loads(m_json.group(1))
                    new_ids, _ = _extract_tokens_and_ids_from_result(res_obj)
                    grounded_ids.update(new_ids)
                    grounded_tokens.update(re.findall(r"[A-Za-z0-9_]+", m_json.group(1)))
                except json.JSONDecodeError as exc:
                    errs.append(f"{sid}: Step {idx + 1} invalid tool_response JSON: {exc}")

    return errs


def build_v7_final() -> dict[str, Any]:
    rng = random.Random(42)

    with open(SRC_DIR / "sft_train.jsonl", encoding="utf-8") as f:
        base_train = [json.loads(line) for line in f if line.strip()]
    with open(SRC_DIR / "sft_val.jsonl", encoding="utf-8") as f:
        base_val = [json.loads(line) for line in f if line.strip()]

    catalog_tools_map = _extract_catalog_tools_from_rows(base_train + base_val)

    existing_train_prompts = {
        re.sub(r"\s+", " ", _extract_user_request(r["messages"][1]["content"]).lower())
        for r in base_train
    }
    existing_val_prompts = {
        re.sub(r"\s+", " ", _extract_user_request(r["messages"][1]["content"]).lower())
        for r in base_val
    }
    used_prompts = existing_train_prompts | existing_val_prompts
    already_asr_bases = {r["id"].split("_asr")[0] for r in base_train if "_asr" in r["id"]}

    # Index eligible non-ASR base samples
    doable_by_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ask_pool: list[dict[str, Any]] = []
    reject_pool: list[dict[str, Any]] = []

    for row in base_train:
        rid = row["id"]
        if "_asr" in rid or rid in already_asr_bases:
            continue
        triage, active_apps, installed_apps = _classify_row(row)
        weak_hit = active_apps & set(WEAK_APPS)
        if triage == "DOABLE" and weak_hit:
            for app in sorted(weak_hit):
                doable_by_app[app].append(row)
        elif triage == "ASK_INFO" and weak_hit:
            ask_pool.append(row)
        elif triage == "REJECT" and (rid.startswith("v6_") or rid.startswith("sim_") or (installed_apps & set(WEAK_APPS))):
            reject_pool.append(row)

    # Select 111 DOABLE (balanced round-robin across 14 weak apps), 20 ASK_INFO, 9 REJECT
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_ids: set[str] = set()

    # Round-robin DOABLE starting from lowest-count apps in WEAK_APPS
    app_cursors = {app: 0 for app in WEAK_APPS}
    while sum(1 for tri, _ in selected if tri == "DOABLE") < 111:
        progressed = False
        for app in WEAK_APPS:
            if sum(1 for tri, _ in selected if tri == "DOABLE") >= 111:
                break
            pool = doable_by_app[app]
            while app_cursors[app] < len(pool) and pool[app_cursors[app]]["id"] in selected_ids:
                app_cursors[app] += 1
            if app_cursors[app] < len(pool):
                cand = pool[app_cursors[app]]
                app_cursors[app] += 1
                selected_ids.add(cand["id"])
                selected.append(("DOABLE", cand))
                progressed = True
        if not progressed:
            break

    # Prioritize ASK_INFO rows covering distinct weak apps first
    ask_pool.sort(key=lambda r: (0 if r["id"].startswith(("v6_", "sim_", "pilot_")) else 1, r["id"]))
    for cand in ask_pool:
        if sum(1 for tri, _ in selected if tri == "ASK_INFO") >= 20:
            break
        if cand["id"] not in selected_ids:
            selected_ids.add(cand["id"])
            selected.append(("ASK_INFO", cand))

    # Select 9 REJECT rows preferring weak-app cancellations (`v6_` / `sim_`) + diversity
    reject_pool.sort(key=lambda r: (0 if r["id"].startswith(("v6_", "sim_", "pilot_")) else 1, r["id"]))
    for cand in reject_pool:
        if sum(1 for tri, _ in selected if tri == "REJECT") >= 9:
            break
        if cand["id"] not in selected_ids:
            selected_ids.add(cand["id"])
            selected.append(("REJECT", cand))

    assert len(selected) == 140, f"Expected 140 selected weak samples, got {len(selected)}"

    new_rows: list[dict[str, Any]] = []
    for tri, base_row in selected:
        cloned = copy.deepcopy(base_row)
        cloned["id"] = f"v7_weak_{base_row['id']}"
        orig_u_msg = cloned["messages"][1]["content"]
        orig_req = _extract_user_request(orig_u_msg)

        new_req = None
        for attempt in range(25):
            cand_req = _transform_unquoted(orig_req, random.Random(42 + len(new_rows) * 100 + attempt))
            norm_req = re.sub(r"\s+", " ", cand_req.lower())
            if norm_req != re.sub(r"\s+", " ", orig_req.lower()) and norm_req not in used_prompts:
                new_req = cand_req
                used_prompts.add(norm_req)
                break
        if new_req is None:
            new_req = f"À ừm, {orig_req[0].lower()}{orig_req[1:]}" if orig_req else orig_req
            used_prompts.add(re.sub(r"\s+", " ", new_req.lower()))

        cloned["messages"][1]["content"] = _replace_user_request(orig_u_msg, new_req)
        new_rows.append(cloned)

    final_train = base_train + new_rows
    final_val = list(base_val)

    # Full validation on all train and val records
    train_ids = [r["id"] for r in final_train]
    val_ids = [r["id"] for r in final_val]
    dup_ids = len(train_ids) - len(set(train_ids)) + len(val_ids) - len(set(val_ids))
    id_overlap = len(set(train_ids) & set(val_ids))

    train_norm_prompts = {
        re.sub(r"\s+", " ", _extract_user_request(r["messages"][1]["content"]).lower())
        for r in final_train
    }
    val_norm_prompts = {
        re.sub(r"\s+", " ", _extract_user_request(r["messages"][1]["content"]).lower())
        for r in final_val
    }
    prompt_overlap = len(train_norm_prompts & val_norm_prompts)

    validation_errors: list[str] = []
    for r in final_train + final_val:
        validation_errors.extend(_validate_record_deep(r, catalog_tools_map))

    triage_counts_train: Counter[str] = Counter()
    triage_counts_val: Counter[str] = Counter()
    app_call_counts_train: Counter[str] = Counter()
    app_call_counts_val: Counter[str] = Counter()

    for r in final_train:
        tri, apps, _ = _classify_row(r)
        triage_counts_train[tri] += 1
        for msg in r["messages"][2:]:
            if msg["role"] == "assistant":
                for tc in parse_tool_calls(msg["content"]):
                    app_call_counts_train[tc["name"].split(".", 1)[0]] += 1

    for r in final_val:
        tri, apps, _ = _classify_row(r)
        triage_counts_val[tri] += 1
        for msg in r["messages"][2:]:
            if msg["role"] == "assistant":
                for tc in parse_tool_calls(msg["content"]):
                    app_call_counts_val[tc["name"].split(".", 1)[0]] += 1

    max_train_chars = max(sum(len(m["content"]) for m in r["messages"]) for r in final_train)
    max_val_chars = max(sum(len(m["content"]) for m in r["messages"]) for r in final_val)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sft_train.jsonl", "w", encoding="utf-8") as fout:
        for r in final_train:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "sft_val.jsonl", "w", encoding="utf-8") as fout:
        for r in final_val:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "name": "data_v7_final",
        "based_on": "data_v6_adapted (1,906 train / 201 val)",
        "added_weak_app_spoken_variants": len(new_rows),
        "total_train": len(final_train),
        "total_val": len(final_val),
        "train_triage_distribution": dict(triage_counts_train),
        "val_triage_distribution": dict(triage_counts_val),
        "summary": (
            "data_v6_adapted + 140 simulator-grounded spoken/ASR variants targeting "
            "the 14 lowest-coverage apps in locked 79.1%/14.1%/6.7% triage balance"
        ),
    }
    report = {
        "dataset": "data_v7_final",
        "counts": {
            "v6_adapted_train": len(base_train),
            "v7_weak_spoken_added": len(new_rows),
            "total_train": len(final_train),
            "total_val": len(final_val),
        },
        "integrity_checks": {
            "duplicate_ids": dup_ids,
            "train_val_id_overlap": id_overlap,
            "train_val_prompt_overlap": prompt_overlap,
            "schema_and_grounding_errors": len(validation_errors),
            "sample_errors": validation_errors[:10],
            "max_train_chars": max_train_chars,
            "max_val_chars": max_val_chars,
        },
        "weak_apps_call_counts_train": {
            app: app_call_counts_train[app] for app in WEAK_APPS
        },
        "train_triage_distribution": dict(triage_counts_train),
        "val_triage_distribution": dict(triage_counts_val),
    }

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    summary = build_v7_final()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    assert summary["integrity_checks"]["duplicate_ids"] == 0
    assert summary["integrity_checks"]["train_val_id_overlap"] == 0
    assert summary["integrity_checks"]["train_val_prompt_overlap"] == 0
    assert summary["integrity_checks"]["schema_and_grounding_errors"] == 0
