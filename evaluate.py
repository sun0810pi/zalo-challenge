"""Local scorer and validator for ZAC 2026 LifeMate using zaic_tools."""
from __future__ import annotations

import argparse
import json
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Any


def _load_zaic_tools(root: Path, split: str):
    split_path = str((root / split).resolve())
    if split_path not in sys.path:
        sys.path.insert(0, split_path)
    import zaic_tools  # noqa: E402
    from zaic_tools.utils import FREE_TEXT_KEYS, _canon_scalar, _tokens  # noqa: E402
    from zaic_tools.validate import catalog_tools, schema_validate  # noqa: E402

    return zaic_tools, FREE_TEXT_KEYS, _canon_scalar, _tokens, catalog_tools, schema_validate


def _arg_match(k: str, pred_v: Any, gt_v: Any, free_text_keys: set[str], _canon_scalar, _tokens) -> bool:
    if k in free_text_keys and isinstance(pred_v, str) and isinstance(gt_v, str):
        t_pred = _tokens(pred_v)
        t_gt = _tokens(gt_v)
        if not t_gt and not t_pred:
            return True
        if not t_gt or not t_pred:
            return False
        overlap = len(t_pred & t_gt) / len(t_pred | t_gt)
        return overlap >= 0.5 or t_gt.issubset(t_pred)
    if isinstance(pred_v, list) and isinstance(gt_v, list) and k == "params":
        return sorted(str(x) for x in pred_v) == sorted(str(x) for x in gt_v)
    return _canon_scalar(pred_v) == _canon_scalar(gt_v)


def _calls_match(pred_call: dict[str, Any], gt_call: dict[str, Any], free_text_keys: set[str], _canon_scalar, _tokens) -> bool:
    if pred_call.get("name") != gt_call.get("name"):
        return False
    p_args = {k: v for k, v in (pred_call.get("arguments") or {}).items() if v not in (None, "", [], {})}
    g_args = {k: v for k, v in (gt_call.get("arguments") or {}).items() if v not in (None, "", [], {})}
    if set(p_args.keys()) != set(g_args.keys()):
        return False
    return all(_arg_match(k, p_args[k], g_args[k], free_text_keys, _canon_scalar, _tokens) for k in g_args)


def compute_call_f1(
    pred_calls: list[dict[str, Any]],
    gt_calls: list[dict[str, Any]],
    free_text_keys: set[str],
    _canon_scalar,
    _tokens,
) -> tuple[float, float]:
    """Return `(name_f1, full_call_f1)` for one sample."""
    if not pred_calls and not gt_calls:
        return 1.0, 1.0
    if not pred_calls or not gt_calls:
        return 0.0, 0.0

    # 1. Tool name multiset F1
    used_gt_name = [False] * len(gt_calls)
    name_tp = 0
    for pc in pred_calls:
        for j, gc in enumerate(gt_calls):
            if not used_gt_name[j] and pc.get("name") == gc.get("name"):
                used_gt_name[j] = True
                name_tp += 1
                break
    name_f1 = (2 * name_tp) / (len(pred_calls) + len(gt_calls))

    # 2. Full call (name + arguments) F1
    used_gt_full = [False] * len(gt_calls)
    full_tp = 0
    for pc in pred_calls:
        for j, gc in enumerate(gt_calls):
            if not used_gt_full[j] and _calls_match(pc, gc, free_text_keys, _canon_scalar, _tokens):
                used_gt_full[j] = True
                full_tp += 1
                break
    full_f1 = (2 * full_tp) / (len(pred_calls) + len(gt_calls))
    return name_f1, full_f1


def evaluate_predictions(
    submission_path: str | Path,
    root: str | Path = ".",
    split: str = "train",
    verbose_errors: int = 5,
) -> dict[str, float]:
    root = Path(root)
    zaic_tools, FREE_TEXT_KEYS, _canon_scalar, _tokens, _catalog_tools, schema_validate = _load_zaic_tools(root, split)
    ToolEnv = zaic_tools.ToolEnv

    catalog = json.loads((root / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
    gt_map: dict[str, dict[str, Any]] = {}
    with open(root / split / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gt_map[row["id"]] = row

    preds: list[dict[str, Any]] = []
    with open(submission_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    n = 0
    schema_errors = 0
    state_matches = 0
    name_f1_sum = 0.0
    full_f1_sum = 0.0
    printed_err = 0

    for pred in preds:
        qid = pred["id"]
        if qid not in gt_map:
            continue
        n += 1
        gt_row = gt_map[qid]
        pred_calls = pred.get("tool_calls") or []

        # Replay predicted calls on fresh ToolEnv to verify state_hash & schema
        env_pred = ToolEnv(split, qid, root=root)
        for v in schema_validate(pred_calls, catalog, env_pred.allowed):
            if not v["ok"]:
                schema_errors += 1
        for c in pred_calls:
            env_pred.call(c.get("name", ""), c.get("arguments") or {})
        replayed_hash = env_pred.get_state_hash()
        sub_hash = pred.get("state_hash") or replayed_hash

        # Ground-truth state hash
        if "label" in gt_row:
            gt_calls = gt_row["label"].get("tool_calls") or []
            env_gt = ToolEnv(split, qid, root=root)
            for c in gt_calls:
                env_gt.call(c.get("name", ""), c.get("arguments") or {})
            gt_hash = env_gt.get_state_hash()

            st_ok = (sub_hash == gt_hash) and (replayed_hash == gt_hash)
            if st_ok:
                state_matches += 1
            nf1, ff1 = compute_call_f1(pred_calls, gt_calls, FREE_TEXT_KEYS, _canon_scalar, _tokens)
            name_f1_sum += nf1
            full_f1_sum += ff1

            if (not st_ok or ff1 < 1.0) and printed_err < verbose_errors:
                printed_err += 1
                print(f"\n[Mismatch #{printed_err}] ID={qid} | state_ok={st_ok} | call_F1={ff1:.2f}")
                print(f"  Audio text : {gt_row.get('audio_text', '')}")
                print(f"  GT calls   : {json.dumps(gt_calls, ensure_ascii=False)}")
                print(f"  Pred calls : {json.dumps(pred_calls, ensure_ascii=False)}")

    if n == 0:
        print("Không có mẫu nào để chấm!")
        return {}

    metrics = {
        "samples": n,
        "schema_errors": schema_errors,
        "state_acc": state_matches / n if "label" in next(iter(gt_map.values())) else 0.0,
        "tool_name_f1": name_f1_sum / n,
        "tool_call_f1": full_f1_sum / n,
        "combined_score": 0.5 * (state_matches / n) + 0.5 * (full_f1_sum / n),
    }
    print("\n" + "=" * 55)
    print(f"KẾT QUẢ ĐÁNH GIÁ ({split} - {n} mẫu)")
    print("=" * 55)
    print(f"- Lỗi Schema (Schema Errors) : {schema_errors}")
    if "label" in next(iter(gt_map.values())):
        print(f"- State Hash Accuracy        : {metrics['state_acc'] * 100:.2f}% ({state_matches}/{n})")
        print(f"- Tool Name F1               : {metrics['tool_name_f1'] * 100:.2f}%")
        print(f"- Full Tool Call F1          : {metrics['tool_call_f1'] * 100:.2f}%")
        print(f"- Điểm tổng hợp (50/50)      : {metrics['combined_score'] * 100:.2f}%")
    print("=" * 55)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chấm điểm file submission.jsonl trên tập train hoặc kiểm tra schema trên public_test")
    ap.add_argument("submission", help="Đường dẫn tới file submission.jsonl")
    ap.add_argument("--split", default="train", choices=["train", "public_test"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--verbose-errors", type=int, default=5)
    args = ap.parse_args()
    evaluate_predictions(args.submission, root=args.root, split=args.split, verbose_errors=args.verbose_errors)
