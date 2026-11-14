"""Local scorer and validator for ZAC 2026 LifeMate using zaic_tools."""
from __future__ import annotations

import argparse
import json
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Any
from collections import defaultdict


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
        overlap = len(t_pred & t_gt)
        p = overlap / len(t_pred)
        r = overlap / len(t_gt)
        if p + r == 0:
            return False
        token_f1 = (2 * p * r) / (p + r)
        return token_f1 > 0.6 or t_gt.issubset(t_pred)
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


def _classify_calls(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return "REJECT"
    if any(c.get("name") == "sys.ask_missing_info" for c in calls):
        return "ASK_INFO"
    return "DOABLE"


def evaluate_predictions(
    submission_path: str | Path,
    root: str | Path = ".",
    split: str = "train",
    verbose_errors: int = 5,
    breakdown: bool = False,
) -> dict[str, float]:
    root = Path(root)
    zaic_tools, FREE_TEXT_KEYS, _canon_scalar, _tokens, _catalog_tools, schema_validate = _load_zaic_tools(root, split)
    ToolEnv = zaic_tools.ToolEnv

    catalog = json.loads((root / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))

    # Check seen train apps for comparison
    train_catalog_p = root / "train" / "train_tools_catalog.json"
    train_app_ids = set()
    if train_catalog_p.exists():
        try:
            t_cat = json.loads(train_catalog_p.read_text(encoding="utf-8"))
            train_app_ids = {a.get("id") for a in t_cat.get("apps", []) if a.get("id")}
        except Exception:
            pass

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
    hash_mismatches = 0
    total_calls = 0
    invalid_calls = 0
    state_matches = 0
    name_f1_sum = 0.0
    full_f1_sum = 0.0
    official_score_sum = 0.0
    printed_err = 0

    # Diagnostic tracking
    triage_classes = ["DOABLE", "ASK_INFO", "REJECT"]
    triage_matrix = {g: {p: 0 for p in triage_classes} for g in triage_classes}
    app_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "state_match": 0, "full_f1_sum": 0.0})
    tool_err_counts: dict[str, list[str]] = defaultdict(list)

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
            total_calls += 1
            call_res = env_pred.call(c.get("name", ""), c.get("arguments") or {})
            if not call_res.get("ok"):
                invalid_calls += 1
                c_name = c.get("name", "unknown")
                tool_err_counts[c_name].append(str(call_res.get("error", "unknown error")))
        replayed_hash = env_pred.get_state_hash()
        sub_hash = pred.get("state_hash")
        if sub_hash is not None and sub_hash != replayed_hash:
            hash_mismatches += 1
        if sub_hash is None:
            sub_hash = replayed_hash

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

            # Official ZAIC 2026 Hard-Gate Scoring Rule:
            # 1. State hash must match; if not -> 0.0 immediately
            # 2. If gt_calls == []: 1.0 if pred_calls == [] and st_ok else 0.0
            # 3. If gt_calls != []: ff1 if st_ok else 0.0 (0.0 if full_tp == 0)
            if not st_ok:
                sample_score = 0.0
            else:
                if not gt_calls:
                    sample_score = 1.0 if not pred_calls else 0.0
                else:
                    sample_score = ff1
            official_score_sum += sample_score

            # Triage tracking
            gt_cls = _classify_calls(gt_calls)
            pred_cls = _classify_calls(pred_calls)
            triage_matrix[gt_cls][pred_cls] += 1

            # App-level tracking (apps called in GT)
            gt_apps = set(c.get("name", "").split(".")[0] for c in gt_calls if "." in c.get("name", ""))
            for app in gt_apps:
                app_stats[app]["count"] += 1
                if st_ok:
                    app_stats[app]["state_match"] += 1
                app_stats[app]["full_f1_sum"] += ff1

            if (not st_ok or ff1 < 1.0) and printed_err < verbose_errors:
                printed_err += 1
                print(f"\n[Mismatch #{printed_err}] ID={qid} | state_ok={st_ok} | call_F1={ff1:.2f} | score={sample_score:.2f}")
                print(f"  Audio text : {gt_row.get('audio_text', '')}")
                print(f"  GT calls   : {json.dumps(gt_calls, ensure_ascii=False)}")
                print(f"  Pred calls : {json.dumps(pred_calls, ensure_ascii=False)}")

    if n == 0:
        print("Không có mẫu nào để chấm!")
        return {}

    has_labels = "label" in next(iter(gt_map.values()))
    metrics = {
        "samples": n,
        "schema_errors": schema_errors,
        "state_acc": state_matches / n if has_labels else 0.0,
        "tool_name_f1": name_f1_sum / n,
        "tool_call_f1": full_f1_sum / n,
        "official_score": official_score_sum / n if has_labels else 0.0,
        "combined_score": 0.5 * (state_matches / n) + 0.5 * (full_f1_sum / n) if has_labels else 0.0,
    }
    print("\n" + "=" * 65)
    print(f"KẾT QUẢ ĐÁNH GIÁ ({split} - {n} mẫu)")
    print("=" * 65)
    print(f"- Lỗi Schema (Schema Errors)      : {schema_errors}")
    print(f"- Hash Mismatches (Sub != Replay) : {hash_mismatches}")
    print(f"- Invalid Calls (ok=False)        : {invalid_calls} / {total_calls} ({(invalid_calls/max(total_calls,1))*100:.1f}%)")
    if has_labels:
        print(f"- State Hash Accuracy         : {metrics['state_acc'] * 100:.2f}% ({state_matches}/{n})")
        print(f"- Tool Name F1                : {metrics['tool_name_f1'] * 100:.2f}%")
        print(f"- Full Tool Call F1           : {metrics['tool_call_f1'] * 100:.2f}%")
        print(f"- ZAIC Official Score (Gate)  : {metrics['official_score'] * 100:.2f}% (Chuẩn công thức BTC)")
        print(f"- Điểm cũ (Linear 50/50)      : {metrics['combined_score'] * 100:.2f}%")
    print("=" * 65)

    if breakdown:
        if has_labels:
            print("\n" + "-" * 65)
            print("1. MA TRẬN NHẦM LẪN 3 NHÓM HÀNH ĐỘNG (ACTION TRIAGE MATRIX)")
            print("-" * 65)
            print(f"{'GT \\ Pred':<15} | {'DOABLE':<10} | {'ASK_INFO':<10} | {'REJECT':<10} | {'Tổng GT':<8} | {'Recall':<8}")
            print("-" * 65)
            total_corr = 0
            for g in triage_classes:
                row = triage_matrix[g]
                row_sum = sum(row.values())
                corr = row[g]
                total_corr += corr
                recall_str = f"{(corr / row_sum * 100):.1f}%" if row_sum > 0 else "N/A"
                print(f"{g:<15} | {row['DOABLE']:>8}   | {row['ASK_INFO']:>8}   | {row['REJECT']:>8}   | {row_sum:>7}  | {recall_str:>7}")
            print("-" * 65)
            pred_sums = {p: sum(triage_matrix[g][p] for g in triage_classes) for p in triage_classes}
            prec_strs = [
                f"{(triage_matrix[p][p] / pred_sums[p] * 100):.1f}%" if pred_sums[p] > 0 else "N/A"
                for p in triage_classes
            ]
            print(f"{'Tổng Pred':<15} | {pred_sums['DOABLE']:>8}   | {pred_sums['ASK_INFO']:>8}   | {pred_sums['REJECT']:>8}   | {n:>7}  | Acc: {(total_corr / n * 100):.1f}%")
            print(f"{'Precision':<15} | {prec_strs[0]:>8}   | {prec_strs[1]:>8}   | {prec_strs[2]:>8}   |")

            print("\n" + "-" * 65)
            print("2. ĐỘ CHÍNH XÁC CHI TIẾT THEO ỨNG DỤNG (PER-APP BREAKDOWN)")
            print("-" * 65)
            print(f"{'Ứng dụng (App ID)':<24} | {'Trạng thái':<10} | {'Số mẫu':<8} | {'State Acc':<10} | {'Call F1':<10}")
            print("-" * 65)
            sorted_apps = sorted(
                app_stats.items(),
                key=lambda item: (item[1]["state_match"] / max(item[1]["count"], 1), item[1]["full_f1_sum"] / max(item[1]["count"], 1))
            )
            for app_id, s in sorted_apps:
                seen_tag = "[Seen]" if app_id in train_app_ids else "[NEW]"
                cnt = s["count"]
                st_pct = (s["state_match"] / cnt) * 100
                f1_pct = (s["full_f1_sum"] / cnt) * 100
                print(f"{app_id:<24} | {seen_tag:<10} | {cnt:<8} | {st_pct:>8.1f}%  | {f1_pct:>8.1f}%")
            print("-" * 65)

        if tool_err_counts:
            print("\n" + "-" * 65)
            print(f"3. CÁC TOOL GỌI LỖI (INVALID CALLS: {invalid_calls} lần)")
            print("-" * 65)
            sorted_errs = sorted(tool_err_counts.items(), key=lambda x: len(x[1]), reverse=True)
            for tool_name, err_list in sorted_errs[:15]:
                sample_err = err_list[0]
                if len(sample_err) > 55:
                    sample_err = sample_err[:52] + "..."
                print(f"- {tool_name:<28}: {len(err_list):>3} lỗi | Lỗi mẫu: {sample_err}")
            print("-" * 65)
        else:
            print("\n[OK] Không có lệnh gọi tool nào bị từ chối (ok=False)!")

    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chấm điểm file submission.jsonl trên tập train hoặc kiểm tra schema trên public_test")
    ap.add_argument("submission", help="Đường dẫn tới file submission.jsonl")
    ap.add_argument("--split", default="train", choices=["train", "public_test", "private_test"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--verbose-errors", type=int, default=5)
    ap.add_argument("--breakdown", action="store_true", help="Hiển thị phân tích ma trận nhầm lẫn triage và độ chính xác theo app")
    args = ap.parse_args()
    evaluate_predictions(args.submission, root=args.root, split=args.split, verbose_errors=args.verbose_errors, breakdown=args.breakdown)
