# eval_val_v4_with_plots.py
# -*- coding: utf-8 -*-
import os
import json
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


ALLOWED = {"address","book","company","game","government","movie","name","organization","position","scene"}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def gold_to_set(label_obj):
    """
    gold: {"address": ["台湾"], "name": ["彭小军"]}  / 也可能是 str
    -> set((type, entity_text))
    """
    s = set()
    if not isinstance(label_obj, dict):
        return s
    for t, ents in label_obj.items():
        if isinstance(ents, str):
            ents = [ents]
        if not isinstance(ents, list):
            continue
        for e in ents:
            if isinstance(e, str) and e:
                s.add((t, e))
    return s


def pred_to_set(pred_json):
    """
    pred_json: {"address": ["台湾"], "name": ["彭小军"]} / 也可能是 str
    -> set((type, entity_text))
    """
    s = set()
    if not isinstance(pred_json, dict):
        return s
    for t, ents in pred_json.items():
        if isinstance(ents, str):
            ents = [ents]
        if not isinstance(ents, list):
            continue
        for e in ents:
            if isinstance(e, str) and e:
                s.add((t, e))
    return s


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def plot_micro_prf(out_png, P, R, F1):
    plt.figure()
    xs = ["Micro-P", "Micro-R", "Micro-F1"]
    ys = [P, R, F1]
    plt.bar(xs, ys)
    plt.ylim(0, 1.0)
    plt.title("Overall Micro Precision/Recall/F1 (Val)")
    plt.ylabel("Score")
    for i, v in enumerate(ys):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_per_type_f1(out_png, rows_sorted):
    """
    rows_sorted: list of dict with keys: type, f1
    """
    plt.figure(figsize=(10, 4))
    xs = [r["type"] for r in rows_sorted]
    ys = [r["f1"] for r in rows_sorted]
    plt.bar(xs, ys)
    plt.ylim(0, 1.0)
    plt.title("Per-type F1 (Val) — sorted")
    plt.ylabel("F1")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_per_type_prf(out_png, rows_sorted):
    """
    分组柱状图：P/R/F1
    """
    import numpy as np

    xs = [r["type"] for r in rows_sorted]
    p = [r["p"] for r in rows_sorted]
    r_ = [r["r"] for r in rows_sorted]
    f1 = [r["f1"] for r in rows_sorted]

    x = np.arange(len(xs))
    w = 0.25

    plt.figure(figsize=(12, 4))
    plt.bar(x - w, p, width=w, label="P")
    plt.bar(x,     r_, width=w, label="R")
    plt.bar(x + w, f1, width=w, label="F1")
    plt.ylim(0, 1.0)
    plt.title("Per-type Precision / Recall / F1 (Val)")
    plt.ylabel("Score")
    plt.xticks(x, xs, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_per_type_counts(out_png, rows_sorted):
    """
    堆叠柱状图：TP / FP / FN（误差分析常用）
    """
    import numpy as np

    xs = [r["type"] for r in rows_sorted]
    tp = [r["tp"] for r in rows_sorted]
    fp = [r["fp"] for r in rows_sorted]
    fn = [r["fn"] for r in rows_sorted]

    x = np.arange(len(xs))

    plt.figure(figsize=(12, 4))
    plt.bar(x, tp, label="TP")
    plt.bar(x, fp, bottom=tp, label="FP")
    bottom2 = [tp[i] + fp[i] for i in range(len(xs))]
    plt.bar(x, fn, bottom=bottom2, label="FN")

    plt.title("Per-type TP/FP/FN Counts (Val)")
    plt.ylabel("Count")
    plt.xticks(x, xs, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_per_type_csv(out_csv, rows):
    fields = ["type", "p", "r", "f1", "tp", "fp", "fn", "support"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    gold_path = "data/val.json"
    pred_path = "pred_val.jsonl"  # 你的 inference 输出

    out_dir = ensure_dir("eval_outputs")

    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)
    n = min(len(gold), len(pred))

    micro_tp = micro_fp = micro_fn = 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_type_support = defaultdict(int)  # gold 中该类实体总数（support）
    bad_parse = 0

    fp_examples = []  # (text, (t,e))
    fn_examples = []  # (text, (t,e))

    for i in range(n):
        g = gold[i]
        p = pred[i]

        text = g.get("text", "")

        gset = gold_to_set(g.get("label", {}))
        for t, _ in gset:
            per_type_support[t] += 1

        pjson = p.get("pred_json", None)
        if pjson is None:
            bad_parse += 1

        pset = pred_to_set(pjson)

        tp_set = gset & pset
        fp_set = pset - gset
        fn_set = gset - pset

        micro_tp += len(tp_set)
        micro_fp += len(fp_set)
        micro_fn += len(fn_set)

        for t, _ in tp_set:
            per_type[t]["tp"] += 1
        for t, _ in fp_set:
            per_type[t]["fp"] += 1
        for t, _ in fn_set:
            per_type[t]["fn"] += 1

        # 收集少量错误样例
        for item in list(fp_set)[:2]:
            if len(fp_examples) < 30:
                fp_examples.append((text, item))
        for item in list(fn_set)[:2]:
            if len(fn_examples) < 30:
                fn_examples.append((text, item))

    P, R, F1 = prf(micro_tp, micro_fp, micro_fn)

    print("=== Val (strict match: type + exact entity text) ===")
    print(f"samples={n}  bad_pred_json={bad_parse}")
    print(f"TP={micro_tp} FP={micro_fp} FN={micro_fn}")
    print(f"Micro-P={P:.4f}  Micro-R={R:.4f}  Micro-F1={F1:.4f}")

    # per-type rows
    rows = []
    all_types = sorted(set(list(per_type.keys()) + list(per_type_support.keys()) + list(ALLOWED)))
    for t in all_types:
        tp = per_type[t]["tp"]
        fp = per_type[t]["fp"]
        fn = per_type[t]["fn"]
        p_, r_, f1_ = prf(tp, fp, fn)
        support = per_type_support.get(t, 0)
        rows.append({
            "type": t,
            "p": round(p_, 6),
            "r": round(r_, 6),
            "f1": round(f1_, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        })

    # 打印 per-type
    print("\n=== Per-type ===")
    for r in rows:
        print(f"{r['type']:12s}  P={r['p']:.4f} R={r['r']:.4f} F1={r['f1']:.4f}  "
              f"(tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, support={r['support']})")

    # 保存表与 summary
    summary = {
        "split": "val",
        "samples": n,
        "bad_pred_json": bad_parse,
        "micro": {
            "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
            "precision": P, "recall": R, "f1": F1
        },
    }
    save_json(os.path.join(out_dir, "metrics_summary.json"), summary)
    write_per_type_csv(os.path.join(out_dir, "per_type_metrics.csv"), rows)

    # 保存少量错误样例
    save_json(os.path.join(out_dir, "fp_examples.json"), [
        {"text": t, "type": e[0], "entity": e[1]} for (t, e) in fp_examples
    ])
    save_json(os.path.join(out_dir, "fn_examples.json"), [
        {"text": t, "type": e[0], "entity": e[1]} for (t, e) in fn_examples
    ])

    # 画图
    rows_sorted = sorted(rows, key=lambda x: x["f1"], reverse=True)

    plot_micro_prf(os.path.join(out_dir, "fig_micro_prf.png"), P, R, F1)
    plot_per_type_f1(os.path.join(out_dir, "fig_per_type_f1.png"), rows_sorted)
    plot_per_type_prf(os.path.join(out_dir, "fig_per_type_prf.png"), rows_sorted)
    plot_per_type_counts(os.path.join(out_dir, "fig_per_type_counts.png"), rows_sorted)

    print("\nSaved:")
    print(" -", os.path.join(out_dir, "metrics_summary.json"))
    print(" -", os.path.join(out_dir, "per_type_metrics.csv"))
    print(" -", os.path.join(out_dir, "fig_micro_prf.png"))
    print(" -", os.path.join(out_dir, "fig_per_type_f1.png"))
    print(" -", os.path.join(out_dir, "fig_per_type_prf.png"))
    print(" -", os.path.join(out_dir, "fig_per_type_counts.png"))
    print(" -", os.path.join(out_dir, "fp_examples.json"))
    print(" -", os.path.join(out_dir, "fn_examples.json"))


if __name__ == "__main__":
    main()
