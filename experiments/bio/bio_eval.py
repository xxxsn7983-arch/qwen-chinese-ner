# -*- coding: utf-8 -*-
import json
import argparse
from typing import Dict, List, Tuple, Any, Set

ALLOWED = {"address","book","company","game","government","movie","name","organization","position","scene"}

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def to_pair_set(d: Dict[str, List[str]]) -> Set[Tuple[str, str]]:
    s = set()
    for k, v in (d or {}).items():
        if k in ALLOWED and isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x:
                    s.add((k, x))
    return s

def safe_gold(label_obj: Any) -> Dict[str, List[str]]:
    if not isinstance(label_obj, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in label_obj.items():
        if k in ALLOWED and isinstance(v, list):
            # 去重但保持原顺序
            seen = set()
            clean = []
            for x in v:
                if isinstance(x, str):
                    x = x.strip()
                    if x and x not in seen:
                        seen.add(x)
                        clean.append(x)
            if clean:
                out[k] = clean
    return out

def safe_pred(pred_obj: Any) -> Dict[str, List[str]]:
    # pred 就是 inference 输出的 dict
    return safe_gold(pred_obj)

def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*p*r)/(p+r) if (p+r) > 0 else 0.0
    return p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default="data/val.json")
    ap.add_argument("--pred", type=str, required=True, help="pred jsonl from bio_inference.py")
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.pred)

    if len(gold_rows) != len(pred_rows):
        print(f"WARNING: gold={len(gold_rows)} pred={len(pred_rows)} length mismatch.")
        # 仍尝试按顺序对齐
    n = min(len(gold_rows), len(pred_rows))

    tp = fp = fn = 0
    per = {k: {"tp":0, "fp":0, "fn":0} for k in ALLOWED}

    bad = 0
    for i in range(n):
        g = safe_gold(gold_rows[i].get("label", {}))
        p = safe_pred(pred_rows[i].get("pred", {}))
        if not isinstance(pred_rows[i].get("pred", {}), dict):
            bad += 1

        g_set = to_pair_set(g)
        p_set = to_pair_set(p)

        inter = g_set & p_set
        only_p = p_set - g_set
        only_g = g_set - p_set

        tp += len(inter)
        fp += len(only_p)
        fn += len(only_g)

        for (t, _) in inter:
            per[t]["tp"] += 1
        for (t, _) in only_p:
            per[t]["fp"] += 1
        for (t, _) in only_g:
            per[t]["fn"] += 1

    P, R, F1 = prf(tp, fp, fn)

    print("=== BIO Val (strict match: type + exact entity text) ===")
    print(f"samples={n} bad_pred_json={bad}")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Micro-P={P:.4f}  Micro-R={R:.4f}  Micro-F1={F1:.4f}")
    print()
    print("=== Per-type ===")
    for t in ["address","book","company","game","government","movie","name","organization","position","scene"]:
        tp_t = per[t]["tp"]; fp_t = per[t]["fp"]; fn_t = per[t]["fn"]
        p_t, r_t, f1_t = prf(tp_t, fp_t, fn_t)
        print(f"{t:<12} P={p_t:.4f} R={r_t:.4f} F1={f1_t:.4f}  (tp={tp_t}, fp={fp_t}, fn={fn_t})")

if __name__ == "__main__":
    main()
