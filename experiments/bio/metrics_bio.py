# metrics_bio.py
# token acc + 实体级 span F1（BIO）

from typing import List, Tuple, Set, Dict

from labels import id2label


def bio_to_spans(tags: List[str]) -> Set[Tuple[int, int, str]]:
    """
    BIO tags -> spans
    span 表示为 (start_idx, end_idx_exclusive, type)
    """
    spans: Set[Tuple[int, int, str]] = set()
    i = 0
    n = len(tags)

    while i < n:
        t = tags[i]
        if t.startswith("B-"):
            etype = t[2:]
            start = i
            i += 1
            while i < n and tags[i] == f"I-{etype}":
                i += 1
            end = i
            spans.add((start, end, etype))
        else:
            i += 1

    return spans


def compute_token_acc(pred_ids, gold_ids, ignore_id: int = -100) -> float:
    """
    pred_ids/gold_ids: shape [B, T]
    """
    total = 0
    correct = 0
    for p_seq, g_seq in zip(pred_ids, gold_ids):
        for p, g in zip(p_seq, g_seq):
            if g == ignore_id:
                continue
            total += 1
            if p == g:
                correct += 1
    return (correct / total) if total > 0 else 0.0


def compute_span_f1(pred_ids, gold_ids, ignore_id: int = -100) -> Dict[str, float]:
    """
    实体级 F1：把 token-level BIO 序列抽成 spans 做集合比对。
    注意：这里是 token-span，不是字符-span；作业通常够用，且比只看 token acc 好看。
    """
    tp = fp = fn = 0

    for p_seq, g_seq in zip(pred_ids, gold_ids):
        p_tags: List[str] = []
        g_tags: List[str] = []

        for p, g in zip(p_seq, g_seq):
            if g == ignore_id:
                continue
            p_tags.append(id2label[int(p)])
            g_tags.append(id2label[int(g)])

        p_spans = bio_to_spans(p_tags)
        g_spans = bio_to_spans(g_tags)

        tp += len(p_spans & g_spans)
        fp += len(p_spans - g_spans)
        fn += len(g_spans - p_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
