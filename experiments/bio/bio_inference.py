# -*- coding: utf-8 -*-
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from peft import LoraConfig, TaskType, get_peft_model

ALLOWED = {"address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ckpt_state_dict(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    兼容你现在的 best.pt 格式：
    torch.load(best.pt) -> dict(keys=['epoch','best_f1','model_state','config'])

    返回：
      - model_state: 权重 dict
      - meta: 其他信息（epoch / best_f1 / config）
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint is not a dict: {ckpt_path}, type={type(obj)}")

    if "model_state" not in obj or not isinstance(obj["model_state"], dict):
        raise ValueError(f"Checkpoint missing 'model_state': {ckpt_path}, keys={list(obj.keys())}")

    meta = {
        "epoch": obj.get("epoch", None),
        "best_f1": obj.get("best_f1", None),
        "config": obj.get("config", None),
    }
    return obj["model_state"], meta


def apply_ckpt_config_to_model(model, ckpt_config: Optional[Dict[str, Any]]):
    """
    把训练时保存的 config（如果有）覆盖到 model.config 上，保证：
      - num_labels
      - id2label / label2id
    否则 decode 可能类型错乱。
    """
    if not isinstance(ckpt_config, dict) or len(ckpt_config) == 0:
        return

    cfg = model.config

    # num_labels
    if "num_labels" in ckpt_config and isinstance(ckpt_config["num_labels"], int):
        cfg.num_labels = ckpt_config["num_labels"]

    # id2label / label2id
    if "id2label" in ckpt_config and isinstance(ckpt_config["id2label"], dict) and ckpt_config["id2label"]:
        cfg.id2label = {int(k): v for k, v in ckpt_config["id2label"].items()}

    if "label2id" in ckpt_config and isinstance(ckpt_config["label2id"], dict) and ckpt_config["label2id"]:
        cfg.label2id = {str(k): int(v) for k, v in ckpt_config["label2id"].items()}


def id2label_from_model(model) -> Dict[int, str]:
    cfg = model.config
    if hasattr(cfg, "id2label") and isinstance(cfg.id2label, dict) and len(cfg.id2label) > 0:
        return {int(k): v for k, v in cfg.id2label.items()}

    # fallback：构造一个 L{i}
    num_labels = getattr(cfg, "num_labels", None)
    if num_labels is None:
        if hasattr(model, "classifier") and hasattr(model.classifier, "out_features"):
            num_labels = int(model.classifier.out_features)
        elif hasattr(model, "score") and hasattr(model.score, "out_features"):
            num_labels = int(model.score.out_features)
        else:
            raise ValueError("Cannot infer num_labels from model. Please set config.id2label/num_labels.")
    return {i: f"L{i}" for i in range(int(num_labels))}


def bio_decode_entities(
    text: str,
    offsets: List[Tuple[int, int]],
    label_ids: List[int],
    id2label: Dict[int, str],
) -> Dict[str, List[str]]:
    """
    输入：原文 text、每个 token 的 (start,end) 偏移、每个 token 的 label_id
    输出：{"type":[实体字符串...]}
    约定 label 形如：B-company / I-company / O
    """
    out: Dict[str, List[str]] = {}

    cur_type = None
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_type, cur_start, cur_end
        if cur_type is None or cur_start is None or cur_end is None:
            return
        ent = text[cur_start:cur_end]
        if ent and cur_type in ALLOWED:
            out.setdefault(cur_type, [])
            if ent not in out[cur_type]:
                out[cur_type].append(ent)
        cur_type = None
        cur_start = None
        cur_end = None

    for (s, e), lid in zip(offsets, label_ids):
        if s == e:
            continue  # special token / padding
        lab = id2label.get(int(lid), "O")
        if lab == "O" or lab.endswith("-O"):
            flush()
            continue

        if "-" in lab:
            prefix, typ = lab.split("-", 1)
        else:
            prefix, typ = "O", ""

        if prefix == "B":
            flush()
            cur_type = typ
            cur_start = s
            cur_end = e
        elif prefix == "I":
            if cur_type == typ and cur_start is not None:
                cur_end = e
            else:
                flush()
                cur_type = typ
                cur_start = s
                cur_end = e
        else:
            flush()

    flush()
    return out


@torch.no_grad()
def infer_batch(model, tokenizer, texts: List[str], device, max_length: int) -> List[Dict[str, List[str]]]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")  # [B, T, 2]
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    logits = out.logits  # [B, T, C]
    pred = logits.argmax(dim=-1).cpu()  # [B, T]

    id2lab = id2label_from_model(model)
    results = []
    for i in range(len(texts)):
        offs_i = [(int(a), int(b)) for a, b in offsets[i].tolist()]
        pred_i = pred[i].tolist()
        ent = bio_decode_entities(texts[i], offs_i, pred_i, id2lab)
        results.append(ent)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="/share/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ckpt", type=str, default="ckpt_bio/best.pt")
    ap.add_argument("--input", type=str, default="data/val.json")
    ap.add_argument("--output", type=str, default="pred_bio_val.jsonl")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)

    # ✅ LoRA 超参：必须与训练一致
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)

    args = ap.parse_args()

    device = pick_device()
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 1) base token classification
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # 2) 先把 ckpt 里的 config 覆盖到 model.config（保证 num_labels / id2label 一致）
    sd, meta = load_ckpt_state_dict(args.ckpt)
    apply_ckpt_config_to_model(model, meta.get("config"))

    # 3) 注入 LoRA（必须和训练一致）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # ✅ 跟训练一致
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # 4) 加载 LoRA 训练 ckpt（可能存在多一层 base_model 前缀，需要 remap）
    model_keys = set(model.state_dict().keys())
    sd_keys = list(sd.keys())

    # 统计原始匹配数
    raw_hit = sum(1 for k in sd_keys if k in model_keys)
    print("Raw key hit:", raw_hit, "/", len(sd_keys))

    def remap_state_dict(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sd_out = {}
        for k, v in sd_in.items():
            nk = k

            # 你现在 ckpt 典型前缀：base_model.base_model.model...
            # 目标通常是：base_model.model...
            if nk.startswith("base_model.base_model."):
                nk = "base_model." + nk[len("base_model.base_model."):]

            # 再兜一层：有些情况下还需要去掉多余的 base_model.
            # （谨慎做：只处理开头连续重复）
            while nk.startswith("base_model.base_model."):
                nk = "base_model." + nk[len("base_model.base_model."):]

            sd_out[nk] = v
        return sd_out

    sd2 = remap_state_dict(sd)
    sd2_keys = list(sd2.keys())
    hit2 = sum(1 for k in sd2_keys if k in model_keys)
    print("Remap key hit:", hit2, "/", len(sd2_keys))

    # 用匹配更多的那份去 load
    sd_to_load = sd2 if hit2 > raw_hit else sd

    missing, unexpected = model.load_state_dict(sd_to_load, strict=False)
    print(f"Loaded ckpt: {args.ckpt} (epoch={meta.get('epoch')}, best_f1={meta.get('best_f1')})")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    # 如果还是 0 命中，直接打印少量对比，方便你定位
    if hit2 == 0 and raw_hit == 0:
        print("\n[Debug] ckpt keys head:")
        for k in sd_keys[:10]:
            print(" ", k)
        print("[Debug] model keys head:")
        for k in list(model_keys)[:10]:
            print(" ", k)

    print(f"Loaded ckpt: {args.ckpt} (epoch={meta.get('epoch')}, best_f1={meta.get('best_f1')})")
    if missing:
        print("Missing keys:", len(missing))
    if unexpected:
        print("Unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()

    rows = load_jsonl(args.input)
    results = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [x["text"] for x in batch]
        preds = infer_batch(model, tokenizer, texts, device, args.max_length)
        for x, p in zip(batch, preds):
            results.append({
                "id": x.get("id", None),
                "text": x["text"],
                "pred": p
            })

    dump_jsonl(args.output, results)
    print("Saved:", args.output, "num_samples=", len(results))


if __name__ == "__main__":
    main()
