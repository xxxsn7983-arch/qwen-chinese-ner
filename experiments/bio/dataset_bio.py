# dataset_bio.py
# 读取 jsonl：{"text": "...", "label": {"scene":["海豚馆"]}}
# 先做字符级 BIO，再对齐到 tokenizer 的 token（依赖 offset_mapping）

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset

from labels import label2id


def find_all_occurrences(text: str, sub: str) -> List[int]:
    """返回 sub 在 text 中所有出现位置的起始索引（允许重复出现）。"""
    if not sub:
        return []
    starts = []
    i = 0
    while True:
        j = text.find(sub, i)
        if j == -1:
            break
        starts.append(j)
        i = j + 1  # 允许重叠出现（按需求也可以 j+len(sub)）
    return starts


def build_char_bio_tags(text: str, label: Dict[str, List[str]]) -> List[str]:
    """
    把 {"scene":["海豚馆"], "company":["阿里巴巴"]} 转成字符级 BIO 标签序列（长度=len(text)）
    规则：
    - 长实体优先
    - 不覆盖已有标签（避免重叠）
    """
    n = len(text)
    tags = ["O"] * n

    # 收集所有(etype, ent) 并按 ent 长度降序
    pairs: List[Tuple[str, str]] = []
    for etype, ents in (label or {}).items():
        if not isinstance(ents, list):
            continue
        for ent in ents:
            if isinstance(ent, str) and ent.strip():
                pairs.append((etype, ent))

    pairs.sort(key=lambda x: len(x[1]), reverse=True)

    for etype, ent in pairs:
        starts = find_all_occurrences(text, ent)
        L = len(ent)

        for s in starts:
            e = s + L
            if s < 0 or e > n:
                continue

            # 检查是否会覆盖已标注字符
            if any(tags[k] != "O" for k in range(s, e)):
                continue

            tags[s] = f"B-{etype}"
            for k in range(s + 1, e):
                tags[k] = f"I-{etype}"

    return tags


def align_tags_to_tokens(
    text: str,
    char_tags: List[str],
    offsets: List[Tuple[int, int]],
    special_label_id: int = -100,
) -> List[int]:
    """
    offsets: tokenizer 返回的 offset_mapping
    把字符标签对齐到 token 标签：
    - special token/空offset -> -100
    - 普通 token -> 取覆盖字符区间内“第一个非O标签”，否则 O
    """
    assert len(char_tags) == len(text), "char_tags 长度必须等于文本长度"

    token_label_ids: List[int] = []

    for (s, e) in offsets:
        # special tokens 常见 offset = (0,0) 或 (None,None)
        if s is None or e is None or s == e:
            token_label_ids.append(special_label_id)
            continue

        s = max(0, s)
        e = min(len(text), e)

        if s >= e:
            token_label_ids.append(special_label_id)
            continue

        covered = char_tags[s:e]
        # 取第一个非 O，否则 O
        tag = "O"
        for t in covered:
            if t != "O":
                tag = t
                break

        token_label_ids.append(label2id.get(tag, label2id["O"]))

    return token_label_ids


class NerBioJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 256,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" not in obj:
                    continue
                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.samples[idx]
        text: str = obj["text"]
        label: Dict[str, List[str]] = obj.get("label", {}) or {}

        # 1) 字符级 BIO
        char_tags = build_char_bio_tags(text, label)

        # 2) tokenizer + offset_mapping 对齐到 token
        #    注意：需要 fast tokenizer 才稳定支持 offsets
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors=None,
        )

        if "offset_mapping" not in enc:
            raise RuntimeError(
                "你的 tokenizer 不支持 return_offsets_mapping。"
                "请确认使用的是 *TokenizerFast（例如 AutoTokenizer 默认通常是 Fast）。"
            )

        offsets = enc["offset_mapping"]
        # transformers 有时返回 list[tuple]，有时返回 list[list]
        offsets = [(int(a), int(b)) for a, b in offsets]

        labels = align_tags_to_tokens(text, char_tags, offsets, special_label_id=-100)

        # 清理 offset_mapping（训练用不到）
        enc.pop("offset_mapping", None)

        # 输出
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": text,
        }
        return item


@dataclass
class NerBioCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 动态 padding
        max_len = max(x["input_ids"].shape[0] for x in batch)

        input_ids, attention_mask, labels = [], [], []
        texts = []

        for x in batch:
            L = x["input_ids"].shape[0]
            pad = max_len - L

            input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([x["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            texts.append(x["text"])

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
            "texts": texts,
        }
