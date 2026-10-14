# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
import torch
import json

LABEL_LIST = ["address", "book", "company", "game", "government",
              "movie", "name", "organization", "position", "scene"]

SYSTEM_PROMPT = (
    "你是一个中文命名实体识别(NER)抽取器。\n"
    "任务：从用户输入的文本中抽取实体，并按实体类型输出。\n\n"
    "【输出规则（必须严格遵守）】：\n"
    "1) 只能输出一行内容\n"
    "2) 只能输出一个JSON对象，禁止解释、禁止换行、禁止代码块、禁止任何多余文本\n"
    "【实体类型（只能从以下10个中选择，禁止新增）】：\n"
    "address, book, company, game, government, movie, name, organization, position, scene\n\n"
    "【JSON格式】：\n"
    "{\"label\":[\"实体1\",\"实体2\"], ...}\n"
    "【实体抽取规则】：\n"
    "1) 实体必须是原文中的连续子串，保持原样，不得改写、推断或翻译\n"
    "2) 同一实体在同一label下只保留一次\n"
    "3) 实体按在原文中出现的顺序排列\n"
    "4) 如果对某个片段是否为实体不确定，宁可不输出\n"
)

class NerDatasetV2(Dataset):
    """
    改进点：
    - 训练/推理 prompt 完全一致
    - 不在样本级 padding（交给 collator 动态 padding，更快）
    - labels 对 padding 全部 -100（避免模型学 pad）
    - 目标文本加 <END>（让模型学会停）
    - 可选：cache_tokenized=True 会把 tokenize 结果缓存到内存，进一步提速
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_source_length: int = 256,
        max_target_length: int = 128,
        cache_tokenized: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.cache_tokenized = cache_tokenized

        self.data: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                label = obj["label"]  # dict
                label_str = json.dumps(label, ensure_ascii=False)

                self.data.append({"text": text, "label": label_str})

        print("data load, size:", len(self.data))

        # 可选缓存
        self._cache: Optional[List[Dict[str, Any]]] = None
        if self.cache_tokenized:
            self._cache = [self._build_one(x["text"], x["label"]) for x in self.data]
            print("tokenized cache built:", len(self._cache))

    def _build_one(self, text: str, label_str: str) -> Dict[str, Any]:
        # 构造对话 prompt（与推理一致）
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # instruction: 只截断，不 padding
        ins = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_source_length,
        )

        # response: 只截断，不 padding
        tgt = self.tokenizer(
            label_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length,
        )

        # 拼接：input_ids = instruction + target
        input_ids = ins["input_ids"] + tgt["input_ids"]
        attention_mask = [1] * len(input_ids)

        # labels：instruction 部分全 -100；target 部分为 token ids
        labels = [-100] * len(ins["input_ids"]) + tgt["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache[index]
        item = self.data[index]
        return self._build_one(item["text"], item["label"])

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class DataCollatorForCausalLM:
    """
    动态 padding：
    - input_ids 用 pad_token_id
    - attention_mask padding 为 0
    - labels padding 为 -100（不计入loss）
    """
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # 某些模型 pad_token_id 为空，常见做法：用 eos 代替
            pad_id = self.tokenizer.eos_token_id

        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            L = len(f["input_ids"])
            pad_len = max_len - L

            batch_input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
