# train_bio.py
# Qwen2.5 encoder + token classification head（BIO）
# 支持：fp16、梯度累积、TensorBoard、保存 best checkpoint
# 可选：LoRA（peft），默认关闭

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from dataset_bio import NerBioJsonlDataset, NerBioCollator
from labels import num_labels, id2label
from metrics_bio import compute_token_acc, compute_span_f1


class QwenTokenClassifier(nn.Module):
    def __init__(self, base_model, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # 兼容 outputs
        hidden = out.last_hidden_state  # [B,T,H]
        logits = self.classifier(self.dropout(hidden))  # [B,T,C]

        loss = None
        if labels is not None:
            # labels: [B,T]，ignore_index=-100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}


def maybe_apply_lora(model, use_lora: bool, r: int, alpha: int, dropout: float):
    if not use_lora:
        return model

    from peft import LoraConfig, get_peft_model, TaskType

    # Qwen 常见可注入模块名（不同版本可能略有差异）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # 我们用 encoder 输出做 token cls
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    total_loss = 0.0
    steps = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"]
        logits = out["logits"]

        total_loss += float(loss.item())
        steps += 1

        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        golds = labels.detach().cpu().tolist()

        all_pred.extend(preds)
        all_gold.extend(golds)

    avg_loss = total_loss / max(1, steps)
    token_acc = compute_token_acc(all_pred, all_gold, ignore_id=-100)
    span = compute_span_f1(all_pred, all_gold, ignore_id=-100)

    return {
        "val_loss": avg_loss,
        "token_acc": token_acc,
        "span_precision": span["precision"],
        "span_recall": span["recall"],
        "span_f1": span["f1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/share/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", type=str, default="data/train.json")
    parser.add_argument("--val_path", type=str, default="data/val.json")
    parser.add_argument("--out_dir", type=str, default="ckpt_bio")
    parser.add_argument("--log_dir", type=str, default="logs_bio")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)  # 2080Ti：8 通常比较稳
    parser.add_argument("--grad_accum", type=int, default=2)  # 有需要再加大
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--fp16", action="store_true", default=True)

    # LoRA（可选）
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Qwen2.5 tokenizer pad_token 可能为空，保证可用
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = NerBioJsonlDataset(args.train_path, tokenizer, max_length=args.max_length)
    val_ds = NerBioJsonlDataset(args.val_path, tokenizer, max_length=args.max_length)

    collate = NerBioCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    base = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)

    # hidden_size 通常在 config 里
    hidden_size = getattr(base.config, "hidden_size", None) or getattr(base.config, "n_embd", None)
    if hidden_size is None:
        raise RuntimeError("无法从 config 读取 hidden_size，请把 base.config 打印出来我帮你对齐。")

    # 可选 LoRA（注入到 base 模型）
    base = maybe_apply_lora(base, args.lora, args.lora_r, args.lora_alpha, args.lora_dropout)

    model = QwenTokenClassifier(base, hidden_size=hidden_size, num_labels=num_labels, dropout=0.1)
    model.to(device)

    # 只训练 LoRA/分类头：如果不用 LoRA，也可以全参训练（0.5B 在 2080Ti 可能吃紧）
    if args.lora:
        # base 已经是 peft 包装，默认只开放 LoRA 权重为可训练
        pass

    # optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    writer = SummaryWriter(args.log_dir)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best_f1 = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.item())

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                writer.add_scalar("train/loss", running_loss, global_step)
                running_loss = 0.0

                pbar.set_postfix({"step": global_step, "lr": scheduler.get_last_lr()[0]})

        # eval each epoch
        metrics = evaluate(model, val_loader, device)
        writer.add_scalar("val/loss", metrics["val_loss"], epoch)
        writer.add_scalar("val/token_acc", metrics["token_acc"], epoch)
        writer.add_scalar("val/span_f1", metrics["span_f1"], epoch)

        print(f"\n[Epoch {epoch}] val_loss={metrics['val_loss']:.4f} "
              f"token_acc={metrics['token_acc']:.4f} span_f1={metrics['span_f1']:.4f}")

        # save best
        if metrics["span_f1"] > best_f1:
            best_f1 = metrics["span_f1"]
            save_path = os.path.join(args.out_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "best_f1": best_f1,
                "model_state": model.state_dict(),
                "config": {
                    "model_path": args.model_path,
                    "max_length": args.max_length,
                    "num_labels": num_labels,
                    "id2label": id2label,
                }
            }, save_path)
            print(f"✅ Saved best to {save_path} (span_f1={best_f1:.4f})")

    writer.close()
    print(f"\nDone. Best span_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
