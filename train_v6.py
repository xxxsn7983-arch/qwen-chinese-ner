# train_v6.py
# -*- coding: utf-8 -*-
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dataset import NerDatasetV2, DataCollatorForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def validate_model(model, device, val_loader):
    model.eval()
    running_loss = 0.0
    n = 0
    with torch.no_grad():
        for data in tqdm(val_loader, file=sys.stdout, desc="Validation Data"):
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            labels = data["labels"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            running_loss += float(outputs.loss.item())
            n += 1
    return running_loss / max(n, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    model_output_dir,
    writer,
    grad_accum_steps=8,
    log_every_n_steps=20,
):
    batch_step = 0
    t_prev_iter_end = time.time()

    best_val_loss = float("inf")
    best_dir = os.path.join(model_output_dir, "best")

    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc=f"Train Epoch: {epoch}")):
            # load time
            t_after_load = time.time()
            load_t = t_after_load - t_prev_iter_end

            # compute time
            t_compute_start = time.time()

            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            labels = data["labels"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / grad_accum_steps
            loss.backward()

            do_step = ((index + 1) % grad_accum_steps == 0) or (index == len(train_loader) - 1)
            if do_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            t_after_step = time.time()
            compute_t = t_after_step - t_compute_start
            total_t = t_after_step - t_prev_iter_end
            t_prev_iter_end = t_after_step

            writer.add_scalar("Loss/train", float(outputs.loss.item()), batch_step)
            batch_step += 1

            if index % log_every_n_steps == 0:
                tqdm.write(
                    f"[epoch {epoch} step {index}] "
                    f"load={load_t:.3f}s  compute={compute_t:.3f}s  total={total_t:.3f}s  "
                    f"loss={float(outputs.loss.item()):.4f}  "
                    f"(accum {grad_accum_steps}, step_update={do_step})"
                )

            if index % 100 == 0 or index == len(train_loader) - 1:
                time2 = time.time()
                avg_step = float(time2 - time1) / float(index + 0.0001)
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {float(outputs.loss.item()):.6f} ; "
                    f"avg step time: {avg_step:.4f}s"
                )

        # ===== epoch end: val + save =====
        val_loss = validate_model(model, device, val_loader)
        writer.add_scalar("Loss/val", float(val_loss), epoch)
        print(f"[Epoch {epoch}] val loss: {val_loss:.6f}")

        epoch_dir = os.path.join(model_output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        print(f"[Epoch {epoch}] model saved to {epoch_dir}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            print(f"[Best] model updated at {best_dir}, val_loss={best_val_loss:.6f}")


def build_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main():
    model_name = os.environ.get("BASE_MODEL", os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))

    # ✅ v6: 使用过采样后的训练集
    train_json_path = "data/train_v6.json"
    val_json_path = "data/val.json"

    max_source_length = 256
    max_target_length = 160

    # ✅ v6: 过采样后先训 2 轮更稳
    epochs = 2
    batch_size = 2
    lr = 1e-4
    grad_accum_steps = 8

    model_output_dir = "output_lora_v6"
    logs_dir = "logs_v6"

    device = build_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type in ("mps", "cuda") else None,
    )

    # LoRA：先保持和 v4 一致，方便对照（只变“数据采样策略”）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    collator = DataCollatorForCausalLM(tokenizer)

    print("Start Load Train Data...")
    training_set = NerDatasetV2(
        train_json_path,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        cache_tokenized=False,
    )
    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    print("Start Load Validation Data...")
    val_set = NerDatasetV2(
        val_json_path,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        cache_tokenized=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    writer = SummaryWriter(logs_dir)

    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=lr,
        # ✅ v6: 不加 weight_decay（避免把 LoRA 更新压回去）
        # weight_decay=0.0,
    )

    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer,
        grad_accum_steps=grad_accum_steps,
        log_every_n_steps=20,
    )

    writer.close()
    print("Done.")
    print("LoRA saved at:", model_output_dir)
    print("Tensorboard logs at:", logs_dir)


if __name__ == "__main__":
    main()
