# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== 路径配置 =====
BASE_MODEL = "/share/Qwen2.5-0.5B-Instruct"
LORA_PATH = "output_lora_v4/best"
MERGED_PATH = "merged_model_v4"

# ===== 设备：强烈建议 CPU merge，更稳 =====
device = "cpu"

print("Base model:", BASE_MODEL)
print("LoRA adapter:", LORA_PATH)
print("Save merged model to:", MERGED_PATH)
print("Using device:", device)

# ===== 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# ===== 加载 base model =====
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,   # CPU 下也可以，省内存
).to(device)

# ===== 加载 LoRA =====
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
)

# ===== 合并 LoRA 权重 =====
print("Merging LoRA into base model...")
model = model.merge_and_unload()

# ===== 保存 =====
os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print("✅ Merge complete.")
print("Merged model saved at:", MERGED_PATH)
