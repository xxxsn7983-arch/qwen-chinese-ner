# -*- coding: utf-8 -*-
import json
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ALLOWED = {"address","book","company","game","government","movie","name","organization","position","scene"}

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def dump_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

SYSTEM_PROMPT = (
    "中文NER抽取器。只能输出一个JSON对象或{}，禁止解释/换行/代码块/任何多余文本。"
    "label仅限 address, book, company, game, government, movie, name, organization, position, scene。"
    "实体必须是原文连续子串，完整保留；同类去重并按出现顺序。"
    "对明显确定的人名/机构/地名/作品名请务必抽取；不确定的片段才不要输出。"
    "不要输出外号/昵称/简称/泛称（如 枪手/球迷 等）。"
    "position只输出明确职位头衔（如 总裁/CEO/经理/主席/主任/局长/院长/校长/教授/部长/书记/委员）。"
    "输出必须以 '}' 结束，输出后立刻停止。"
)


def extract_first_json_obj(s: str):
    s = s.strip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def normalize_pred(pred_json):
    if not isinstance(pred_json, dict):
        return {}

    # ===== 精度优先：来自 val 的高频 FP 经验规则 =====
    BAD_GENERIC = {
        "organization": {"枪手"},
        "company": {"银行"},
        "book": {"晨报"},
        "position": {"专家", "老师", "记者", "作者", "负责人"},
    }

    COMPANY_SUFFIX = (
        "公司", "集团", "银行", "证券", "基金", "保险",
        "大学", "学院", "研究院", "中心", "委员会", "局", "部", "厅", "院"
    )

    out = {}

    for k, v in pred_json.items():
        if k in ALLOWED and isinstance(v, list):
            seen = set()
            clean = []
            for x in v:
                if isinstance(x, str) and x not in seen:
                    seen.add(x)
                    clean.append(x)

            if clean:
                out[k] = clean
    return out

def main():
    model_path = "merged_model_v4"
    test_path = "data/val.json"  # JSONL：每行 {"id":..., "text":...}
    out_path = "pred_val_v4.jsonl"  # 建议 inference 输出也用 jsonl

    # ===== 速度相关超参（只做 1 + 2）=====
    batch_size = 16                  # MPS 建议 4~8；CUDA 可 8~32
    max_new_tokens = 48            # ✅ 只改这个：减少废话空间
    # ====================================

    device = pick_device()
    print("Using device:", device)
    print(f"batch_size={batch_size}, max_new_tokens={max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type in ("cuda", "mps") else None,
    ).to(device)
    model.eval()

    test_data = load_jsonl(test_path)
    #测试用
    # test_data = test_data[:200]

    results = []

    # 统计
    n_total = 0
    n_json_ok = 0
    sum_gen_tokens = 0
    sum_seconds = 0.0

    t_all0 = time.time()

    # ====== 分 batch 处理 ======
    for i in tqdm(range(0, len(test_data), batch_size), desc="Infer(batch)"):
        batch = test_data[i:i+batch_size]
        texts = [x["text"] for x in batch]
        ids = [x.get("id", None) for x in batch]

        t0 = time.time()

        # 构造 prompts（每条一份）
        prompts = []
        for text_in in texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_in},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        # ✅ 批量 tokenize（padding 对齐）
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # gen: [B, prompt+new]
        prompt_len = inputs["input_ids"].shape[1]
        gen_part = gen[:, prompt_len:]  # [B, new_tokens]
        # 每条生成 token 数（去掉 pad）
        # 注意：generate 输出里 pad_token 可能也出现，这里用“非pad”的长度更接近真实
        gen_tokens_each = [int(gen_part.shape[1])] * gen_part.shape[0]

        # ✅ 批量 decode（skip_special_tokens=True 更干净）
        pred_texts = tokenizer.batch_decode(gen_part, skip_special_tokens=True)

        dt = time.time() - t0

        # 逐条解析
        for j, pred_text in enumerate(pred_texts):
            pred_text = pred_text.strip()

            pred_str = extract_first_json_obj(pred_text) or "{}"
            pred_json = None
            ok = False
            try:
                pred_json = json.loads(pred_str)
                pred_json = normalize_pred(pred_json)
                ok = True
            except Exception:
                pred_json = None

            results.append({
                "id": ids[j],
                "text": texts[j],
                "pred": pred_str,
                "pred_json": pred_json,
                "gen_tokens": int(gen_tokens_each[j]),
            })

            n_total += 1
            sum_gen_tokens += int(gen_tokens_each[j])
            if ok:
                n_json_ok += 1

        sum_seconds += dt

        # 每 100 条简单报一下
        if n_total % 100 == 0:
            avg_tokens = sum_gen_tokens / max(n_total, 1)
            json_rate = n_json_ok / max(n_total, 1)
            # dt 是这一批的时间，换算成每条
            per_item = dt / max(len(batch), 1)
            tqdm.write(f"[Stats @ {n_total}] json_rate={json_rate:.3f} avg_gen_tokens={avg_tokens:.1f} sec/item≈{per_item:.3f}")

    dump_jsonl(out_path, results)

    t_all = time.time() - t_all0
    avg_tokens = sum_gen_tokens / max(n_total, 1)
    json_rate = n_json_ok / max(n_total, 1)
    sec_per_item = t_all / max(n_total, 1)

    print("Saved:", out_path)
    print("========== Inference Summary ==========")
    print("Total samples:", n_total)
    print(f"JSON parse ok: {n_json_ok}  (rate={json_rate:.4f})")
    print(f"Avg generated tokens per sample: {avg_tokens:.2f}")
    print(f"Total seconds: {t_all:.1f}s  (sec/item≈{sec_per_item:.3f}s)")
    print("=======================================")

if __name__ == "__main__":
    main()
