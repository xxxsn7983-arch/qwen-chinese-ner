import json
import os

def trans(file_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 用 "w" 覆盖写，避免重复累积
    with open(save_path, "w", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line.get("text", "")
                label = line.get("label", {})  # 兼容 test 没 label

                trans_label = {}
                for key, items in label.items():
                    trans_label[key] = list(items.keys())

                trans_obj = {
                    "text": text,
                    "label": trans_label
                }
                w.write(json.dumps(trans_obj, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    trans("raw_data/train.json", "data/train.json")
    trans("raw_data/dev.json", "data/val.json")

