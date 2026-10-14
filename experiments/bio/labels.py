# labels.py
ENTITY_TYPES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene"
]

LABELS = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]

label2id = {lab: i for i, lab in enumerate(LABELS)}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(LABELS)
