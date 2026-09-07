"""Print one ChaosNLI example to check the jsonl fields."""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = os.path.join(PROJECT_ROOT, "ChaosNLI", "data", "chaosNLI_v1.0", "chaosNLI_mnli_m.jsonl")

with open(path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print("Number of examples:", len(data))
example = data[0]
print("ID:", example["uid"])
print("Premise:", example["example"]["premise"])
print("Hypothesis:", example["example"]["hypothesis"])
print("Old label:", example["old_label"])
print("Majority label:", example["majority_label"])
print("Human distribution:", example["label_dist"])
print("Entropy:", example["entropy"])
