"""
Run DeBERTa-v3 or ModernBERT on ChaosNLI-SNLI or ChaosNLI-MNLI-m.

Usage:
    python src/evaluate_models.py deberta snli
    python src/evaluate_models.py modernbert mnli_m 64
"""

import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


MODELS = {
    "modernbert": "tasksource/ModernBERT-large-nli",
    "deberta": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}

DATASETS = {
    "snli": "chaosNLI_snli.jsonl",
    "mnli_m": "chaosNLI_mnli_m.jsonl",
}

LABELS = ["entailment", "neutral", "contradiction"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "ChaosNLI", "data", "chaosNLI_v1.0")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class ChaosNLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]["example"]
        return {
            "index": index,
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
        }


def normalize_label(label):
    return str(label).strip().lower()


def find_label_id(id2label, target):
    # Different checkpoints use slightly different names for the 3 NLI classes.
    aliases = {
        "entailment": {"entailment", "entails", "e", "label_0"},
        "neutral": {"neutral", "n", "label_1"},
        "contradiction": {"contradiction", "contradictory", "c", "label_2"},
    }
    wanted = aliases[target.lower()]
    for idx, name in id2label.items():
        if normalize_label(name) in wanted:
            return int(idx)
    return None


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Usage: python src/evaluate_models.py MODEL DATASET [BATCH_SIZE]\n"
            "  MODEL:    modernbert | deberta\n"
            "  DATASET:  snli | mnli_m"
        )
        sys.exit(1)

    model_key = sys.argv[1].lower()
    dataset_key = sys.argv[2].lower()
    batch_size = int(sys.argv[3]) if len(sys.argv) == 4 else 32

    if model_key not in MODELS:
        print("Unknown model:", model_key)
        sys.exit(1)
    if dataset_key not in DATASETS:
        print("Unknown dataset:", dataset_key)
        sys.exit(1)

    model_name = MODELS[model_key]
    data_file = os.path.join(DATA_DIR, DATASETS[dataset_key])
    output_file = os.path.join(RESULTS_DIR, f"{model_key}_{dataset_key}_predictions.json")

    print("Model:", model_name)
    print("Dataset:", dataset_key)
    print("Batch size:", batch_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    id2label = {int(i): name for i, name in model.config.id2label.items()}
    print("Label mapping:", id2label)

    e_id = find_label_id(id2label, "entailment")
    n_id = find_label_id(id2label, "neutral")
    c_id = find_label_id(id2label, "contradiction")
    if None in (e_id, n_id, c_id):
        raise ValueError("Could not map all three NLI labels: " + str(id2label))

    examples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print("Examples:", len(examples))

    loader = DataLoader(ChaosNLIDataset(examples), batch_size=batch_size, shuffle=False)
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=model_key + " on " + dataset_key):
            inputs = tokenizer(
                batch["premise"],
                batch["hypothesis"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            probs = torch.softmax(model(**inputs).logits, dim=-1)

            # Store probs in a fixed order: E, N, C
            std_probs = torch.stack(
                [probs[:, e_id], probs[:, n_id], probs[:, c_id]],
                dim=1,
            )
            pred_ids = torch.argmax(std_probs, dim=1)
            conf = torch.max(std_probs, dim=1).values

            std_probs = std_probs.cpu().tolist()
            pred_ids = pred_ids.cpu().tolist()
            conf = conf.cpu().tolist()

            for i, example_idx in enumerate(batch["index"].tolist()):
                item = examples[example_idx]
                example = item["example"]
                results.append(
                    {
                        "uid": item["uid"],
                        "dataset": dataset_key,
                        "premise": example["premise"],
                        "hypothesis": example["hypothesis"],
                        "old_label": item["old_label"],
                        "old_labels": item.get("old_labels", []),
                        "human_majority_label": item["majority_label"],
                        "human_distribution": item["label_dist"],
                        "human_entropy": item["entropy"],
                        "model_distribution": std_probs[i],
                        "predicted_label": LABELS[pred_ids[i]],
                        "model_confidence": conf[i],
                    }
                )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved", len(results), "predictions to", output_file)


if __name__ == "__main__":
    main()
