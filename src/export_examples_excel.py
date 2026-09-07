"""Write qualitative examples from comparison.json to an Excel file."""

import json
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
INPUT_FILE = os.path.join(RESULTS_DIR, "comparison.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "qualitative_examples.xlsx")

COLUMNS = [
    "uid",
    "premise",
    "hypothesis",
    "human_distribution",
    "human_entropy",
    "entropy_bin",
    "old_label",
    "human_majority_label",
    "deberta_prediction",
    "modernbert_prediction",
    "deberta_distribution",
    "modernbert_distribution",
    "deberta_confidence",
    "modernbert_confidence",
    "deberta_jsd",
    "modernbert_jsd",
]


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for dataset_name, dataset_results in results.items():
            for category, examples in dataset_results["qualitative_examples"].items():
                rows = [{col: example[col] for col in COLUMNS} for example in examples]
                if not rows:
                    continue
                sheet_name = (dataset_name + "_" + category)[:31]
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

    print("Saved examples to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
