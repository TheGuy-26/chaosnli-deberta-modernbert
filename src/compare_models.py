"""
Compare DeBERTa-v3 and ModernBERT predictions with ChaosNLI labels.

This is our earlier comparison script (agreement, entropy bins, plots).
The numbers in the report come from analyze_findings.py, which uses the
official ChaosNLI JSD/KL implementation.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

FILES = {
    "snli": {
        "modernbert": "modernbert_snli_predictions.json",
        "deberta": "deberta_snli_predictions.json",
    },
    "mnli_m": {
        "modernbert": "modernbert_mnli_m_predictions.json",
        "deberta": "deberta_mnli_m_predictions.json",
    },
}

LABELS = ["entailment", "neutral", "contradiction"]

OLD_LABEL_MAPPING = {
    "e": "entailment",
    "n": "neutral",
    "c": "contradiction",
}


def load_predictions(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def jsd(p, q):
    # JS divergence, base 2. Official ChaosNLI scores use SciPy JS distance
    # instead; see analyze_findings.py.
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    m = 0.5 * (p + q)

    def kld(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float(0.5 * kld(p, m) + 0.5 * kld(q, m))


def normalized_label(label):
    label = str(label).strip().lower()
    if label in {"entailment", "entails", "e", "label_0"}:
        return "entailment"
    if label in {"neutral", "n", "label_1"}:
        return "neutral"
    if label in {"contradiction", "contradictory", "c", "label_2"}:
        return "contradiction"
    return label


def model_confidence(model_dist):
    return float(np.max(model_dist))


def entropy_bins(data):
    """Split examples into low / medium / high human entropy."""
    entropies = np.array([item["human_entropy"] for item in data], dtype=float)
    lower = float(np.quantile(entropies, 1 / 3))
    upper = float(np.quantile(entropies, 2 / 3))

    bins = []
    for value in entropies:
        if value <= lower:
            bins.append("low")
        elif value <= upper:
            bins.append("medium")
        else:
            bins.append("high")
    return bins, lower, upper


def analyze_model(data):
    jsd_value = []
    majority_agreement = []
    old_label_agreement = []
    model_confidences = []
    human_entropies = []
    examples = []

    bins, lower_threshold, upper_threshold = entropy_bins(data)

    for item, entropy_bin in zip(data, bins):
        human_distribution = item["human_distribution"]
        model_distribution = item["model_distribution"]

        human_majority_label = LABELS[int(np.argmax(human_distribution))]
        predicted_label = normalized_label(item["predicted_label"])
        majority_matches = predicted_label == human_majority_label
        majority_agreement.append(majority_matches)

        old_label_raw = item.get("old_label")
        old_label = OLD_LABEL_MAPPING.get(old_label_raw, normalized_label(old_label_raw))
        old_matches = predicted_label == old_label
        old_label_agreement.append(old_matches)

        js = jsd(human_distribution, model_distribution)
        jsd_value.append(js)

        confidence = model_confidence(model_distribution)
        human_entropy = float(item["human_entropy"])
        model_confidences.append(confidence)
        human_entropies.append(human_entropy)

        examples.append(
            {
                "uid": item["uid"],
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "human_distribution": human_distribution,
                "human_entropy": human_entropy,
                "entropy_bin": entropy_bin,
                "old_label": old_label,
                "human_majority_label": human_majority_label,
                "model_distribution": model_distribution,
                "predicted_label": predicted_label,
                "model_confidence": confidence,
                "jsd": js,
                "majority_agreement": majority_matches,
                "old_label_agreement": old_matches,
            }
        )

    confidence_correlation, confidence_p = spearmanr(model_confidences, human_entropies)
    jsd_correlation, jsd_p = spearmanr(jsd_value, human_entropies)

    entropy_bin_results = {}
    for bin_name in ["low", "medium", "high"]:
        bin_examples = [ex for ex in examples if ex["entropy_bin"] == bin_name]
        if not bin_examples:
            continue
        entropy_bin_results[bin_name] = {
            "number_of_examples": len(bin_examples),
            "mean_human_entropy": float(np.mean([x["human_entropy"] for x in bin_examples])),
            "mean_jsd": float(np.mean([x["jsd"] for x in bin_examples])),
            "majority_label_agreement": float(np.mean([x["majority_agreement"] for x in bin_examples])),
            "old_label_agreement": float(np.mean([x["old_label_agreement"] for x in bin_examples])),
            "mean_model_confidence": float(np.mean([x["model_confidence"] for x in bin_examples])),
        }

    return {
        "number_of_examples": len(data),
        "entropy_lower_threshold": lower_threshold,
        "entropy_upper_threshold": upper_threshold,
        "old_label_agreement": float(np.mean(old_label_agreement)),
        "majority_label_agreement": float(np.mean(majority_agreement)),
        "mean_jsd": float(np.mean(jsd_value)),
        "mean_model_confidence": float(np.mean(model_confidences)),
        "mean_human_entropy": float(np.mean(human_entropies)),
        "confidence_human_entropy_spearman": float(confidence_correlation),
        "confidence_human_entropy_p_value": float(confidence_p),
        "jsd_human_entropy_spearman": float(jsd_correlation),
        "jsd_human_entropy_p_value": float(jsd_p),
        "entropy_bins": entropy_bin_results,
        "examples": examples,
    }


def compare_models(deberta_data, modernbert_data):
    if len(deberta_data) != len(modernbert_data):
        raise ValueError("DeBERTa and ModernBERT contain different numbers of examples.")

    comparison = []
    for deberta_item, modernbert_item in zip(deberta_data, modernbert_data):
        if deberta_item["uid"] != modernbert_item["uid"]:
            raise ValueError("UID mismatch.")

        deberta_prediction = normalized_label(deberta_item["predicted_label"])
        modernbert_prediction = normalized_label(modernbert_item["predicted_label"])
        human_distribution = deberta_item["human_distribution"]
        human_majority_label = LABELS[int(np.argmax(human_distribution))]
        old_label = OLD_LABEL_MAPPING.get(
            deberta_item.get("old_label"),
            normalized_label(deberta_item.get("old_label")),
        )

        models_agree = deberta_prediction == modernbert_prediction
        deberta_human_agree = deberta_prediction == human_majority_label
        modernbert_human_agree = modernbert_prediction == human_majority_label

        comparison.append(
            {
                "uid": deberta_item["uid"],
                "premise": deberta_item["premise"],
                "hypothesis": deberta_item["hypothesis"],
                "human_distribution": human_distribution,
                "human_entropy": deberta_item["human_entropy"],
                "entropy_bin": None,
                "old_label": old_label,
                "human_majority_label": human_majority_label,
                "deberta_prediction": deberta_prediction,
                "modernbert_prediction": modernbert_prediction,
                "deberta_distribution": deberta_item["model_distribution"],
                "modernbert_distribution": modernbert_item["model_distribution"],
                "deberta_confidence": model_confidence(deberta_item["model_distribution"]),
                "modernbert_confidence": model_confidence(modernbert_item["model_distribution"]),
                "deberta_jsd": jsd(human_distribution, deberta_item["model_distribution"]),
                "modernbert_jsd": jsd(human_distribution, modernbert_item["model_distribution"]),
                "models_agree": models_agree,
                "models_disagree": not models_agree,
                "deberta_human_agree": deberta_human_agree,
                "modernbert_human_agree": modernbert_human_agree,
                "one_model_human_agree": deberta_human_agree != modernbert_human_agree,
                "both_agree_original_diff": models_agree and deberta_prediction != old_label,
                "both_agree_human_diff": models_agree and deberta_prediction != human_majority_label,
                "original_human_disagreement": old_label != human_majority_label,
            }
        )
    return comparison


def assign_entropy_bins(comparison):
    entropies = np.array([item["human_entropy"] for item in comparison], dtype=float)
    lower_threshold = float(np.quantile(entropies, 1 / 3))
    upper_threshold = float(np.quantile(entropies, 2 / 3))

    for item in comparison:
        value = item["human_entropy"]
        if value <= lower_threshold:
            item["entropy_bin"] = "low"
        elif value <= upper_threshold:
            item["entropy_bin"] = "medium"
        else:
            item["entropy_bin"] = "high"
    return lower_threshold, upper_threshold


def summarize_model_comparison(comparison):
    total = len(comparison)
    return {
        "number_of_examples": total,
        "model_agreement": float(np.mean([x["models_agree"] for x in comparison])),
        "model_disagreement": float(np.mean([x["models_disagree"] for x in comparison])),
        "both_models_agree_but_differ_from_original": float(
            np.mean([x["both_agree_original_diff"] for x in comparison])
        ),
        "both_models_agree_but_differ_from_human_majority": float(
            np.mean([x["both_agree_human_diff"] for x in comparison])
        ),
        "only_one_model_agrees_with_human_majority": float(
            np.mean([x["one_model_human_agree"] for x in comparison])
        ),
        "original_label_differs_from_human_majority": float(
            np.mean([x["original_human_disagreement"] for x in comparison])
        ),
        "deberta_only_human_agreement": float(
            np.mean([x["deberta_human_agree"] and not x["modernbert_human_agree"] for x in comparison])
        ),
        "modernbert_only_human_agreement": float(
            np.mean([x["modernbert_human_agree"] and not x["deberta_human_agree"] for x in comparison])
        ),
    }


def summarize_entropy_bin(comparison):
    results = {}
    for bin_name in ["low", "medium", "high"]:
        bin_examples = [item for item in comparison if item["entropy_bin"] == bin_name]
        if not bin_examples:
            continue
        results[bin_name] = {
            "number_of_examples": len(bin_examples),
            "model_agreement": float(np.mean([x["models_agree"] for x in bin_examples])),
            "model_disagreement": float(np.mean([x["models_disagree"] for x in bin_examples])),
            "both_models_agree_but_differ_from_original": float(
                np.mean([x["both_agree_original_diff"] for x in bin_examples])
            ),
            "both_models_agree_but_differ_from_human_majority": float(
                np.mean([x["both_agree_human_diff"] for x in bin_examples])
            ),
            "only_one_model_agrees_with_human_majority": float(
                np.mean([x["one_model_human_agree"] for x in bin_examples])
            ),
            "deberta_only_human_agreement": float(
                np.mean([x["deberta_human_agree"] and not x["modernbert_human_agree"] for x in bin_examples])
            ),
            "modernbert_only_human_agreement": float(
                np.mean([x["modernbert_human_agree"] and not x["deberta_human_agree"] for x in bin_examples])
            ),
            "deberta_human_agreement": float(np.mean([x["deberta_human_agree"] for x in bin_examples])),
            "modernbert_human_agreement": float(np.mean([x["modernbert_human_agree"] for x in bin_examples])),
            "original_label_differs_from_human_majority": float(
                np.mean([x["original_human_disagreement"] for x in bin_examples])
            ),
            "mean_human_entropy": float(np.mean([x["human_entropy"] for x in bin_examples])),
            "mean_deberta_confidence": float(np.mean([x["deberta_confidence"] for x in bin_examples])),
            "mean_modernbert_confidence": float(np.mean([x["modernbert_confidence"] for x in bin_examples])),
            "mean_deberta_jsd": float(np.mean([x["deberta_jsd"] for x in bin_examples])),
            "mean_modernbert_jsd": float(np.mean([x["modernbert_jsd"] for x in bin_examples])),
        }
    return results


def select_qualitative_examples(comparison):
    return {
        "model_disagreement": [x for x in comparison if x["models_disagree"]],
        "both_agree_original_diff": [x for x in comparison if x["both_agree_original_diff"]],
        "both_agree_human_diff": [x for x in comparison if x["both_agree_human_diff"]],
        "deberta_only_human": [
            x for x in comparison if x["deberta_human_agree"] and not x["modernbert_human_agree"]
        ],
        "modernbert_only_human": [
            x for x in comparison if x["modernbert_human_agree"] and not x["deberta_human_agree"]
        ],
        "original_human_disagreement": [x for x in comparison if x["original_human_disagreement"]],
    }


def sort_qualitative_examples(examples):
    sorted_examples = {}
    for category, items in examples.items():
        sorted_examples[category] = sorted(items, key=lambda x: x["human_entropy"], reverse=True)
    return sorted_examples


def plot_model_agreement_by_entropy(entropy_results, dataset_name):
    bins = ["low", "medium", "high"]
    deberta_values = [entropy_results[b]["deberta_human_agreement"] for b in bins]
    modernbert_values = [entropy_results[b]["modernbert_human_agreement"] for b in bins]

    x = np.arange(len(bins))
    width = 0.3
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, deberta_values, width, label="DeBERTa-v3")
    plt.bar(x + width / 2, modernbert_values, width, label="ModernBERT")
    plt.xticks(x, ["Low disagreement", "Medium disagreement", "High disagreement"])
    plt.ylabel("Agreement with human majority")
    plt.xlabel("Human disagreement")
    plt.title("Model agreement by human disagreement - " + dataset_name)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, dataset_name + "_human_agreement_by_entropy.png"), dpi=300)
    plt.close()


def plot_jsd_by_entropy(entropy_results, dataset_name):
    bins = ["low", "medium", "high"]
    deberta_values = [entropy_results[x]["mean_deberta_jsd"] for x in bins]
    modernbert_values = [entropy_results[x]["mean_modernbert_jsd"] for x in bins]

    x = np.arange(len(bins))
    width = 0.3
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, deberta_values, width, label="DeBERTa-v3")
    plt.bar(x + width / 2, modernbert_values, width, label="ModernBERT")
    plt.xticks(x, ["Low disagreement", "Medium disagreement", "High disagreement"])
    plt.ylabel("Mean Jensen-Shannon divergence")
    plt.xlabel("Human disagreement")
    plt.title("Model-human distribution divergence: " + dataset_name)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, dataset_name + "_jsd_by_entropy.png"), dpi=300)
    plt.close()


def plot_confidence_vs_entropy(deberta_results, modernbert_results, dataset_name):
    deberta_examples = deberta_results["examples"]
    modernbert_examples = modernbert_results["examples"]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        [x["human_entropy"] for x in deberta_examples],
        [x["model_confidence"] for x in deberta_examples],
        alpha=0.4,
        label="DeBERTa-v3",
    )
    plt.scatter(
        [x["human_entropy"] for x in modernbert_examples],
        [x["model_confidence"] for x in modernbert_examples],
        alpha=0.4,
        label="ModernBERT",
    )
    plt.ylabel("Model confidence")
    plt.xlabel("Human annotation entropy")
    plt.title("Model confidence vs human uncertainty - " + dataset_name)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, dataset_name + "_confidence_vs_entropy.png"), dpi=300)
    plt.close()


def main():
    all_results = {}

    for dataset_name, model_files in FILES.items():
        print("\nDataset:", dataset_name)

        deberta_data = load_predictions(model_files["deberta"])
        modernbert_data = load_predictions(model_files["modernbert"])
        print("DeBERTa examples:", len(deberta_data))
        print("ModernBERT examples:", len(modernbert_data))

        deberta_results = analyze_model(deberta_data)
        modernbert_results = analyze_model(modernbert_data)
        comparison = compare_models(deberta_data, modernbert_data)
        lower_threshold, upper_threshold = assign_entropy_bins(comparison)
        model_comparison = summarize_model_comparison(comparison)
        entropy_comparison = summarize_entropy_bin(comparison)
        qualitative = sort_qualitative_examples(select_qualitative_examples(comparison))

        plot_model_agreement_by_entropy(entropy_comparison, dataset_name)
        plot_jsd_by_entropy(entropy_comparison, dataset_name)
        plot_confidence_vs_entropy(deberta_results, modernbert_results, dataset_name)

        all_results[dataset_name] = {
            "deberta": deberta_results,
            "modernbert": modernbert_results,
            "model_comparison": model_comparison,
            "entropy_comparison": entropy_comparison,
            "entropy_lower_threshold": lower_threshold,
            "entropy_upper_threshold": upper_threshold,
            "qualitative_examples": qualitative,
            "comparison": comparison,
        }

        print("Model agreement:", round(model_comparison["model_agreement"], 4))
        print("Model disagreement:", round(model_comparison["model_disagreement"], 4))
        print("DeBERTa only agrees with human:", round(model_comparison["deberta_only_human_agreement"], 4))
        print("ModernBERT only agrees with human:", round(model_comparison["modernbert_only_human_agreement"], 4))

    output_file = os.path.join(RESULTS_DIR, "comparison.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("Saved", output_file)
    print("Plots in", PLOTS_DIR)


if __name__ == "__main__":
    main()
