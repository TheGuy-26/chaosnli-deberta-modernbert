"""
Official ChaosNLI metrics for our two models vs the 2020 encoder baselines.

JSD is SciPy Jensen-Shannon distance (same as the ChaosNLI code).
KL is KL(human || model) in nats.
Old accuracy = original SNLI/MNLI label.
New accuracy = 100-annotator majority.
"""

import json
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", ".mplconfig"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy
from scipy.stats import spearmanr


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "ChaosNLI", "data", "chaosNLI_v1.0")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BASELINE_FILE = os.path.join(
    PROJECT_ROOT,
    "ChaosNLI",
    "data",
    "model_predictions",
    "model_predictions_for_snli_mnli.json",
)

LABELS = ["entailment", "neutral", "contradiction"]
SHORT_TO_FULL = {"e": "entailment", "n": "neutral", "c": "contradiction"}

BASELINE_MODELS = [
    "bert-large",
    "roberta-large",
    "xlnet-large",
    "bart-large",
    "albert-xxlarge",
]

DATASETS = {
    "snli": "chaosNLI_snli.jsonl",
    "mnli_m": "chaosNLI_mnli_m.jsonl",
}


def normalize_label(label):
    label = str(label).strip().lower()
    if label in SHORT_TO_FULL:
        return SHORT_TO_FULL[label]
    if label in SHORT_TO_FULL.values():
        return label
    aliases = {
        "entails": "entailment",
        "contradictory": "contradiction",
        "label_0": "entailment",
        "label_1": "neutral",
        "label_2": "contradiction",
    }
    return aliases.get(label, label)


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def softmax(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=float) / temperature
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / exp.sum()


def normalize_dist(dist):
    dist = np.asarray(dist, dtype=float)
    dist = np.clip(dist, 1e-15, None)
    return dist / dist.sum()


def jsd_distance(human, model):
    value = jensenshannon(normalize_dist(human), normalize_dist(model))
    if np.isnan(value):
        return 0.0
    return float(value)


def kl_divergence(human, model):
    return float(scipy_entropy(normalize_dist(human), normalize_dist(model)))


def dist_entropy(dist):
    # bits, same unit as ChaosNLI human entropy
    return float(scipy_entropy(normalize_dist(dist), base=2))


def predicted_from_dist(dist):
    return LABELS[int(np.argmax(dist))]


def load_gold(dataset_key):
    gold = {}
    for item in load_jsonl(os.path.join(DATA_DIR, DATASETS[dataset_key])):
        gold[item["uid"]] = {
            "uid": item["uid"],
            "premise": item["example"]["premise"],
            "hypothesis": item["example"]["hypothesis"],
            "human_distribution": item["label_dist"],
            "human_entropy": item["entropy"],
            "old_label": normalize_label(item["old_label"]),
            "human_majority_label": normalize_label(item["majority_label"]),
        }
    return gold


def load_our_predictions(model_key, dataset_key):
    path = os.path.join(RESULTS_DIR, f"{model_key}_{dataset_key}_predictions.json")
    preds = {}
    for item in load_json(path):
        preds[item["uid"]] = {
            "distribution": normalize_dist(item["model_distribution"]).tolist(),
            "predicted_label": normalize_label(item["predicted_label"]),
            "confidence": float(item["model_confidence"]),
        }
    return preds


def evaluate(gold, preds):
    rows = []
    for uid, item in gold.items():
        if uid not in preds:
            continue
        pred = preds[uid]
        human = item["human_distribution"]
        model = pred["distribution"]
        predicted = pred["predicted_label"]
        rows.append(
            {
                "uid": uid,
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "human_distribution": human,
                "model_distribution": model,
                "human_entropy": item["human_entropy"],
                "model_entropy": dist_entropy(model),
                "old_label": item["old_label"],
                "human_majority_label": item["human_majority_label"],
                "predicted_label": predicted,
                "confidence": pred["confidence"],
                "jsd": jsd_distance(human, model),
                "kl": kl_divergence(human, model),
                "old_acc": predicted == item["old_label"],
                "new_acc": predicted == item["human_majority_label"],
            }
        )
    if not rows:
        raise ValueError("No overlapping examples between gold and predictions.")
    return rows


def summarize(rows):
    conf = np.array([r["confidence"] for r in rows])
    hent = np.array([r["human_entropy"] for r in rows])
    ment = np.array([r["model_entropy"] for r in rows])
    jsd = np.array([r["jsd"] for r in rows])
    conf_corr, conf_p = spearmanr(conf, hent)
    jsd_corr, jsd_p = spearmanr(jsd, hent)
    ent_corr, ent_p = spearmanr(ment, hent)
    return {
        "n": len(rows),
        "jsd": float(np.mean(jsd)),
        "kl": float(np.mean([r["kl"] for r in rows])),
        "old_acc": float(np.mean([r["old_acc"] for r in rows])),
        "new_acc": float(np.mean([r["new_acc"] for r in rows])),
        "mean_confidence": float(np.mean(conf)),
        "mean_human_entropy": float(np.mean(hent)),
        "mean_model_entropy": float(np.mean(ment)),
        "entropy_gap": float(np.mean(hent) - np.mean(ment)),
        "confidence_vs_human_entropy_spearman": float(conf_corr),
        "confidence_vs_human_entropy_p": float(conf_p),
        "jsd_vs_human_entropy_spearman": float(jsd_corr),
        "jsd_vs_human_entropy_p": float(jsd_p),
        "model_vs_human_entropy_spearman": float(ent_corr),
        "model_vs_human_entropy_p": float(ent_p),
    }


def entropy_bins(rows, n_bins=3):
    entropies = np.array([r["human_entropy"] for r in rows])
    edges = np.quantile(entropies, np.linspace(0, 1, n_bins + 1))
    edges[0] = min(entropies.min(), edges[0])
    edges[-1] = max(entropies.max(), edges[-1])
    names = ["low", "medium", "high"]
    binned = {name: [] for name in names}

    for row in rows:
        value = row["human_entropy"]
        placed = False
        for i, name in enumerate(names):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                in_bin = lo <= value <= hi
            else:
                in_bin = lo <= value < hi
            if in_bin:
                binned[name].append(row)
                placed = True
                break
        if not placed:
            binned[names[-1]].append(row)

    results = {}
    for i, name in enumerate(names):
        subset = binned[name]
        results[name] = {
            "n": len(subset),
            "entropy_range": [float(edges[i]), float(edges[i + 1])],
            **summarize(subset),
        }
    return results, edges.tolist()


def apply_temperature(rows, temperature):
    scaled = []
    for row in rows:
        # we stored softmax probs, not logits
        logits = np.log(normalize_dist(row["model_distribution"]))
        dist = softmax(logits, temperature=temperature).tolist()
        predicted = predicted_from_dist(dist)
        new_row = dict(row)
        new_row["model_distribution"] = dist
        new_row["predicted_label"] = predicted
        new_row["confidence"] = float(max(dist))
        new_row["model_entropy"] = dist_entropy(dist)
        new_row["jsd"] = jsd_distance(row["human_distribution"], dist)
        new_row["kl"] = kl_divergence(row["human_distribution"], dist)
        new_row["old_acc"] = predicted == row["old_label"]
        new_row["new_acc"] = predicted == row["human_majority_label"]
        scaled.append(new_row)
    return scaled


def temperature_for_entropy_match(rows, target_entropy, lo=0.2, hi=8.0, steps=40):
    best_t = 1.0
    best_gap = abs(summarize(rows)["mean_model_entropy"] - target_entropy)
    for t in np.linspace(lo, hi, steps):
        gap = abs(summarize(apply_temperature(rows, float(t)))["mean_model_entropy"] - target_entropy)
        if gap < best_gap:
            best_gap = gap
            best_t = float(t)
    return best_t


def split_temperature_search(rows, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    split = len(rows) // 2
    cal = [rows[i] for i in indices[:split]]
    test = [rows[i] for i in indices[split:]]

    best_t = 1.0
    best_jsd = summarize(cal)["jsd"]
    for t in np.linspace(0.5, 5.0, 19):
        jsd = summarize(apply_temperature(cal, float(t)))["jsd"]
        if jsd < best_jsd:
            best_jsd = jsd
            best_t = float(t)

    return {
        "temperature": best_t,
        "calibration_jsd": best_jsd,
        "held_out_before": summarize(test),
        "held_out_after": summarize(apply_temperature(test, best_t)),
    }


def pairwise_comparison(gold, left_preds, right_preds):
    rows = []
    for uid, item in gold.items():
        if uid not in left_preds or uid not in right_preds:
            continue
        left = left_preds[uid]
        right = right_preds[uid]
        agree = left["predicted_label"] == right["predicted_label"]
        left_ok = left["predicted_label"] == item["human_majority_label"]
        right_ok = right["predicted_label"] == item["human_majority_label"]
        rows.append(
            {
                "uid": uid,
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "human_distribution": item["human_distribution"],
                "human_entropy": item["human_entropy"],
                "old_label": item["old_label"],
                "human_majority_label": item["human_majority_label"],
                "deberta_label": left["predicted_label"],
                "modernbert_label": right["predicted_label"],
                "deberta_dist": left["distribution"],
                "modernbert_dist": right["distribution"],
                "deberta_conf": left["confidence"],
                "modernbert_conf": right["confidence"],
                "models_agree": agree,
                "both_confident": left["confidence"] >= 0.9 and right["confidence"] >= 0.9,
                "both_wrong": (not left_ok) and (not right_ok),
                "both_agree_and_wrong": agree and (not left_ok),
                "left_only_human": left_ok and not right_ok,
                "right_only_human": right_ok and not left_ok,
            }
        )
    return rows


def summarize_pairwise(rows):
    n = len(rows)
    return {
        "n": n,
        "model_agreement": float(np.mean([r["models_agree"] for r in rows])),
        "both_confident": float(np.mean([r["both_confident"] for r in rows])),
        "both_agree_and_wrong": float(np.mean([r["both_agree_and_wrong"] for r in rows])),
        "both_confident_and_wrong": float(
            np.mean([r["both_confident"] and r["both_wrong"] for r in rows])
        ),
        "deberta_only_human": float(np.mean([r["left_only_human"] for r in rows])),
        "modernbert_only_human": float(np.mean([r["right_only_human"] for r in rows])),
    }


def pairwise_by_entropy(rows):
    entropies = np.array([r["human_entropy"] for r in rows])
    edges = np.quantile(entropies, [0, 1 / 3, 2 / 3, 1])
    names = ["low", "medium", "high"]
    out = {}
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        if i == 2:
            subset = [r for r in rows if lo <= r["human_entropy"] <= hi]
        else:
            subset = [r for r in rows if lo <= r["human_entropy"] < hi]
        out[name] = {
            "n": len(subset),
            "entropy_range": [float(lo), float(hi)],
            **summarize_pairwise(subset),
            "mean_human_entropy": float(np.mean([r["human_entropy"] for r in subset])),
            "mean_deberta_conf": float(np.mean([r["deberta_conf"] for r in subset])),
            "mean_modernbert_conf": float(np.mean([r["modernbert_conf"] for r in subset])),
        }
    return out


def compact_example(row, extra=None):
    item = {
        "uid": row["uid"],
        "premise": row["premise"],
        "hypothesis": row["hypothesis"],
        "human_distribution": {
            "entailment": round(float(row["human_distribution"][0]), 3),
            "neutral": round(float(row["human_distribution"][1]), 3),
            "contradiction": round(float(row["human_distribution"][2]), 3),
        },
        "human_entropy": round(float(row["human_entropy"]), 3),
        "old_label": row["old_label"],
        "human_majority_label": row["human_majority_label"],
    }
    if extra:
        item.update(extra)
    return item


def qualitative_examples(pairwise_rows, deberta_rows, modernbert_rows):
    deberta_by_uid = {r["uid"]: r for r in deberta_rows}
    modernbert_by_uid = {r["uid"]: r for r in modernbert_rows}

    confident_wrong = [
        r
        for r in pairwise_rows
        if r["both_confident"] and r["models_agree"] and r["both_wrong"] and r["human_entropy"] >= 1.0
    ]
    confident_wrong.sort(key=lambda r: r["human_entropy"], reverse=True)

    model_split = [r for r in pairwise_rows if (not r["models_agree"]) and r["human_entropy"] >= 1.0]
    model_split.sort(key=lambda r: abs(r["deberta_conf"] - r["modernbert_conf"]), reverse=True)

    follow_new = [
        r
        for r in pairwise_rows
        if r["old_label"] != r["human_majority_label"]
        and r["models_agree"]
        and r["deberta_label"] == r["human_majority_label"]
        and r["human_entropy"] >= 0.8
    ]
    follow_new.sort(key=lambda r: r["human_entropy"], reverse=True)

    follow_old = [
        r
        for r in pairwise_rows
        if r["old_label"] != r["human_majority_label"]
        and r["models_agree"]
        and r["deberta_label"] == r["old_label"]
        and r["human_entropy"] >= 0.8
    ]
    follow_old.sort(key=lambda r: r["human_entropy"], reverse=True)

    def decorate(row):
        d = deberta_by_uid[row["uid"]]
        m = modernbert_by_uid[row["uid"]]
        return compact_example(
            row,
            extra={
                "deberta": {
                    "label": row["deberta_label"],
                    "confidence": round(row["deberta_conf"], 3),
                    "distribution": [round(x, 3) for x in row["deberta_dist"]],
                    "jsd": round(d["jsd"], 3),
                },
                "modernbert": {
                    "label": row["modernbert_label"],
                    "confidence": round(row["modernbert_conf"], 3),
                    "distribution": [round(x, 3) for x in row["modernbert_dist"]],
                    "jsd": round(m["jsd"], 3),
                },
            },
        )

    return {
        "both_confident_wrong_high_entropy": [decorate(r) for r in confident_wrong[:8]],
        "models_disagree_high_entropy": [decorate(r) for r in model_split[:8]],
        "models_follow_new_human_majority": [decorate(r) for r in follow_new[:6]],
        "models_follow_old_label_against_humans": [decorate(r) for r in follow_old[:6]],
    }


def plot_metric_by_entropy(dataset, metric, ylabel, filename, deberta_bins, modernbert_bins, baselines):
    categories = ["Low disagreement", "Medium disagreement", "High disagreement"]
    x = np.arange(len(categories))
    width = 0.16
    series = [
        ("DeBERTa-v3", [deberta_bins[b][metric] for b in ["low", "medium", "high"]]),
        ("ModernBERT", [modernbert_bins[b][metric] for b in ["low", "medium", "high"]]),
    ]
    if "roberta-large" in baselines:
        series.append(
            (
                "RoBERTa-large (2020)",
                [baselines["roberta-large"]["entropy_bins"][b][metric] for b in ["low", "medium", "high"]],
            )
        )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for i, (name, values) in enumerate(series):
        ax.bar(x + (i - 1) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Human annotation entropy tertile")
    ax.set_title(ylabel + " by human disagreement (" + dataset.upper() + ")")
    if metric in {"jsd", "mean_confidence", "old_acc", "new_acc"}:
        ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close(fig)


def plot_agreement_vs_entropy(dataset, pairwise_bins):
    categories = ["Low disagreement", "Medium disagreement", "High disagreement"]
    agreement = [pairwise_bins[b]["model_agreement"] for b in ["low", "medium", "high"]]
    both_wrong = [pairwise_bins[b]["both_agree_and_wrong"] for b in ["low", "medium", "high"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(categories))
    ax.plot(x, agreement, marker="o", label="DeBERTa-v3 agrees with ModernBERT")
    ax.plot(x, both_wrong, marker="o", label="Both agree, both differ from human majority")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of examples")
    ax.set_xlabel("Human annotation entropy tertile")
    ax.set_title("Model agreement vs human disagreement (" + dataset.upper() + ")")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, dataset + "_model_consensus_by_entropy.png"), dpi=300)
    plt.close(fig)


def round_floats(obj, ndigits=4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def print_row(name, summary):
    print(
        f"{name:18s}  JSD={summary['jsd']:.4f}  KL={summary['kl']:.4f}  "
        f"old={summary['old_acc']:.4f}  new={summary['new_acc']:.4f}"
    )


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    all_findings = {}
    baseline_cache = None

    for dataset_key in DATASETS:
        print("\n" + dataset_key)
        gold = load_gold(dataset_key)
        deberta_preds = load_our_predictions("deberta", dataset_key)
        modernbert_preds = load_our_predictions("modernbert", dataset_key)
        deberta_rows = evaluate(gold, deberta_preds)
        modernbert_rows = evaluate(gold, modernbert_preds)

        if baseline_cache is None:
            print("Loading 2020 baseline predictions...")
            baseline_cache = load_json(BASELINE_FILE)

        baseline_results = {}
        for model_name in BASELINE_MODELS:
            preds = {}
            for uid in gold:
                if uid not in baseline_cache[model_name]:
                    continue
                item = baseline_cache[model_name][uid]
                dist = softmax(item["logits"]).tolist()
                preds[uid] = {
                    "distribution": dist,
                    "predicted_label": normalize_label(item["predicted_label"]),
                    "confidence": float(max(dist)),
                }
            rows = evaluate(gold, preds)
            bins, edges = entropy_bins(rows)
            baseline_results[model_name] = {
                "overall": summarize(rows),
                "entropy_bins": bins,
                "entropy_edges": edges,
            }
            print_row(model_name, baseline_results[model_name]["overall"])

        deberta_summary = summarize(deberta_rows)
        modernbert_summary = summarize(modernbert_rows)
        deberta_bins, deberta_edges = entropy_bins(deberta_rows)
        modernbert_bins, modernbert_edges = entropy_bins(modernbert_rows)
        print_row("deberta-v3", deberta_summary)
        print_row("modernbert", modernbert_summary)

        pairwise_rows = pairwise_comparison(gold, deberta_preds, modernbert_preds)
        pairwise_overall = summarize_pairwise(pairwise_rows)
        pairwise_bins = pairwise_by_entropy(pairwise_rows)

        deberta_entropy_t = temperature_for_entropy_match(
            deberta_rows, deberta_summary["mean_human_entropy"]
        )
        modernbert_entropy_t = temperature_for_entropy_match(
            modernbert_rows, modernbert_summary["mean_human_entropy"]
        )
        deberta_split_t = split_temperature_search(deberta_rows)
        modernbert_split_t = split_temperature_search(modernbert_rows)
        qualitative = qualitative_examples(pairwise_rows, deberta_rows, modernbert_rows)

        plot_metric_by_entropy(
            dataset_key, "new_acc", "Agreement with human majority",
            dataset_key + "_new_acc_by_entropy.png",
            deberta_bins, modernbert_bins, baseline_results,
        )
        plot_metric_by_entropy(
            dataset_key, "jsd", "Jensen-Shannon distance",
            dataset_key + "_jsd_by_entropy.png",
            deberta_bins, modernbert_bins, baseline_results,
        )
        plot_metric_by_entropy(
            dataset_key, "mean_confidence", "Mean model confidence",
            dataset_key + "_confidence_by_entropy.png",
            deberta_bins, modernbert_bins, baseline_results,
        )
        plot_agreement_vs_entropy(dataset_key, pairwise_bins)

        all_findings[dataset_key] = {
            "deberta": {
                "overall": deberta_summary,
                "entropy_bins": deberta_bins,
                "entropy_edges": deberta_edges,
                "entropy_matching_temperature": deberta_entropy_t,
                "entropy_matched": summarize(apply_temperature(deberta_rows, deberta_entropy_t)),
                "held_out_temperature_scaling": deberta_split_t,
            },
            "modernbert": {
                "overall": modernbert_summary,
                "entropy_bins": modernbert_bins,
                "entropy_edges": modernbert_edges,
                "entropy_matching_temperature": modernbert_entropy_t,
                "entropy_matched": summarize(apply_temperature(modernbert_rows, modernbert_entropy_t)),
                "held_out_temperature_scaling": modernbert_split_t,
            },
            "baselines_2020": baseline_results,
            "model_pair": {
                "overall": pairwise_overall,
                "entropy_bins": pairwise_bins,
            },
            "qualitative_examples": qualitative,
        }

    output_file = os.path.join(RESULTS_DIR, "findings_summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(round_floats(all_findings), f, indent=2, ensure_ascii=False)

    print("\nWrote", output_file)
    print("Plots in", PLOTS_DIR)


if __name__ == "__main__":
    main()
