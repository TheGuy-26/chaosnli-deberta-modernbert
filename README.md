# ChaosNLI seminar project

Dujana Abrar, Jiayin Feng

We evaluate two off-the-shelf NLI models on ChaosNLI:

- DeBERTa-v3: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- ModernBERT: `tasksource/ModernBERT-large-nli`

We use ChaosNLI-SNLI and ChaosNLI-MNLI-m (not alphaNLI).

## Files

```
ChaosNLI/          official dataset and 2020 baseline predictions
src/evaluate_models.py
src/analyze_findings.py
results/           predictions, findings_summary.json, plots
report/nli_disagreement.tex
REPORT.pdf
requirements.txt
```

`ChaosNLI/` is the public release from Nie et al. (EMNLP 2020).

## Setup

From this folder:

```bash
pip install -r requirements.txt
```

Predictions are already in `results/`. You do not need to rerun the models unless you want to.

## Run

```bash
python src/evaluate_models.py deberta snli
python src/evaluate_models.py deberta mnli_m
python src/evaluate_models.py modernbert snli
python src/evaluate_models.py modernbert mnli_m
```

Optional batch size: `python src/evaluate_models.py modernbert snli 64`.

Then recompute the official ChaosNLI metrics (JSD, KL, old/new accuracy) and plots:

```bash
python src/analyze_findings.py
```

This writes `results/findings_summary.json` and `results/plots/`.

## Report

The report is `REPORT.pdf`. Source: `report/nli_disagreement.tex`.

```bash
cd report
pdflatex nli_disagreement.tex
pdflatex nli_disagreement.tex
cp nli_disagreement.pdf ../REPORT.pdf
```
