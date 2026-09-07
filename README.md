# ChaosNLI seminar project

Dujana Abrar, Jiayin Feng

We look at how well two off-the-shelf NLI models match human disagreement on ChaosNLI:

- DeBERTa-v3: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- ModernBERT: `tasksource/ModernBERT-large-nli`

ChaosNLI has three subsets (SNLI, MNLI-m, alphaNLI). We only use SNLI and MNLI-m, because both checkpoints are 3-way NLI classifiers.

## Files

```
ChaosNLI/          dataset, 2020 baseline predictions
src/               experiment code
results/           saved predictions, plots, summary json
report/            LaTeX source
REPORT.pdf         compiled report
requirements.txt
```

## Setup

Run everything from this folder:

```bash
pip install -r requirements.txt
```

You need the ChaosNLI files under `ChaosNLI/data/`. Predictions for both models are already in `results/`, so you do not have to rerun the Hugging Face models unless you want to.

## Run

Score a model (optional; outputs already exist):

```bash
python src/evaluate_models.py deberta snli
python src/evaluate_models.py deberta mnli_m
python src/evaluate_models.py modernbert snli
python src/evaluate_models.py modernbert mnli_m
```

Optional last argument is batch size, e.g. `python src/evaluate_models.py modernbert snli 64`.

Official ChaosNLI metrics vs the 2020 baselines (this is what the report uses):

```bash
python src/analyze_findings.py
```

Writes `results/findings_summary.json` and plots under `results/plots/`.

Earlier pairwise comparison (not the official scoreboard):

```bash
python src/compare_models.py
python src/export_examples_excel.py
```

`src/testing/` has small scripts we used to check that the models load and that the jsonl files look right.

## Report

`REPORT.pdf` is the written report. Source: `report/nli_disagreement.tex` (also copied to `nli_disagreement.tex` in the submission folder). Compile with:

```bash
cd report
pdflatex nli_disagreement.tex
pdflatex nli_disagreement.tex
```
