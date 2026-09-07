---
title: "Evaluating NLI Models Under Disagreement"
subtitle: "DeBERTa-v3 and ModernBERT on ChaosNLI"
author:
  - Dujana Abrar
  - Jiayin Feng
date: "August 2026"
---

# 1. What we asked

Natural Language Inference (NLI) is usually scored as if each premise–hypothesis pair had one correct label. ChaosNLI (Nie et al., 2020) shows that this is often false: with 100 annotations per example, many SNLI and MNLI items have substantial human disagreement.

Nie et al. found that 2020-era models (BERT, RoBERTa, XLNet, BART, ALBERT) (1) fail to recover human label distributions and (2) collapse to near-chance accuracy on high-disagreement items. The open question for this project is whether **later, stronger NLI models** close that gap.

We evaluate two widely used off-the-shelf NLI checkpoints:

- **DeBERTa-v3**: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (trained on MNLI, FEVER-NLI, ANLI, LingNLI, WANLI; SNLI was deliberately excluded)
- **ModernBERT**: `tasksource/ModernBERT-large-nli` (multi-task NLI training, including MNLI, ANLI, WANLI, LingNLI, and many other NLI-format tasks)

on ChaosNLI-SNLI (1,514 examples) and ChaosNLI-MNLI-matched (1,599 examples), using the **official ChaosNLI metrics**: Jensen–Shannon distance (JSD), KL divergence, old accuracy (original dataset label), and new accuracy (100-annotator majority).

# 2. Is this already done?

We checked ACL Anthology, arXiv, the ChaosNLI scoreboard, and Hugging Face model cards. **These two checkpoints have not been evaluated on ChaosNLI.** Closest prior work:

| Work | What it did | Why it is not this project |
| --- | --- | --- |
| Nie et al. (EMNLP 2020) | BERT / RoBERTa / XLNet / BART / ALBERT on ChaosNLI | 2020 encoders only |
| Zhou et al. (2022), Meissner et al. (2021), Wang et al. (2022) | Training or calibration to match human distributions | They *train* for disagreement; we test frozen SOTA NLI models |
| Lee et al. (EMNLP 2023) | GPT-3 / Flan-T5 on ChaosNLI | Generative LLMs, not encoder NLI checkpoints |
| 2025–2026 calibration papers | DeBERTa-v3-**base** fine-tuned on MNLI, then recalibrated | Not the large multi-dataset NLI checkpoint; not ModernBERT |

Lee et al. already showed that scaling to GPT-3 does **not** beat RoBERTa-large on ChaosNLI JSD. What remains untested is whether **encoder NLI progress since 2020** — DeBERTa-v3 and ModernBERT, both trained on much more NLI data — transfers to human-disagreement modeling.

That is the gap this report fills.

# 3. Headline findings

**Finding 1. Majority-label accuracy and human-distribution matching have come apart.**
DeBERTa-v3 is the strongest model we tested on ChaosNLI majority labels (SNLI new accuracy 79.3%, above RoBERTa-large's 78.7%), but it is **worse** than every 2020 baseline on JSD and KL. ModernBERT is the opposite: it **beats all 2020 models on JSD and KL**, including on MNLI, while its majority accuracy is slightly below RoBERTa-large.

**Finding 2. DeBERTa-v3 is more overconfident than 2020 RoBERTa.**
On SNLI its mean max-probability is 0.95 and its mean predictive entropy is 0.21 bits, versus human entropy 0.80 bits. On MNLI the entropy gap is even larger (0.80 bits), and DeBERTa confidence is almost uncorrelated with human uncertainty (Spearman $\rho = -0.05$). High-disagreement SNLI items still receive 92% mean confidence while majority accuracy falls to 61%.

**Finding 3. Two independently trained modern NLI models often agree with each other when humans do not.**
On high-entropy SNLI, DeBERTa and ModernBERT still predict the same label 68% of the time, and in 22% of those high-entropy items they agree with each other **and** disagree with the human majority. The models have converged on a shared hard-label solution, not on the human opinion distribution.

**Finding 4. A single temperature parameter recovers most of the JSD gap — without changing accuracy.**
Held-out temperature scaling drops DeBERTa SNLI JSD from 0.276 to 0.187 ($T = 2.75$) and MNLI JSD from 0.358 to 0.206 ($T = 4.50$). Accuracy is unchanged. The missing piece is calibration to ambiguity, not a better argmax classifier.

# 4. Official ChaosNLI comparison

Lower JSD / KL is better. Higher accuracy is better. 2020 numbers were recomputed from the official ChaosNLI prediction files and match the paper scoreboard.

## ChaosNLI-SNLI ($n = 1{,}514$)

| Model | JSD $\downarrow$ | KL $\downarrow$ | Old acc. $\uparrow$ | New acc. $\uparrow$ |
| --- | ---: | ---: | ---: | ---: |
| BERT-large (2020) | 0.230 | 0.502 | 0.727 | 0.738 |
| RoBERTa-large (2020) | 0.221 | 0.494 | 0.749 | 0.787 |
| XLNet-large (2020) | 0.226 | 0.505 | 0.743 | 0.781 |
| BART-large (2020) | 0.220 | 0.471 | 0.742 | 0.783 |
| ALBERT-xxlarge (2020) | 0.235 | 0.534 | 0.715 | 0.781 |
| **DeBERTa-v3 (ours)** | **0.278** | **0.858** | 0.709 | **0.793** |
| **ModernBERT (ours)** | **0.214** | **0.378** | 0.708 | 0.768 |

DeBERTa-v3 has the **best new accuracy** and the **worst distribution scores**. Its old SNLI accuracy is also lower than RoBERTa's (0.709 vs 0.749). That is consistent with the checkpoint card: SNLI was excluded from training because of label quality. The model matches the 100-annotator majority better than the original 5-annotator SNLI labels — and still produces distributions that are far too peaked.

## ChaosNLI-MNLI ($n = 1{,}599$)

| Model | JSD $\downarrow$ | KL $\downarrow$ | Old acc. $\uparrow$ | New acc. $\uparrow$ |
| --- | ---: | ---: | ---: | ---: |
| BERT-large (2020) | 0.315 | 0.845 | 0.612 | 0.569 |
| RoBERTa-large (2020) | 0.311 | 0.870 | 0.674 | 0.635 |
| XLNet-large (2020) | 0.312 | 0.882 | 0.674 | 0.619 |
| BART-large (2020) | 0.317 | 0.885 | 0.664 | 0.592 |
| ALBERT-xxlarge (2020) | 0.316 | 0.862 | 0.649 | 0.590 |
| **DeBERTa-v3 (ours)** | **0.364** | **1.465** | **0.695** | 0.634 |
| **ModernBERT (ours)** | **0.285** | **0.638** | 0.671 | 0.624 |

On MNLI, 2020 models were already close to the chance JSD baseline ($\approx 0.31$). ModernBERT is the first encoder in this comparison to clearly beat that cluster (JSD 0.285, KL 0.638). DeBERTa-v3 improves old MNLI accuracy (0.695 vs RoBERTa 0.674) while making KL almost twice as bad.

This is the result the original paper predicted: **the ability to hit the majority label is not the same as the ability to model human disagreement.** Five years of NLI accuracy gains did not automatically fix the second problem. One of the two modern models made JSD worse.

![Jensen–Shannon distance by human-disagreement tertile on ChaosNLI-SNLI. DeBERTa-v3 is worse than 2020 RoBERTa-large in every bin; ModernBERT is better.](results/plots/snli_jsd_by_entropy.png){width=85%}

![Jensen–Shannon distance by human-disagreement tertile on ChaosNLI-MNLI. ModernBERT is the first model in this comparison to clearly beat the 2020 JSD cluster.](results/plots/mnli_m_jsd_by_entropy.png){width=85%}

# 5. Disagreement bins

Examples are split into tertiles of ChaosNLI annotation entropy (low / medium / high human disagreement).

## Agreement with the 100-annotator majority

| Split | Low | Medium | High |
| --- | ---: | ---: | ---: |
| SNLI DeBERTa-v3 | 0.948 | 0.820 | 0.612 |
| SNLI ModernBERT | 0.926 | 0.781 | 0.598 |
| SNLI RoBERTa-large | 0.952 | 0.806 | 0.602 |
| MNLI DeBERTa-v3 | 0.705 | 0.619 | 0.576 |
| MNLI ModernBERT | 0.711 | 0.584 | 0.576 |
| MNLI RoBERTa-large | 0.704 | 0.636 | 0.567 |

On high-disagreement items, all three models sit around 60% (SNLI) and 57% (MNLI). Nie et al.'s core observation still holds: remaining errors concentrate where humans already disagree, and newer models do not solve that slice.

![Agreement with the 100-annotator majority by human-disagreement tertile on ChaosNLI-SNLI. High-disagreement accuracy remains near 60% for all models.](results/plots/snli_new_acc_by_entropy.png){width=85%}

## Mean confidence on the same bins (SNLI)

| Model | Low | Medium | High |
| --- | ---: | ---: | ---: |
| DeBERTa-v3 | 0.978 | 0.946 | 0.918 |
| ModernBERT | 0.924 | 0.850 | 0.795 |
| RoBERTa-large | 0.953 | 0.878 | 0.816 |

DeBERTa-v3 barely becomes less confident as humans become more divided. ModernBERT tracks disagreement more, which is why its JSD is better even though its argmax accuracy is not.

![Mean model confidence by human-disagreement tertile on ChaosNLI-SNLI. DeBERTa-v3 stays above 91% confident even on high-disagreement items.](results/plots/snli_confidence_by_entropy.png){width=85%}

On MNLI, DeBERTa confidence vs human entropy is $\rho = -0.05$ (essentially none). ModernBERT is only weakly better ($\rho = -0.17$). Neither model's softmax is a reliable uncertainty signal for MNLI ambiguity.

# 6. Model–model consensus vs human disagreement

This comparison is not in Nie et al. (2020) and is not reported for these checkpoints elsewhere. We treat DeBERTa-v3 and ModernBERT as two independently trained “votes” and ask whether they echo each other more than they echo humans.

| SNLI entropy tertile | Models agree with each other | Both agree, both differ from human majority | Mean DeBERTa confidence |
| --- | ---: | ---: | ---: |
| Low | 0.913 | 0.018 | 0.978 |
| Medium | 0.794 | 0.095 | 0.946 |
| High | 0.679 | 0.224 | 0.918 |

| MNLI entropy tertile | Models agree with each other | Both agree, both differ from human majority | Mean DeBERTa confidence |
| --- | ---: | ---: | ---: |
| Low | 0.803 | 0.191 | 0.930 |
| Medium | 0.788 | 0.291 | 0.941 |
| High | 0.705 | 0.263 | 0.915 |

On MNLI, even in the **low**-disagreement bin, the two models jointly miss the human majority 19% of the time. On high-disagreement SNLI, they still agree with each other two thirds of the time. Hard-label NLI training appears to have produced a shared convention that is more consistent across model families than across human annotators.

![DeBERTa-v3 vs ModernBERT consensus on ChaosNLI-SNLI. On high-entropy items the two models still share a label 68% of the time, and jointly miss the human majority in 22% of cases.](results/plots/snli_model_consensus_by_entropy.png){width=85%}

![The same comparison on ChaosNLI-MNLI. Joint errors remain high even in the low-disagreement bin (19%).](results/plots/mnli_m_model_consensus_by_entropy.png){width=85%}

# 7. Can simple calibration fix it?

The project proposal allowed temperature scaling if the first results showed a confidence–uncertainty mismatch. They do.

We recover logits as $\log(p)$, search $T$ on a random 50% split to minimize JSD, and evaluate on the held-out half. Argmax accuracy cannot change under $T > 0$.

| Model / split | $T$ | Held-out JSD before $\rightarrow$ after | Held-out KL before $\rightarrow$ after |
| --- | ---: | --- | --- |
| DeBERTa SNLI | 2.75 | 0.276 $\rightarrow$ 0.187 | 0.848 $\rightarrow$ 0.195 |
| ModernBERT SNLI | 1.75 | 0.217 $\rightarrow$ 0.188 | 0.389 $\rightarrow$ 0.206 |
| DeBERTa MNLI | 4.50 | 0.358 $\rightarrow$ 0.206 | 1.430 $\rightarrow$ 0.211 |
| ModernBERT MNLI | 2.75 | 0.278 $\rightarrow$ 0.192 | 0.603 $\rightarrow$ 0.183 |

Two observations:

1. After softening, both models land near JSD $\approx 0.19$. Most of DeBERTa's JSD penalty was overconfidence, not a wrong ranking of labels.
2. DeBERTa needs a much larger $T$, especially on MNLI ($T = 4.5$). That is a quantitative measure of how peaked the checkpoint is.

Temperature scaling is **not** a full solution: it applies one global sharpness and cannot represent item-specific bimodal disagreement (for example, humans split 50/50 between entailment and contradiction). It does show that the off-the-shelf softmax is the wrong object to compare with ChaosNLI, unless it is recalibrated.

Matching mean model entropy to mean human entropy (no label leakage, only entropy matching) gives the same qualitative picture: DeBERTa SNLI JSD falls from 0.278 to 0.187 at $T = 2.6$.

# 8. Qualitative error patterns

High-entropy failures are not random. Recurring patterns:

**Lexical contradiction that humans do not treat as contradiction.**
Premise: *A man in a dark suit talks with a middle-aged man and woman on a stage…*
Hypothesis: *A couple of people are sitting in a room.*
Humans: N 58 / C 24 / E 18. Both models: contradiction at $\ge$98% confidence. The models latch onto *talks* vs *sitting* and ignore that the pair is still compatible with a loose scene description, which is why many annotators chose neutral.

**Pragmatic inference vs strict entailment.**
Premise: *A boy in jeans, black t-shirt, and blue cap on a bike, leaps over steps.*
Hypothesis: *A boy is practicing bike tricks.*
Humans split E 45 / N 45. Both models: neutral at $\ge$95%. The original SNLI label was also neutral. The 100-annotator majority is a tie toward entailment; the models reproduce the old conservative convention.

**Paraphrase / word-order items.**
Premise: *Closed on the Sabbath.* Hypothesis: *Sabbath is closed.*
Humans are split (C 39 / E 37 / N 24). Both models: entailment at 98%. This is a high-confidence, high-JSD miss on an item that is genuinely ambiguous (near-paraphrase vs not-at-issue content).

**When the models disagree, ModernBERT is usually softer.**
On *A skateboard does a trick without a skateboarder*, humans are mixed (N 49 / C 37). DeBERTa is 99% neutral; ModernBERT is 48% contradiction / 41% neutral and has JSD 0.08 versus DeBERTa's 0.45. ModernBERT's better aggregate JSD is visible at the example level: it puts mass on more than one plausible label.

# 9. Why these results happen

The numbers are not a mystery. They follow from how these models were trained, what JSD/KL actually punish, and what human disagreement on NLI actually is.

## Hard-label training rewards a peaked softmax

Both checkpoints are standard 3-way classifiers trained with cross-entropy on a single gold label per example. That loss is minimized by putting almost all probability on one class. ChaosNLI, by contrast, scores the *full* softmax against a 100-person distribution that is often spread across two labels.

JSD and especially KL are harsh on this mismatch. If humans put 20% on contradiction and the model puts 0.1%, KL explodes even when the model's top label is the majority. That is why DeBERTa-v3 can be the **best** majority classifier and the **worst** distribution model at the same time: it is very good at picking a mode, and very bad at leaving residual mass on the labels that annotators still choose.

This is the same dissociation Nie et al. (2020) already reported (DistilBERT had the best KL and the worst accuracy). We show it has not gone away; if anything, a stronger 2022 NLI model made it worse.

## DeBERTa-v3 is a sharper majority-label specialist

Three properties of this checkpoint explain the overconfidence.

1. **More NLI data, still one-hot targets.** It was fine-tuned on MNLI, FEVER-NLI, ANLI, LingNLI, and WANLI (about 885k pairs). Extra hard-label data improves the ranking of the majority class (hence SNLI new accuracy 79.3% and MNLI old accuracy 69.5%), but it also trains the model to be more certain. Mean SNLI confidence is 0.95; mean predictive entropy is 0.21 bits against human entropy 0.80.

2. **SNLI was deliberately excluded.** The model card states that SNLI was left out because of label quality. That shows up directly: DeBERTa's *old* SNLI accuracy (0.709) is below RoBERTa (0.749), while *new* accuracy against the 100-annotator majority is higher (0.793 vs 0.787). The checkpoint avoided fitting the original 5-annotator SNLI artifacts, so it matches the re-annotated majority better --- but it still emits a near-one-hot distribution.

3. **Confidence does not track ambiguity.** On MNLI, Spearman $\rho$ between DeBERTa confidence and human entropy is $-0.05$. The model is about as confident on messy items as on clean ones (mean confidence 0.93 even in the high-entropy tertile). So the softmax is not an uncertainty estimate; it is a peaked class score. That is why high-disagreement SNLI items still get 92% mean confidence while majority accuracy falls to 61%.

Temperature scaling is the diagnostic for this account. A global $T = 2.75$ (SNLI) or $T = 4.5$ (MNLI) cuts DeBERTa JSD roughly in half **without changing a single predicted label**. The ranking was often already right; the sharpness was wrong. MNLI needing $T = 4.5$ is a quantitative statement that this checkpoint's logits are extremely peaked.

## ModernBERT is softer, not more accurate

ModernBERT's training mix is broader and more heterogeneous: many NLI-format tasks (MNLI, ANLI, WANLI, LingNLI, SICK, logic/document NLI, label-NLI, and others) under multi-task fine-tuning. Two consequences follow.

First, it never becomes as peaky. Mean SNLI confidence is 0.86 rather than 0.95; mean model entropy is 0.53 bits rather than 0.21. A softer distribution is automatically closer to a spread-out human distribution, which is why JSD/KL improve even when the argmax is not better.

Second, WANLI and related datasets are built from annotator disagreement. The model therefore sees more label variation during training than a classic MNLI-only encoder, even though it is still trained with hard labels. That does not teach it the ChaosNLI 100-way distribution, but it can discourage collapsing onto a single convention as hard as DeBERTa does.

This also explains the example-level pattern: on *A skateboard does a trick without a skateboarder*, humans split N/C. DeBERTa is 99% neutral (JSD 0.45); ModernBERT puts 48% on contradiction and 41% on neutral (JSD 0.08). ModernBERT is not “smarter” about the item; it is less willing to zero out a plausible alternative. That is sufficient to win JSD and not sufficient to win accuracy.

The original ChaosNLI paper already warned that accuracy and distribution matching are different abilities. ModernBERT is the 2025 instance of the DistilBERT pattern: better distribution scores, not better majority scores.

## High-disagreement items are a different problem from “hard NLI”

Accuracy collapsing to about 60% on the high-entropy tertile is expected if those items are **underspecified**, not merely difficult. Typical sources in our qualitative sample:

- lexical mismatch that some annotators treat as contradiction and others as irrelevant (*talks* vs *sitting*);
- pragmatic enrichment vs strict entailment (*leaps over steps* $\rightarrow$ *practicing tricks*);
- near-paraphrase or word-order variants (*Closed on the Sabbath* / *Sabbath is closed*).

A better encoder can learn more of the majority convention on low-entropy items (DeBERTa's 94.8% SNLI low-bin accuracy). It cannot invent a second annotator. Nothing in hard-label NLI training represents “45% of people will call this entailment.” So the high-disagreement slice stays near the 2020 numbers, and most remaining errors still live there, exactly as Nie et al. argued.

## Why the two modern models agree with each other

DeBERTa-v3 and ModernBERT have different architectures and training recipes, but they share the same 3-way label space and heavily overlapping NLI sources (MNLI, ANLI, WANLI, LingNLI). They therefore learn the same **dataset convention**: lexical overlap, conservative neutrality on extra details, and sharp contradiction on mismatched verbs.

Humans disagree for linguistic and world-knowledge reasons that are not aligned with that convention. So on high-entropy SNLI the two models still pick the same label 68% of the time, and in 22% of those items they agree with each other **and** not with the human majority. On MNLI they jointly miss the majority even in the *low*-entropy bin (19%). This is not two systems independently discovering the same truth; it is two systems trained on similar one-hot NLI data reproducing the same answer key.

## What temperature scaling can and cannot explain

Temperature scaling supports the overconfidence story and bounds it. After $T$ is fit, both models land near JSD $\approx 0.19$, which means a large part of DeBERTa's JSD penalty was global sharpness. It does **not** mean the models now capture disagreement: one $T$ cannot represent an item that is 50/50 entailment vs contradiction while the next item is a clean 95% entailment. That would require instance-specific uncertainty, which hard-label training does not provide.

## What we are *not* claiming

These are mechanistic readings of the metrics and training setups, not a causal ablation. We did not retrain DeBERTa with ModernBERT's mix, nor ModernBERT with one-hot MNLI only. The account is: **hard-label NLI progress sharpens the majority decision; ChaosNLI scores the leftover probability mass; DeBERTa optimized the first and paid on the second; ModernBERT stayed softer and won JSD without winning accuracy.** That is enough to explain the pattern, and it is consistent with Nie et al. (2020), Meissner et al. (2021), and Baan et al. (2022).

# 10. What to claim in the submission

Safe, supportable claims:

1. **This evaluation is new.** The two public NLI checkpoints had not been scored on ChaosNLI with the official JSD / KL / old / new protocol.
2. **Standard NLI progress since 2020 is not progress on disagreement.** DeBERTa-v3 improves majority accuracy and **worsens** distribution matching relative to RoBERTa-large.
3. **ModernBERT is the interesting exception on JSD/KL**, including the first clear JSD win on ChaosNLI-MNLI among the models we compared — but it does not beat RoBERTa on majority accuracy, so the two capabilities still do not move together.
4. **High-disagreement items remain unsolved**, and modern models remain overconfident there.
5. **Independent modern NLI models agree with each other more than with divided humans**, which suggests a shared training convention rather than recovered human uncertainty.
6. **Post-hoc temperature scaling closes much of the JSD gap**, which implies the main defect is miscalibrated sharpness, not a failure to rank the majority label.

What we should **not** claim: that ModernBERT “solves” ChaosNLI, that DeBERTa is a worse NLI model in general, or that temperature scaling captures genuine annotator disagreement. High-entropy accuracy is still near the 2020 numbers.

# 11. Limitations

- We evaluate frozen checkpoints, not models trained on ChaosNLI soft labels. Soft-label training is already known to help (Zhou et al., 2022) and would be a different experiment.
- $\alpha$NLI is in ChaosNLI but not in this report (the selected checkpoints are 3-way NLI classifiers).
- Temperature scaling uses a random split of ChaosNLI itself. It is diagnostic, not a deployed calibrator.
- JSD/KL follow the official ChaosNLI implementation (SciPy Jensen–Shannon *distance*, KL in nats) so they can be compared with the 2020 scoreboard.

# 12. How to reproduce

```bash
python src/evaluate_models.py deberta snli
python src/evaluate_models.py deberta mnli_m
python src/evaluate_models.py modernbert snli
python src/evaluate_models.py modernbert mnli_m
python src/analyze_findings.py
```

Predictions are in `results/*_predictions.json`. Paper-comparable metrics, entropy bins, temperature scaling, and qualitative examples are in `results/findings_summary.json`. Plots are in `results/plots/`.

# References

- Nie, Y., Zhou, X., and Bansal, M. 2020. What Can We Learn from Collective Human Opinions on Natural Language Inference Data? *EMNLP*.
- Zhou, X., Nie, Y., and Bansal, M. 2022. Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning. *Findings of ACL*.
- Lee, N., Kella, C., and Choi, Y. 2023. Can Large Language Models Capture Dissenting Human Voices? *EMNLP*.
- Meissner, J. M., et al. 2021. Capturing Label Distribution: A Case Study in NLI.
- Baan, J., et al. 2022. Stop Measuring Calibration When Humans Disagree. *EMNLP*.
- Wang, Y., et al. 2022. Capture Human Disagreement Distributions by Calibrated Networks for Natural Language Inference. *Findings of ACL*.
- Laurer, M., et al. DeBERTa-v3-large-mnli-fever-anli-ling-wanli. Hugging Face. https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- Sileo, D. / tasksource. ModernBERT-large-nli. Hugging Face. https://huggingface.co/tasksource/ModernBERT-large-nli
