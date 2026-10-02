# SemEval-2026 (POLAR): Polarization Manifestation Identification in Telugu

A multilingual NLP system for **SemEval-2026 Task 9 (POLAR), Subtask 3: Manifestation Identification**, focused on detecting *how* polarization is expressed in Telugu and Telugu-English social media text.

The project studies two complementary approaches under a low-resource, highly imbalanced setting:

1. **Prompt engineering** with a locally hosted Llama-3 8B model
2. **Fine-tuning multilingual encoders** with imbalance-aware losses, threshold calibration, and weighted ensembling

---

## Task Overview

The task is a **six-label multi-label classification problem**. A post may contain multiple polarization manifestations at the same time.

| Label | What it captures |
|---|---|
| Stereotype | Generalizing a group through fixed traits |
| Vilification | Portraying a group as evil or threatening |
| Dehumanization | Denying or degrading a group's humanity |
| Extreme Language / Absolutism | Inflammatory or absolute language |
| Lack of Empathy / Understanding | Dismissing suffering or perspectives |
| Invalidation | Denying legitimacy of views or experiences |

**Primary metric:** Macro-F1 across the six labels.

Macro-F1 gives equal weight to every label, making performance on rare manifestations important rather than allowing frequent labels to dominate the score.

---

## Dataset

The official POLAR dataset covers **22 languages** and more than **110K annotated instances**. The Telugu split contains:

| Split | Samples |
|---|---:|
| Training | 2,366 |
| Development | 118 |
| Test | 1,066 |
| **Total** | **3,550** |

The training set contains **1,274 polarized** and **1,092 non-polarized** samples.

### Telugu Label Distribution (Training Set)

| Label | Positive Samples | Prevalence |
|---|---:|---:|
| Lack of Empathy | 622 | 26.3% |
| Invalidation | 539 | 22.8% |
| Vilification | 536 | 22.7% |
| Extreme Language | 318 | 13.4% |
| Stereotype | 265 | 11.2% |
| Dehumanization | 59 | 2.5% |

The strongest imbalance is **dehumanization**, with only **59 positive examples**. This scarcity became a central modelling and error-analysis consideration.

---

## Approach

```text
                         Telugu Social Media Data
                                  |
                                  v
                       Preprocessing & Label Analysis
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             Phase 1: Prompting          Phase 2: Fine-tuning
               Llama-3 8B              XLM-R / MuRIL / XLM-R Large
                    |                           |
             Prompt ablations          Loss / model / data ablations
                    |                  Threshold calibration
                    |                  Weighted ensembling
                    +-------------+-------------+
                                  |
                                  v
                            Macro-F1 Analysis
```

### Phase 1: Prompt Engineering

A locally hosted **Llama-3 8B** model was evaluated through **Ollama** using multiple prompt structures, including:

- Robustness variation
- Active recall
- Structured subtasks
- Self-verification
- Minimal instruction
- Step-by-step reasoning
- Other prompt variants

**Prompt baseline.** The initial generic prompt achieved **0.313 Macro-F1**.

**Targeted stereotype prompt.** Explicitly modelling recurring implicit stereotype patterns substantially improved positive-example detection:

> **Stereotype recall: 66.7% → 94.2%**, measured on 120 positive stereotype examples.

**Best prompt configuration.** Robustness-variation prompting achieved **0.605 Macro-F1**.

The prompt evaluation used a **filtered validation set of 126 examples**, retaining only rows with at least one positive manifestation label. This prevents an all-zero prediction strategy from dominating the evaluation.

### Phase 2: Multilingual Encoder Fine-tuning

Multilingual transformer encoders were trained using **5-fold stratified cross-validation**.

| Category | Techniques |
|---|---|
| Models | XLM-RoBERTa, MuRIL, XLM-RoBERTa-Large |
| Losses | Weighted BCE, Focal Loss, Asymmetric Loss |
| Regularization | Label smoothing |
| Decision rule | Per-label threshold calibration |
| Combination | Weighted model ensembling |
| Data | External data augmentation |

Common training controls included:

- Learning-rate warmup
- Early stopping
- Gradient clipping
- Stratified folds
- Per-label positive weighting
- Fixed random seed for fine-tuning experiments

---

## Experimental Results

The two phases use **different evaluation settings**. Their scores are reported separately and should not be treated as directly comparable.

### Prompt-based Results

**Llama-3 8B, filtered validation set of 126 non-zero-label examples**

| Configuration | Macro-F1 |
|---|---:|
| Baseline prompt | 0.313 |
| **Robustness-variation prompting** | **0.605** |

### Fine-tuned Encoder Results

**5-fold stratified cross-validation, out-of-fold (OOF) Macro-F1**

| Configuration | OOF Macro-F1 |
|---|---:|
| **MuRIL + XLM-R weighted ensemble** | **0.473** |
| XLM-R Large + label smoothing | 0.460 |
| ASL + threshold calibration | 0.447 |
| Instruction-style encoder | 0.428 |
| Two-stage focal-loss pipeline | 0.414 |
| External data augmentation | 0.356 |
| ASL + label smoothing | 0.155 |

The strongest encoder configuration was a **weighted MuRIL + XLM-R ensemble** at **0.473 OOF Macro-F1**. This is an internal cross-validation result, not an official hidden-test leaderboard score.

---

## Official Benchmark Context

The official SemEval-2026 Task 9 paper reports Telugu results for Subtask 3 on the hidden test set.

| Official Rank | System | Macro-F1 |
|---|---|---:|
| 1st | SMASH | 0.445 |
| 2nd | PolaFusion | 0.429 |
| 3rd | Sagarmatha | 0.424 |
| — | Official baseline | 0.392 |

These results use the hidden test set and therefore a different evaluation protocol from this project's internal OOF and filtered-validation experiments. **The 0.473 OOF score is not comparable to these leaderboard scores.**

---

## Key Findings

1. **Prompt structure had a large effect.** Changing only the prompt structure improved Macro-F1 from 0.313 to 0.605 on the same filtered validation set. In a low-resource multilingual setting, task formulation can materially change model behaviour without changing model weights.

2. **MuRIL contributed to the strongest encoder ensemble.** The MuRIL + XLM-R weighted ensemble achieved the best internal fine-tuning result at 0.473 OOF Macro-F1.

3. **Larger model size did not resolve the data bottleneck.** XLM-R Large reached 0.460, below the smaller-model ensemble at 0.473. Increasing encoder capacity alone was not enough to overcome the limits of the training data.

4. **Rare labels dominated the difficulty.** Dehumanization had only 59 positive training examples, about 12 per validation fold, so a few prediction changes can swing its per-label F1 substantially.

5. **Label smoothing was harmful under severe imbalance.** ASL combined with label smoothing collapsed to 0.155 OOF Macro-F1. A regularization strategy that helps in balanced classification does not necessarily transfer to a severely imbalanced multi-label setting.

6. **Complex fine-tuning strategies converged.** Several substantially different approaches landed in the 0.41–0.47 OOF Macro-F1 range, suggesting that data scale, rare-label coverage, and distribution were the main constraints rather than architecture.

7. **External augmentation did not automatically help.** External data augmentation scored 0.356 OOF Macro-F1, below the stronger configurations, highlighting the importance of distribution alignment when adding external examples to a low-resource target dataset.

---

## Error Analysis

**Rare-label difficulty.** Dehumanization was the most data-scarce manifestation, with only 59 positive training examples.

**Semantically close manifestations.** Categories with overlapping rhetorical meaning were difficult to separate, particularly:

- Vilification
- Dehumanization
- Extreme Language
- Lack of Empathy
- Invalidation

These overlaps motivated targeted prompt design and per-label threshold calibration.

---

## Experimental Design

The project used controlled experiments to isolate the effect of individual modelling decisions.

**Cross-validation.** Fine-tuned encoders were evaluated with 5 stratified folds, out-of-fold predictions, and Macro-F1 as the primary metric.

**Threshold calibration.** Instead of a universal 0.5 probability threshold, thresholds were calibrated independently for each label using out-of-fold predictions. This matters because the six manifestations have very different prevalence.

**Ablation strategy.** Experiments varied:

- Prompt structure
- Encoder architecture
- Loss function
- Label smoothing
- Data augmentation
- Thresholds
- Ensemble weights

This made it possible to identify both improvements and failure modes, rather than reporting only the strongest configuration.

---

## Reproducing the Project

```bash
git clone https://github.com/Aashritha-2005/semeval-2026.git
cd semeval-2026
pip install -r requirements.txt
```

**1. Preprocess the dataset**

```bash
python data_preprocessing.py
```

Produces the cleaned training data used by the downstream pipeline.

**2. Train the multilingual encoders**

```bash
python final_train.py
```

Runs fine-tuning with the 5-fold cross-validation setup.

**3. Ensemble and calibrate**

```bash
python ensemble_v2.py
```

Combines model outputs and applies per-label threshold calibration.

**4. Generate predictions**

```bash
python inference_v2.py
```

Writes predictions for the test set to `output_v2/`.

### Result-to-Code Mapping

| Result | Evaluation | Pipeline |
|---|---|---|
| 0.473 OOF Macro-F1 | 5-fold CV | `final_train.py` → `ensemble_v2.py` |
| 0.605 Macro-F1 | 126-example filtered validation | Prompt-engineering experiments conducted during development |

---

## Repository Structure

| File / Directory | Purpose |
|---|---|
| `data_preprocessing.py` | Data preprocessing and label analysis |
| `baseline1.py` | Initial baseline |
| `baseline2_with_split.py` | Baseline with data split |
| `base_pipeline.py` | Shared training utilities |
| `final_train.py` | Final multilingual encoder training |
| `ensemble_v2.py` | Weighted ensemble and threshold calibration |
| `inference_v2.py` | Test-set inference |
| `output_v2/` | Generated prediction outputs |
| `mps/` | Local experiments on Apple Silicon (MPS backend) |

---

## Tech Stack

Python · PyTorch · Hugging Face Transformers · pandas · NumPy · scikit-learn · Ollama · Llama-3 8B · XLM-RoBERTa · MuRIL

---

## Limitations

- The prompting result (0.605) uses a filtered validation set, while the encoder results use 5-fold OOF evaluation, so the two are not directly comparable.
- The 0.473 result is an internal OOF score, not an official hidden-test score.
- The dehumanization label has only 59 positive training examples, so its performance is statistically noisy.
- External augmentation did not improve validation performance, showing the difficulty of transferring examples across distributions.

---

## Official Task Reference

**SemEval-2026 Task 9 (POLAR): Detecting Multilingual, Multicultural and Multievent Online Polarization**
Subtask 3: POLAR MANIFEST, Manifestation Identification

- Task paper: Naseem et al. (2026), [arXiv:2604.06817](https://arxiv.org/abs/2604.06817)
- Task website: [POLAR @ SemEval-2026](https://polar-semeval.github.io/)

---

## Project Summary

This project explores how prompt design, multilingual representation learning, imbalance-aware objectives, calibration, and model ensembling affect multi-label polarization manifestation detection in low-resource Telugu social media text.

| Result | Value |
|---|---|
| Best prompt-based result (filtered 126-example validation) | **0.605 Macro-F1** |
| Best multilingual encoder ensemble | **0.473 OOF Macro-F1** |
| ASL + label smoothing failure case | 0.155 OOF Macro-F1 |
| Dehumanization training positives | 59 |
| Encoder evaluation protocol | 5-fold stratified CV |

The project emphasizes measurement-driven experimentation and error analysis, rather than reporting only a single final model score.
