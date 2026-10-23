# Project Plan: Lead I or Twelve?

**Research question:** How much of a 12-lead CNN's ability to tell normal from abnormal ECGs survives when it only sees the 6 limb leads, or only Lead I, and which diagnostic categories account for the loss?

**Hypothesis:** The loss concentrates in MI and HYP (which need the chest leads) and is small for CD (a timing problem, visible from the limb leads).

**Owners:** **H** = Hudson (data, baselines, evaluation) · **P** = Paco (CNN, experiments, noise) · **Both**

---

## Progress at a glance

- [x] Phase 0: Data pipeline (Checkpoint 1)
- [x] Phase 1: Baselines (H)
- [x] Phase 2: Shared evaluation code (H)
- [ ] Phase 3: 1D ResNet (P)
- [ ] Phase 4: Main experiment, 3 lead sets × 3 seeds (P)
- [ ] Phase 5: Noise robustness (P)
- [ ] Phase 6: Error analysis (Both)
- [ ] Phase 7: Figures and results table (Both)
- [ ] Phase 8: Slides and rehearsal (Both)

**If time runs short:**
1. **Must have:** Phases 1–4 with one seed, plus the per-category recall chart. That alone answers the research question.
2. **Should have:** all 3 seeds, confidence intervals, and the error analysis.
3. **Nice to have:** the noise experiment. If it gets cut, list it under future work.

---

## Phase 0: Data pipeline ✅ (done in `Checkpoint1_PTBXL.ipynb`)

- [x] Download PTB-XL 1.0.3 at 100 Hz (21,799 recordings, 18,869 patients)
- [x] Build the 5 category labels (NORM / MI / STTC / CD / HYP) and the binary `abnormal` target
- [x] Drop the 411 recordings with no diagnostic label, leaving 21,388
- [x] ~~Split by fold: 1–8 train / 9 validation / 10 test~~ → **replaced in Phase 1** by a patient-level 70 / 15 / 15 split, saved to `data/splits.csv`:
  train 14,974 · eyeball 1,069 (5%) · black box 2,138 (10%) · test 3,207
- [x] Check in code that no patient appears in more than one split
- [x] Define lead sets A (12 leads), B (6 limb leads) and C (Lead I only)
- [x] Standardize each lead using training-set statistics only (`to_model_input`)
- [x] Extract the hand-made baseline features (76 / 40 / 10 per recording)

**Cleanup left over from Checkpoint 1**
- [x] Fix the "fot" typo in cell 11
- [ ] Compute the cell 10 numbers (58% abnormal, 2,111 repeat patients) in code instead of typing them in
- [x] Save the shared loading code (labels, splits, `LEAD_SETS`) as `common.py`, so every notebook loads the data the same way. *(The waveform loading and `mu`/`sd` will be added in Phase 3.)*

---

## Phase 1: Baselines (H, about 1 day) ✅

Notebook: `02_baselines.ipynb`

**How each split is used:** train = fit the models · **black box** = numbers only (tuning, choosing the cutoff) · **eyeball** = look at individual errors by hand · test = Phase 2, used once.

- [x] Build the patient-level 70 / 15 / 15 split (validation divided into eyeball 5% + black box 10%) and save it to `data/splits.csv`
- [x] Load the features `F` and pick each lead set's columns with `baseline_features(cfg)`
- [x] Fill missing values with the median, fit on **train only** (`SimpleImputer(strategy="median")`)
- [x] Scale features with `StandardScaler`, fit on train only (the logistic regression needs this)
- [x] Train `LogisticRegression(class_weight="balanced", max_iter=2000)` for A, B and C
- [x] Train `HistGradientBoostingClassifier` for A, B and C (no imputation needed, it handles NaNs)
- [x] Do light tuning on the **black-box set only**: the regularization strength `C` for the logistic regression, and the learning rate and max depth for the trees
- [x] Save probabilities for every split to `results/baseline_{model}_{cfg}.npz`
- [x] Preview per-category recall on the eyeball set, plus one example error

**Done when:** you have a 2 × 3 table of validation AUROC (models × lead sets). ✅

| black-box AUROC | A, 12 leads | B, limb leads | C, Lead I |
|---|---|---|---|
| Logistic regression | 0.886 | 0.873 | 0.813 |
| Gradient-boosted trees | **0.927** | 0.904 | 0.845 |

---

## Phase 2: Shared evaluation code (H, about 1 day) ✅

Files: `common.py` (labels, split, lead sets) · `evaluate.py` (the protocol) · `test_evaluate.py` (11 tests, run `pytest test_evaluate.py`) · `03_evaluation.ipynb`

- [x] **Threshold:** on the **black-box** set, pick the cutoff that gives at least 95% specificity (95% of normal recordings correctly called normal)
- [x] Apply that cutoff to the test set and report:
  - [x] sensitivity at 95% specificity
  - [x] AUROC and AUPRC
  - [x] macro F1
  - [x] confusion matrix
- [x] **Per-category recall:** among test recordings labeled MI (then STTC, CD, HYP), what fraction is flagged abnormal at that cutoff? *This is the test of the hypothesis.*
- [x] **95% confidence intervals:** 1,000 bootstrap resamples of test **patients**, taking the 2.5th and 97.5th percentiles
- [x] **Paired differences** (12 leads − limb, 12 leads − Lead I, limb − Lead I) with confidence intervals, using the same resamples for every model
- [x] Run it on the Phase 1 baseline outputs to make sure it works end to end
- [x] Tests on made-up data where the answer is known, plus 5 deliberately injected bugs to confirm the tests catch them

**Done when:** the baselines have a full results table with confidence intervals and per-category recall. ✅
Outputs: `results/test_metrics_baselines.csv`, `results/test_paired_differences_baselines.csv`, `figures/*.png`

| test set, gradient-boosted trees | AUROC | sensitivity | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|
| A · 12 leads | 0.935 | 75.7% | 81.4% | 84.9% | 76.1% | 83.8% |
| B · limb leads | 0.906 | 65.9% | 71.1% | 74.3% | 71.2% | 75.3% |
| C · Lead I | 0.852 | 56.7% | 58.0% | 70.5% | 58.4% | 73.2% |

---

## Phase 3: 1D ResNet (P, 2–3 days)

Notebook: `03_cnn.ipynb` (run on Colab with a GPU)

**Data**
- [ ] Load the split from `data/splits.csv` (**not** `strat_fold`), and recompute the per-lead `mu`/`sd` on the new train split. Checkpoint 1's `to_model_input` still uses folds 1–8.
- [ ] Early stopping and tuning use the **black-box** set; the eyeball set is only for looking at errors
- [ ] A PyTorch `Dataset` that returns `to_model_input(X, LEAD_SETS[cfg])` and a 6-value target `[abnormal, NORM, MI, STTC, CD, HYP]`
- [ ] Training `DataLoader` with shuffling; validation and test without

**Model**
- [ ] Input layer: `Conv1d(C_in → 64, kernel=15)`, then BatchNorm and ReLU
- [ ] 4 residual stages with 64 → 128 → 192 → 256 channels, each halving the time length (stride 2)
- [ ] Global average pooling, then `Linear(256 → 6)`
- [ ] Loss: `BCEWithLogitsLoss`

**Training (identical for every lead set)**
- [ ] Adam with learning rate 1e-3, batch size 128, up to 30 epochs
- [ ] Learning-rate schedule (cosine or ReduceLROnPlateau)
- [ ] Early stopping on validation AUROC; save the best checkpoint
- [ ] Set seeds for `torch`, `numpy` and `random`

**Tuning**
- [ ] Tune only on **config A, validation only** (learning rate, dropout, depth)
- [ ] Freeze the final settings and write them down in this file:

```
lr = ___   batch = ___   epochs = ___   dropout = ___   kernel = ___   channels = ___
```

**Done when:** config A reaches a validation AUROC of roughly 0.9 or higher (published PTB-XL results are about 0.93).

---

## Phase 4: Main experiment (P, about 1 day, mostly unattended compute)

Only the number of input channels changes between runs.

|            | Seed 0 | Seed 1 | Seed 2 |
|------------|:------:|:------:|:------:|
| A, 12 leads | [ ]    | [ ]    | [ ]    |
| B, 6 limb leads | [ ]    | [ ]    | [ ]    |
| C, Lead I  | [ ]    | [ ]    | [ ]    |

- [ ] Save validation and test probabilities for each run to `results/cnn_{cfg}_seed{s}.npz`
- [ ] Run every run through `evaluate.py`
- [ ] Report the mean ± standard deviation across seeds
- [ ] Note any surprises (for example, B beating A, or C close to A). Those are worth a slide.

---

## Phase 5: Noise robustness (P, about 1 day)

- [ ] Download the MIT-BIH Noise Stress Test Database (`nstdb` on PhysioNet): records `bw` (baseline drift), `em` (electrode motion), `ma` (muscle activity)
- [ ] Resample it from 360 Hz to 100 Hz
- [ ] Add noise to **raw** Lead I test recordings at signal-to-noise ratios of 24, 12 and 6 dB. Noise goes in **before** standardization.
- [ ] Evaluate the trained config C models; **no retraining**
- [ ] Plot AUROC and sensitivity against noise level, one line per noise type

---

## Phase 6: Error analysis (Both, about 1 day)

- [ ] Confusion matrix for each lead set (from `evaluate.py`)
- [ ] Find 3–4 test recordings that **A gets right but C gets wrong**, ideally MI or HYP
- [ ] Plot their 12-lead waveforms and point out the chest-lead changes Lead I can't see
- [ ] Find at least one recording that **every model gets wrong**, and look at why (noise? a borderline label?)
- [ ] Compare the baselines with the CNN: where does the CNN win, and is it mostly in shape-based categories?
- [ ] Write 3–5 bullet points of takeaways for the slides

---

## Phase 7: Figures and results table (Both, about 1 day)

Save every figure to `figures/` as PNG.

- [ ] **Fig 1:** ROC and precision-recall curves for A, B and C on shared axes (CNN solid, best baseline dashed)
- [ ] **Fig 2:** Per-category recall grouped by lead set ⭐ *the headline figure*
- [ ] **Fig 3:** Performance against noise level (Phase 5)
- [ ] **Fig 4:** Waveforms of correct and incorrect predictions (Phase 6)
- [ ] **Fig 5:** Class balance (already in Checkpoint 1)
- [ ] **Table:** models (LR, GBT, CNN) × lead sets (A, B, C): AUROC, AUPRC, sensitivity at 95% specificity, macro F1, with confidence intervals and seed standard deviations

---

## Phase 8: Slides and rehearsal (Both, about 2 days)

About 15 minutes of presentation plus 3 minutes of questions. **You both have to speak.** These are the 10 slides the rubric requires, in order:

- [ ] 1. **Title and team:** title, names, one-sentence objective
- [ ] 2. **Problem and motivation:** smartwatches only record Lead I; why deep learning fits
- [ ] 3. **Research question:** the question and hypothesis, word for word
- [ ] 4. **Related work:** Wagner et al. 2020 (PTB-XL), Strodthoff et al. 2021 (benchmarks); how this project differs
- [ ] 5. **Dataset:** source, counts, labels, splits, class imbalance, limitations (1990s hospital recordings, older and sicker patients)
- [ ] 6. **Method and pipeline:** preprocessing, then baselines and CNN, then output; include an architecture diagram
- [ ] 7. **Experimental design:** 3 nested lead sets, fixed hyperparameters, 3 seeds, 95%-specificity cutoff, how leakage is prevented
- [ ] 8. **Results:** Figs 1–3 and the table
- [ ] 9. **Conclusion:** answer the question directly, without overstating
- [ ] 10. **Limitations and future work:** real wearable noise, population differences, retraining with noise

**Who presents what** (fill in): H: slides ___ · P: slides ___

- [ ] Full timed run-through #1
- [ ] Full timed run-through #2
- [ ] Prepare answers to likely questions (why not a transformer? why 95% specificity? how do you know there's no leakage?)

---

## Final self-check (from the rubric)

- [ ] We can state the research question in one sentence
- [ ] We can explain why the baseline is meaningful
- [ ] Every modeling decision used validation data, never test data
- [ ] Comparisons are fair and reproducible (same settings, seeds written down)
- [ ] The slides show both successes **and failures**
- [ ] The conclusion answers the question without overstating the evidence
- [ ] **Both** of us can explain the dataset, model, results and limitations

## Reproducibility checklist

- [ ] `requirements.txt` with package versions
- [ ] Seeds and hyperparameters written down (Phase 3 box)
- [ ] All notebooks run top to bottom without errors
- [ ] Code pushed to a shared repository
