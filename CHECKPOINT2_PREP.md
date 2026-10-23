# Checkpoint 2 Prep: Hudson & Paco

**Project:** Lead I or Twelve? Measuring how much heart-disease signal survives a wearable-style single-lead ECG (PTB-XL)

How to use this file:
- Read sections 1–3 together the night before.
- Split section 5 so each of you rehearses half of the questions **out loud**. Then swap.
- Section 6 lists the things we have *not* done yet. Say them plainly if asked; don't bluff.

Deeper material lives in other files:
- `RESEARCH_NOTES.md` covers the literature.
- `DECISIONS.md` gives the reason behind every choice (D1–D21).
- `PROJECT_PLAN.md` is the checklist.

---

## 1. The 30-second pitch (both of us should be able to say this)

> Smartwatch ECGs record only Lead I, one of the twelve views a clinical ECG takes. Heart attacks and muscle thickening are mostly diagnosed from the chest views a watch can't see. We train the same model on 12 leads, on the 6 limb leads, and on Lead I alone, using 21,000 PTB-XL hospital ECGs. Then we measure how much of the ability to tell normal from abnormal survives, and **which diseases account for what's lost**. We report results at a fixed 95% specificity, because a screening device worn by healthy people can't afford many false alarms.

**Research question, word for word:** How much of a 12-lead deep classifier's ability to separate normal from abnormal ECGs is retained when the input is restricted first to the six limb leads and then to Lead I alone, and which diagnostic categories account for the loss?

**Hypothesis:** The loss concentrates in MI and HYP, which depend on the chest leads, and is much smaller for CD, which is about timing.

---

## 2. Where we are (status in one table)

| Phase | Status | Owner | What exists |
|---|---|---|---|
| 0. Data pipeline | ✅ done | H | `Checkpoint1_PTBXL.ipynb`: download, labels, lead sets, features |
| 1. Baselines | ✅ done | H | `02_baselines.ipynb`: logistic regression + gradient-boosted trees × 3 lead sets |
| 2. Evaluation code | ✅ done | H | `evaluate.py` (tested by `test_evaluate.py`, 11 tests), `03_evaluation.ipynb` |
| 3. 1D ResNet (CNN) | ⏳ next | P | architecture and training settings are specified in `PROJECT_PLAN.md` |
| 4. Main experiment | ⏳ | P | 3 lead sets × 3 seeds |
| 5. Noise robustness | ⏳ | P | MIT-BIH noise added to Lead I at 24/12/6 dB |
| 6–8. Errors, figures, slides | ⏳ | Both | |

**Honest framing if she says we're behind:**
- The pipeline and the evaluation protocol are finished and tested.
- Baseline results are on the test set, with confidence intervals.
- The CNN plugs into the same evaluation code, so once it's trained, results come out the same way.
- The remaining risk is CNN training time. Our fallback is one seed per lead set first, then add seeds.

---

## 3. Results so far (memorize the bold numbers)

Test set: 3,207 recordings from 2,791 patients, 57.6% abnormal. The cutoff was chosen on the black-box set for at least 95% specificity and never adjusted on test. Brackets are 95% confidence intervals.

**Gradient-boosted trees (our stronger baseline):**

| Lead set | AUROC | Sensitivity @ 95% spec | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|
| A · 12 leads | **0.935** [0.927, 0.944] | **75.7%** | 81.4% | 84.9% | 76.1% | 83.8% |
| B · 6 limb leads | 0.906 [0.895, 0.916] | 65.9% | 71.1% | 74.3% | 71.2% | 75.3% |
| C · Lead I | **0.852** [0.837, 0.864] | **56.7%** | 58.0% | 70.5% | 58.4% | 73.2% |

(The disease columns are per-category recall: of the recordings carrying that label, the fraction flagged abnormal.)

Logistic regression is weaker but shows the same pattern: AUROC 0.896 → 0.877 → 0.822.

**Recall lost from 12 leads to Lead I (trees), and where the loss happens:**

| Category | Total loss A→C | Loss from removing chest leads (A→B) | Loss from limb leads to Lead I (B→C) |
|---|---|---|---|
| MI | **−23.4 pts** [20.2, 27.0] | −10.4 | −13.1 |
| CD | **−17.7 pts** | −4.9 | **−12.8** |
| STTC | −14.4 pts | −10.5 | −3.8 |
| HYP | **−10.6 pts** (smallest) | −8.6 | −2.0 (not significant) |

**What it means (four sentences to say):**
1. **AUROC understates the cost.** It drops 0.08, but at the same false-alarm rate, the Lead I model misses about a quarter of what the 12-lead model catches (75.7% → 56.7%).
2. **MI loses the most, as predicted.**
3. **HYP loses the least, and CD loses a lot, both against our hypothesis.** We have explanations (section 5, Q14–Q15).
4. **Surprise:** for MI and CD, more of the loss comes from going from six limb leads down to one than from removing the chest leads. Refined story: *"Chest leads matter for STTC and HYP. The other limb leads matter for MI and CD."*

**Literature check:** Saglietto et al. 2024 used a CNN on PTB-XL and reported 93.2 → 85.1 average AUC going from 12 leads to Lead I. Our 0.935 → 0.852 drop is almost the same size.

Figures are in `figures/`:
- `fig1_roc_pr_baselines.png`
- `fig2_per_category_recall_baselines.png` (the headline figure)
- `fig_recall_loss_baselines.png`
- `fig_confusion_baselines.png`

---

## 4. Know the pipeline cold (anyone could be asked)

1. **Data:**
   - PTB-XL 1.0.3, 100 Hz, so each recording is a 1000 × 12 array (10 seconds × 12 leads).
   - 21,799 recordings from 18,869 patients.
   - We dropped 411 recordings with no diagnostic category, leaving **21,388**.
2. **Labels:**
   - 71 SCP statements map to 5 superclasses: NORM, MI, STTC, CD, HYP. A recording can carry several.
   - **Abnormal = carries any of MI/STTC/CD/HYP.** The 445 recordings that carry both NORM and an abnormal code count as abnormal.
3. **Split:**
   - Patient-level 70/15/15, saved once to `data/splits.csv`.
   - The 15% validation part is itself split in two:

   | Split | Recordings | Used for |
   |---|---|---|
   | Train | 14,974 | fitting models |
   | Eyeball (5%) | 1,069 | looking at individual errors by hand |
   | Black box (10%) | 2,138 | numbers only: tuning, cutoff, early stopping |
   | Test (15%) | 3,207 | used once, at the end |

   - The eyeball / black-box idea comes from Andrew Ng's *Machine Learning Yearning*: once you've studied individual recordings in a set, its metrics are biased, so decisions need a set you've never looked at.
4. **Lead sets:** A = all 12 leads. B = I, II, III, aVR, aVL, aVF. C = Lead I. Each set sits inside the one above it.
5. **Baselines:**
   - Hand-made features: 4 rhythm features from Lead I, plus 6 shape features per lead. That gives 76 / 40 / 10 features for A / B / C.
   - Logistic regression (median imputation + scaling) and HistGradientBoosting, both with balanced class weights.
   - Small grid search on black-box AUROC.
6. **CNN (Paco, next):**
   - 1D ResNet: Conv1d(k=15, 64 channels), then 4 residual stages (64 → 128 → 192 → 256, stride 2), then global average pooling, then 6 outputs (abnormal + 5 categories), trained with BCE loss.
   - Adam, learning rate 1e-3, batch size 128, up to 30 epochs, early stopping on black-box AUROC.
   - The only thing that changes between lead sets is the number of input channels.
7. **Evaluation (`evaluate.py`):**
   - Cutoff = the smallest score with at least 95% of black-box normals at or below it. Flag a recording when p > cutoff.
   - Metrics: AUROC, AUPRC, sensitivity, specificity, macro F1, per-category recall.
   - **1,000-resample bootstrap over patients** for confidence intervals.
   - **Paired** bootstrap differences between lead sets.

---

## 5. Questions she's likely to ask, with answers

### About the problem and design

**Q1. Why is this a deep-learning problem?**
The difference between normal and abnormal is mostly in the *shape* of the waveform, and shape is hard to summarize with hand-chosen numbers. A CNN learns shape filters directly from the raw signal. The design is also a clean experiment: hold the network fixed, change only the inputs, and any drop comes from the missing information.

**Q2. Why three lead sets instead of just 12 vs 1?**
The middle set (6 limb leads, which is what a KardiaMobile 6L records) separates two different losses: losing the chest leads (A→B) and losing the other frontal-plane views (B→C). That middle set is where our most interesting finding came from.

**Q3. What's your baseline, and why is it meaningful rather than a straw man?**
It's how ECGs were screened automatically before deep learning: expert-chosen measurements (heart rate, heart-rate variability, QRS width, amplitudes) fed into logistic regression. We added gradient-boosted trees so that a CNN win can't be blamed on a straight-line decision boundary. The baselines also run identically at 12, 6 and 1 lead, so they give a second, independent read on the lead-loss question.

**Q4. Why not a transformer or a pretrained foundation model?**
- On PTB-XL-sized data, 1D ResNets are the strongest established models (around 0.93 AUROC in Strodthoff 2021), and about 90% of the 2021 PhysioNet Challenge entries were CNNs.
- Foundation models match them but don't clearly beat them.
- Our question is about how much information each lead set holds, not about which architecture wins. A well-understood CNN keeps the comparison clean.
- A pretrained 12-lead model would also confound the lead comparison.

### About the data and leakage

**Q5. You changed the split from the proposal. Why, and what did it cost?**
- We moved from the official folds (1–8 / 9 / 10) to a patient-level 70/15/15 split so we'd have enough validation data to divide it into an eyeball set and a black-box set.
- **The cost:** folds 9 and 10 had an extra round of human label review, so our test labels may be slightly noisier.
- We documented this (DECISIONS.md, D4) and will put it on the limitations slide.
- *(Bring this up ourselves before she does. See open question Q3 in DECISIONS.md.)*

**Q6. How do you know there's no leakage?**
- The split groups recordings by **patient**, because 2,111 patients have more than one recording. The code *asserts* that no patient appears in two splits.
- Imputation, scaling and per-lead normalization statistics are fit on train only.
- Every tuning decision, the cutoff, and CNN early stopping use the black-box set.
- The test set was scored once, and the baselines are now frozen (D20).
- Predictions save their split label, and the evaluation code checks it against `splits.csv`.

**Q7. Class imbalance?**
- 57.6% of recordings are abnormal, the same in every split because we stratified on the full label combination.
- We use balanced class weights.
- We report AUPRC and a fixed-specificity operating point, which doesn't depend on a 0.5 threshold.

**Q8. Is PTB-XL Lead I the same as an Apple Watch?**
No. PTB-XL was recorded with gel electrodes on still, lying-down hospital patients from 1989 to 1996. A watch uses dry electrodes on a moving person, and the Apple Watch is FDA-cleared only for atrial fibrillation. So our Lead I numbers are an **optimistic ceiling**, which is why Phase 5 adds real recorded noise. The population is also older and sicker than watch users, so prevalence-dependent numbers like precision won't transfer.

**Q9. Why 100 Hz and not 500?**
Strodthoff et al. found no meaningful gain from 500 Hz for CNNs on these tasks. It's 5× less data, and it's closer to what wearables keep after filtering.

### About evaluation

**Q10. Why 95% specificity instead of accuracy or a 0.5 threshold?**
- A screening device runs on millions of mostly healthy people, so even a small false-alarm rate swamps follow-up care.
- Fixing specificity gives every model the same operating point, so sensitivities can be compared directly.
- 0.5 is arbitrary, and accuracy depends on how common disease is in the data.

**Q11. Where does the cutoff come from? Couldn't you have tuned it on test?**
It's chosen on the black-box set and applied unchanged to test. That's why test specificity varies a little (94.8%–97.0%) instead of being exactly 95%, which is honest behavior.

**Q12. Why bootstrap over patients and not recordings?**
Recordings from the same patient are correlated (3,207 test recordings from 2,791 patients). Resampling recordings would treat them as independent and make the intervals too narrow.

**Q13. How do you know a difference between lead sets is real?**
- We use **paired** bootstrap differences: each resample scores both models on the same patients, and we check whether the 95% interval of the *difference* excludes 0. Comparing two separate intervals would be too conservative, because the models' errors are correlated.
- All overall gaps are significant.
- Not significant:
  - HYP B→C for trees: −2.0 [−1.5, +5.9].
  - MI A→B for logistic regression.
- Caveat: we ran many comparisons without correcting for multiple testing, so borderline results deserve caution.

### About the surprises

**Q14. Why does HYP lose the *least*? Your hypothesis said it would lose the most.**
Three reasons:
1. **Co-labels.** 80% of test HYP recordings also carry another abnormal label (versus about 55% for the other categories). Per-category recall counts a HYP recording as caught if it's flagged abnormal *for any reason*, so HYP is often caught through its ST/T or conduction changes.
2. Voltage criteria for hypertrophy are weak even with 12 leads, so there's less to lose.
3. Lead I still carries hypertrophy information: about 88% AUC in Saglietto, versus 94% with 12 leads.

Next step: report recall for HYP-only recordings.

**Q15. Why does CD lose so much? Isn't it a timing problem?**
- PTB-XL's CD category mixes two kinds of findings:
  - Pure timing findings (complete bundle branch blocks, AV block), which survive Lead I.
  - Findings that need a specific view: incomplete RBBB (defined by a pattern in V1) and fascicular blocks (defined by the heart's electrical axis, which needs II, III and aVF).
- In Saglietto, incomplete RBBB lost 22 points and left fascicular block lost 19 points with Lead I alone.
- That fits our result that most of the CD loss is B→C (12.8 points) rather than A→B (4.9 points).
- Refined hypothesis: complete blocks survive and incomplete or fascicular ones don't. We can test it by sub-label.

**Q16. Why does MI lose so much between the limb leads and Lead I?**
Inferior MI, the most common kind, shows in leads II, III and aVF, which are limb leads but not Lead I. Our example error in the eyeball set was exactly that: an inferior MI. Saglietto's biggest single-lead drop was inferior MI (95.2 → 71.6). Next step: split MI by location (inferior, anterior, lateral).

**Q17. Your 12-lead trees already get 0.935. Won't the CNN be pointless?**
Published benchmarks aren't directly comparable, because they use a different task (5-class macro AUROC versus our binary target). The real test is the CNN against our trees on identical splits. In Strodthoff, feature-based models were about 0.05 behind CNNs. The more interesting question is whether the CNN gets *more out of Lead I* than hand-made features do. If it doesn't, that's a finding too.

### About process and teamwork

**Q18. Who did what?**
- **Hudson:** data pipeline, split, baselines, evaluation code and tests.
- **Paco:** CNN, the main experiment across 3 seeds, noise robustness, and leading the error analysis.
- **Both:** design, interpretation, slides.

*(Paco should be ready to walk through the CNN plan in section 4, item 6, in his own words.)*

**Q19. How do you know your evaluation code is correct?**
- `test_evaluate.py` has 11 tests on made-up data where the right answer is known.
- We also deliberately put 5 realistic bugs into the code, one at a time, to check that the tests catch them. On the first try they caught only 3, so we tightened two tests. Now all 5 are caught.
- Every number on the slides goes through this one file.

**Q20. Is it reproducible?**
- Package versions are pinned in `requirements.txt`.
- The split is fixed in `data/splits.csv`, with seed 42.
- Every decision is logged in `DECISIONS.md`.
- The code is on GitHub. The raw data isn't, because it's 1.6 GB and PhysioNet hosts it; `Checkpoint1_PTBXL.ipynb` re-downloads it.
- The CNN will run 3 seeds and report mean ± standard deviation.

**Q21. What's the biggest risk to finishing?**
CNN training time across 9 runs. Mitigations:
- Train on Colab with a GPU.
- At 100 Hz, one run should take minutes, not hours.
- The must-have core is 3 lead sets × 1 seed plus the per-category chart. Extra seeds and the noise experiment come after.

---

## 6. Things we haven't done yet (say so if asked)

- No CNN results yet. All numbers so far are from the baselines.
- No noise-robustness experiment yet.
- No breakdown by sub-diagnosis yet (MI by location, complete vs incomplete blocks). We'll do this if time allows; it would test the refined hypothesis.
- No correction for multiple comparisons.
- The test set lost the extra label review of the official folds (Q5).

## 7. Questions *we* could ask her

1. Is changing from the official folds to our 70/15/15 eyeball/black-box split acceptable, as long as we document it?
2. Would she rather see the hypothesis refined (sub-diagnosis breakdown), or the noise experiment, if we only have time for one?
3. For the final presentation, should baseline results get their own slide, or be folded into the results slide next to the CNN?

## 8. Night-before checklist

- [ ] Both of us can give the 30-second pitch without notes
- [ ] Both of us can say the three headline numbers: 0.935 → 0.852 AUROC; 75.7% → 56.7% sensitivity; MI loses the most and HYP the least
- [ ] Paco can explain the eyeball / black-box split and the patient-level bootstrap
- [ ] Hudson can explain the CNN architecture and training plan
- [ ] `figures/fig2_per_category_recall_baselines.png` is open and ready to show
- [ ] GitHub repo link ready to share
