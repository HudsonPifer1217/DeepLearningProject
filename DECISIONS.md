# Decision Log

Every choice made while building this project, with the reason behind it. New decisions go at the bottom of the "Decisions" list. When a decision is reversed, mark the old entry **Superseded by D#** instead of deleting it.

**Who decided:** **User** = Hudson (or the team), **Claude** = made by Claude while implementing; any of these can be revisited. **Proposal** = taken from the project proposal.

**Format:** each entry gives the date, who decided, the decision, why, and what we considered instead.

---

## Open questions (need a human answer)

- [ ] **Q1.** Is a 5% eyeball / 10% black-box division of validation what the course means? (See D5.) If not, change `chunk == 14` / `chunk < 17` in `02_baselines.ipynb`, **delete `data/splits.csv`**, and rerun.
- [ ] **Q2.** Should the 4 rhythm features stay in the Lead-I baseline? (See D9.) They come from Lead I, so they're legitimate for configuration C, but keeping them means every configuration gets full timing information.
- [ ] **Q3.** Do we tell the instructor we switched from the official folds to a random 70/15/15 split? (See D4.) The cost is that the test set no longer has the second round of label review that folds 9–10 had.

- [ ] **Q4.** Should we filter Lead I to 0.2–25 Hz in an extra "watch-like" configuration? Mayo's Apple Watch study (Attia et al., 2022) did this to mimic the watch signal. It would be cheap to add. See `RESEARCH_NOTES.md`.
- [ ] **Q5.** Should we refine the hypothesis now that the research is in? The literature suggests:
  - MI loss depends on location (inferior MI collapses with Lead I; lateral MI barely changes).
  - CD splits two ways: complete bundle branch blocks survive Lead I, while incomplete RBBB and left fascicular blocks don't.

  Testing that would need a per-statement breakdown (for example IMI vs AMI, CRBBB vs IRBBB), not just the 5 categories.

---

## Decisions

### D1: Rewrite the downloader with timeouts and connection reuse
- **Date:** 2026-10-21 · **By:** Claude · **Where:** `Checkpoint1_PTBXL.ipynb`, cell 3
- **Decision:** Replace `urllib.request.urlretrieve` with a `requests.Session` per thread. Each request has a (10 s connect, 30 s read) timeout and up to 5 retries, waiting longer before each one (1, 2, 4, 8 s). Use 16 workers and print progress.
- **Why:** `urlretrieve` has no timeout, so a single stalled connection hung its thread forever. The original run froze after 4,096 of 21,799 recordings. The fixed version downloaded the remaining 35,406 files in about 6 minutes.
- **Alternatives:** Downloading PhysioNet's zip archive (about 1.7 GB, mostly the 500 Hz data we don't use); `wget -r` (no progress inside the notebook).

### D2: Add `pyarrow` to the venv and pin package versions
- **Date:** 2026-10-21 · **By:** Claude · **Where:** `.venv`, `requirements.txt`
- **Decision:** Install `pyarrow`, which the parquet feature cache requires. Write `requirements.txt` with exact versions.
- **Why:** Without `pyarrow`, the feature cell would have crashed. Pinned versions let Paco and the grader reproduce the environment.

### D3: Exclude recordings that have no diagnostic category
- **Date:** carried over from Checkpoint 1 · **By:** Team
- **Decision:** Drop the 411 recordings whose statements don't map to any of NORM, MI, STTC, CD or HYP. 21,388 remain.
- **Why:** There's no ground truth to train or evaluate them against.

### D4: Replace the official folds with a 70 / 15 / 15 split
- **Date:** 2026-10-22 · **By:** User · **Supersedes:** Proposal (folds 1–8 / 9 / 10)
- **Decision:** 70% train, 15% validation, 15% test.
- **Why:** User's choice for this project.
- **Trade-off to mention on the slides:** PTB-XL's folds 9 and 10 had an extra round of human label review. A random test set doesn't have that guarantee, so some test labels may be noisier.

### D5: Divide validation into eyeball (5%) and black box (10%)
- **Date:** 2026-10-22 · **By:** User (the division) / Claude (the 5:10 sizes) · **Where:** `02_baselines.ipynb`, section 2
- **Decision:** Eyeball set = 1,069 recordings, used to look at errors by hand. Black-box set = 2,138 recordings, used only through its numbers: tuning, the cutoff, model selection.
- **Why:** This follows Andrew Ng's *Machine Learning Yearning*. Once we've studied individual recordings in a set, our choices are shaped by them and its metrics are biased, so decisions need a set we never look at. I made the black-box set larger because it drives decisions, and a 5% eyeball set still has 100+ errors to study.
- **Alternatives:** 7.5 / 7.5.

### D6: Split by patient, balanced on the category combination
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `02_baselines.ipynb`, section 2
- **Decision:** `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` with `groups = patient_id`, stratified on the 5-category label combination (for example `"01100"`). Its 20 chunks of about 5% each are assigned 14 → train, 1 → eyeball, 2 → black box, 3 → test.
- **Why:** 2,111 patients have several recordings. Splitting by recording would let a model memorize a patient (leakage). Balancing on the full combination keeps the multi-label category mix nearly identical in every split (57.6% abnormal in each). The code asserts that no patient appears in two splits.
- **Alternatives:** `GroupShuffleSplit` (no balancing); combining the existing `strat_fold` values (only gives 10% steps, so 70/15/15 isn't possible).

### D7: The split is saved once and reused by every notebook
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `data/splits.csv`
- **Decision:** The first run writes `data/splits.csv` (`ecg_id`, `patient_id`, `split`). Every later run, and every later notebook, loads it instead of rebuilding it.
- **Why:** The baselines and the CNN must be evaluated on exactly the same recordings. Loading the file also protects against a scikit-learn upgrade quietly changing the split.

### D8: Phase 1 baseline models and how they're set up
- **Date:** 2026-10-22 · **By:** Proposal (the models) / Claude (the details) · **Where:** `02_baselines.ipynb`, section 4
- **Decision:**
  - Logistic regression: median imputation, then `StandardScaler`, then `LogisticRegression(class_weight="balanced", max_iter=2000)`.
  - Gradient-boosted trees: `HistGradientBoostingClassifier(class_weight="balanced", max_iter=500, random_state=42)`, which handles missing values natively.
- **Why:** Balanced class weights offset the 58/42 class imbalance. Imputation and scaling are fit on train only, inside the pipeline, so nothing leaks. The tree model's built-in early stopping holds out part of **train**, not the black-box set.

### D9: Keep the rhythm features in every configuration's baseline
- **Date:** carried over from Checkpoint 1 · **By:** Team · **Status:** see Q2
- **Decision:** All three configurations get `hr_bpm`, `rr_sdnn_ms`, `rr_rmssd_ms` and `n_beats`. They're computed from Lead I's R-peaks.
- **Why:** A watch records Lead I, so those features are legitimately available in configuration C.
- **Consequence:** Timing-based information is equal across A, B and C. Differences between configurations come only from the shape features.

### D10: Hyperparameter search
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `02_baselines.ipynb`, section 4
- **Decision:**
  - Logistic regression: `C` ∈ {0.001, 0.01, 0.1, 1, 10}.
  - Gradient-boosted trees: learning rate ∈ {0.05, 0.1} × max depth ∈ {3, 6, None}.
  - The best setting is chosen by black-box AUROC, separately for each configuration.
- **Why:** Small grids are enough here. The results show tuning changes AUROC by less than 0.005, so the findings don't hinge on hyperparameters.

### D11: Save test predictions now, look at them only in Phase 2
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `results/baseline_{model}_{cfg}.npz`
- **Decision:** Each file stores predictions for every recording, along with its split label. No test metric is computed or displayed in Phase 1.
- **Why:** Phase 2 can then evaluate without retraining, and no Phase 1 decision can be influenced by the test set.

### D12: The 95%-specificity cutoff is chosen on the black-box set
- **Date:** 2026-10-22 · **By:** Proposal (95% specificity) / Claude (which set) · **Where:** `cutoff_at_specificity` in `02_baselines.ipynb`
- **Decision:** The cutoff is the 95th percentile of model scores among *normal* recordings in the black-box set. It's then applied unchanged to the eyeball set (now) and the test set (Phase 2).
- **Why:** Choosing the cutoff is a modeling decision, so per D5 it belongs to the black-box set.

### D13: The eyeball-set per-category recall is a preview, not a result
- **Date:** 2026-10-22 · **By:** Claude
- **Decision:** The Phase 1 per-category recall chart is labeled as a preview. The formal comparison, on the test set with confidence intervals, happens in Phase 2.
- **Why:** The eyeball set is small (about 130 HYP and 275 MI recordings), so each recall number is uncertain by roughly ±5–8 points. It has also been looked at, so it's biased.

### D14: Example-error plot shows limb leads as well as chest leads
- **Date:** 2026-10-22 · **By:** Claude
- **Decision:** The plot shows leads I, II, III, aVF, V2 and V3, not just I and V1–V4.
- **Why:** The first example found was an *inferior* MI, which is visible in II, III and aVF (limb leads), not the chest leads. Plotting only chest leads would have implied something false. This also raised the point that MI location may matter for the hypothesis.

### D15: The CNN must load the new split and recompute normalization
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `PROJECT_PLAN.md`, Phase 3
- **Decision:** The CNN loads `data/splits.csv` and recomputes each lead's mean and standard deviation on the new train split. Early stopping and tuning use the black-box set.
- **Why:** `to_model_input` in Checkpoint 1 still normalizes with statistics from folds 1–8. Those folds overlap the new eyeball, black-box and test sets, which would leak information.

### D16: Shared code lives in `common.py` and `evaluate.py`
- **Date:** 2026-10-22 · **By:** Claude
- **Decision:**
  - `common.py` holds the label construction, the split loading and the lead-set definitions.
  - `evaluate.py` holds the evaluation protocol.
  - New notebooks import both instead of copying code.
  - `02_baselines.ipynb` keeps its own inline copy, so it still runs as it did.
- **Why:** If the baselines and the CNN each had their own evaluation code, any difference in their scores could come from the code, not the models.

### D17: Cutoff rule is "at least 95% specificity", and a recording is flagged when `p > cutoff`
- **Date:** 2026-10-22 · **By:** Claude · **Supersedes:** the quantile rule in D12 (for Phase 2 onward)
- **Decision:** cutoff = the ⌈0.95·n⌉-th smallest score among the normal recordings in the black-box set. Flag a recording as abnormal when its score is strictly greater than the cutoff.
- **Why:** The tree models produce tied scores. With `np.quantile` and `>=` (the Phase 1 preview rule), tied normals at the cutoff all get flagged, which can push specificity below 95%. The new rule guarantees at least 95% on the black-box set, and a test checks the tie case. The Phase 1 eyeball preview used the old rule, but it's labeled as a preview, so I left it.

### D18: Confidence intervals come from resampling patients
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `bootstrap_indices` in `evaluate.py`
- **Decision:** Each of the 1,000 resamples draws test *patients* with replacement and keeps all of each patient's recordings. The cutoff is held fixed across resamples.
- **Why:**
  - Recordings from the same patient are correlated (the test set has 3,207 recordings from 2,791 patients). Resampling recordings would make the intervals too narrow.
  - The cutoff is fixed because it was chosen on independent data (the black-box set), so re-estimating it inside each resample would be wrong.

### D19: Lead-set comparisons use paired bootstrap differences
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `paired_difference` in `evaluate.py`
- **Decision:** Compare A − B, A − C and B − C by computing both models on the *same* resample each time, then taking the 2.5th and 97.5th percentiles of the difference. A difference is called significant when its confidence interval excludes 0.
- **Why:** The research question is about how much is *lost*. Comparing two overlapping intervals is too conservative, because the two models are scored on identical recordings and their errors are correlated. Pairing accounts for that.
- **Known limitation:** There are many comparisons (2 models × 3 pairs × 6 metrics) and no multiple-comparison correction. Treat borderline "significant" results with care.

### D20: The six baselines are frozen after the test evaluation
- **Date:** 2026-10-22 · **By:** Claude
- **Decision:** `03_evaluation.ipynb` scored the six Phase 1 baselines on the test set once. No baseline setting may be changed because of these results.
- **Why:** Re-tuning after seeing test results would leak the test set into model selection. If a baseline ever has to change (for example Q2, removing the rhythm features), the change must be justified on the black-box set and reported as a deviation.

### D21: Added a test suite and checked that the tests catch bugs
- **Date:** 2026-10-22 · **By:** Claude · **Where:** `test_evaluate.py` (11 tests); `pytest` added to `requirements.txt`
- **Decision:** Test `evaluate.py` on made-up data where the right answer is known. Then inject 5 realistic bugs into a copy of the code, one at a time, and confirm each makes a test fail. The bugs: cutoff off-by-one, `>=` vs `>`, a broken patient bootstrap, a broken paired difference, and wrong per-category recall.
- **Why:** The first version of the tests missed two of those bugs, the off-by-one and the tie rule, so I tightened two tests. Every result in the presentation flows through this code.

### D22: What goes in the GitHub repo
- **Date:** 2026-10-22 · **By:** User (create the repo) / Claude (what's included) · **Where:** `.gitignore`, `git@github.com:HudsonPifer1217/DeepLearningProject.git`
- **Decision:** `Project1/` is its own git repository, separate from the home-folder repo it sits inside. The repo includes the notebooks, `.py` files, md files, `figures/`, `results/` and `data/splits.csv`. It excludes `data/ptb-xl/` (the 1.6 GB raw data and its caches), `.venv/` and Python caches.
- **Why:** PhysioNet already hosts the raw data, and `Checkpoint1_PTBXL.ipynb` re-downloads it. `splits.csv` (about 0.4 MB) is committed so Paco's CNN trains and tests on exactly the same recordings. The prediction files (about 7 MB) are committed so `03_evaluation.ipynb` runs without retraining the baselines.

### D23: Every notebook runs on Colab from a clone of this repo
- **Date:** 2026-10-23 · **By:** Paco / Claude · **Where:** first code cell of each notebook, `fetch_ptbxl` in `common.py`
- **Decision:** Each notebook starts with a cell that, on Colab only, clones this repo into `/content/DeepLearningProject` and works from inside it. Locally the cell does nothing. `fetch_ptbxl` downloads single PTB-XL files on demand, so `02` and `03` fetch only the two label CSVs (plus one record for the example-error plot) instead of the full 1.6 GB.
- **Why:** The instructor runs the notebooks on Colab, where every notebook starts on a fresh machine without the repo, the `.py` modules or the data. Before this, only `Checkpoint1_PTBXL.ipynb` could run there, and it saved to `/content/ptb-xl`, a path the other notebooks never read.

### D24: The hand-made feature table is committed as `data/handcrafted_features.parquet`
- **Date:** 2026-10-23 · **By:** Paco / Claude · **Supersedes:** the cache location `data/ptb-xl/handcrafted_features.parquet` (part of D22)
- **Decision:** Move the feature cache out of the ignored raw-data folder and commit it (about 7 MB).
- **Why:** Rebuilding it needs the full download plus feature extraction over 21,799 recordings. With it committed, `02_baselines.ipynb` runs on Colab in about a minute. Checked: retraining the 12-lead trees on this file reproduces the saved black-box AUROC (0.927).

### D25: Checkpoint 2 additions to `02_baselines.ipynb`
- **Date:** 2026-10-23 · **By:** Paco / Claude
- **Decision:** Added (1) a preprocessing-pipeline table, including what changed since Checkpoint 1, (2) the `X` and `y` shapes per split and lead set, and (3) train vs black-box AUROC for every model, to check for over- and underfitting.
- **Why:** The Checkpoint 2 brief asks for each of these directly. None of them changes a model, so the baselines stay frozen (D20).

### D26: The tree baselines vary slightly across machines; the committed predictions are the reference
- **Date:** 2026-10-23 · **By:** Paco / Claude
- **Finding:** With the pinned packages and identical features, logistic regression reproduces Hudson's predictions exactly on Windows, but `HistGradientBoostingClassifier` does not: its results depend on the CPU and threading. Re-running `02_baselines.ipynb` on Windows moved test AUROC by at most 0.0003, sensitivity by at most 0.5 points and per-category recall by at most about 1 point. Black-box AUROC for the B and C trees showed as 0.905 / 0.846 instead of 0.904 / 0.845. No conclusion changes.
- **Decision:** The prediction files in `results/` from Hudson's run stay the frozen reference (D20), and every reported number comes from them through `03_evaluation.ipynb`, which is deterministic. Re-running `02` (on Colab, say) overwrites them locally with near-identical predictions; don't commit those.
- **Also:** The feature table was rebuilt with the pinned packages before being committed (D24). An older copy built with different package versions differed in 34 of about 1.6 million values, enough to change a few predictions.
