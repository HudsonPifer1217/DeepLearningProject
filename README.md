# Lead I or Twelve?

How much of the ability to tell normal from abnormal ECGs survives when a model sees 12 leads, the 6 limb leads, or Lead I alone (the smartwatch lead)? Built on PTB-XL 1.0.3.

## Run on Colab

Open a notebook and choose **Runtime → Run all**. The first cell clones this repo and installs what's missing, so nothing has to be uploaded.

| Notebook | What it does | Colab |
|---|---|---|
| `Checkpoint1_PTBXL.ipynb` | Downloads PTB-XL (about 6 minutes), builds labels, lead sets and features | [open](https://colab.research.google.com/github/HudsonPifer1217/DeepLearningProject/blob/main/Checkpoint1_PTBXL.ipynb) |
| `02_baselines.ipynb` | Preprocessing, split, logistic regression and gradient-boosted trees | [open](https://colab.research.google.com/github/HudsonPifer1217/DeepLearningProject/blob/main/02_baselines.ipynb) |
| `03_evaluation.ipynb` | Test-set results with confidence intervals and figures | [open](https://colab.research.google.com/github/HudsonPifer1217/DeepLearningProject/blob/main/03_evaluation.ipynb) |

`02` and `03` don't need `Checkpoint1` to run first: the feature table, the split and the baseline predictions are committed in `data/` and `results/`.

## Run locally

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
```

Running `Checkpoint1_PTBXL.ipynb` once downloads the raw data to `data/ptb-xl/`. `pytest test_evaluate.py` checks the evaluation code.

See `PROJECT_PLAN.md` for the plan, `DECISIONS.md` for why things are the way they are.
