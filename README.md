# ML Algorithms Comparison

A reproducible comparison of **Classification**, **Clustering**, and **Boosting**
algorithms, with an optional **Regression** module, built on three real tabular
datasets. Every number in this README was produced by running the code in this
repository — nothing here is hand-typed or simulated.

## 1. Project overview

This project trains and evaluates multiple scikit-learn algorithms per task on
fixed, leakage-safe pipelines and a fixed random seed, then reports honest,
executed results: no cherry-picking, no invented metrics, and no claim that any
one family of algorithms (e.g. boosting) is unconditionally best.

## 2. Problem statement and objectives

- **Module A — Classification**: predict whether a user purchases a product
  from demographic and salary data.
- **Module B — Clustering**: discover customer segments from spending behavior,
  with no predefined labels.
- **Module C — Boosting**: compare boosting ensembles against the standard
  classifiers from Module A, on the identical split and metrics.
- **Module D — Regression (optional)**: predict medical insurance charges from
  demographic and health data.

## 3. Datasets and target columns

| File | Rows (after de-dup) | Target | Module |
|---|---|---|---|
| `data/Social_Network_Ads.csv` | 400 | `Purchased` (binary: 0/1) | A, C |
| `data/Mall_Customers.csv` | 200 | none (unsupervised) | B |
| `data/insurance_pre.csv` | 1337 (1 exact duplicate row dropped) | `charges` (continuous) | D |

All three datasets had **zero missing values**. `insurance_pre.csv` contained one
exact duplicate row, which is dropped during loading (`src/data_loader.py`).
`Purchased` is moderately imbalanced (257 "no" vs. 143 "yes"), so the
train/test split is stratified and both precision/recall are reported
alongside accuracy.

## 4. Classification, clustering, and boosting, briefly

- **Classification** learns a mapping from input features to a discrete label
  using labeled training examples (here: will a user purchase, yes/no).
- **Clustering** groups unlabeled observations by similarity, with no target
  to predict — the "right answer" is not known in advance and must be judged
  by internal metrics like silhouette score and by inspecting segment
  characteristics.
- **Boosting** builds an ensemble of weak learners (typically shallow trees)
  sequentially, where each new learner focuses on the errors of the ensemble
  so far. It is a different *training strategy* for classifiers, not a
  separate problem type — which is why Module C reuses Module A's exact
  target, split, and preprocessing to make the comparison fair.

## 5. Algorithms implemented

**Classification / Boosting (Social_Network_Ads.csv, target `Purchased`)**
- Logistic Regression — linear decision boundary over scaled features.
- Decision Tree — axis-aligned recursive splits, no scaling required.
- Random Forest — bagged ensemble of decision trees.
- SVM (RBF kernel) — maximum-margin classifier; wrapped in
  `CalibratedClassifierCV` to obtain probability estimates for ROC-AUC.
- KNN — majority vote of the 5 nearest neighbors in scaled feature space.
- AdaBoost — reweights misclassified samples each boosting round.
- Gradient Boosting — fits each new tree to the residual gradient of the loss.
- HistGradientBoosting — histogram-binned gradient boosting (LightGBM-style),
  scikit-learn's fastest boosting implementation.

**Clustering (Mall_Customers.csv, features `Annual Income (k$)` and
`Spending Score (1-100)`)**
- K-Means — partitions data into K spherical clusters by minimizing
  within-cluster variance; K chosen via silhouette score across K=2..10.
- Agglomerative (Hierarchical) Clustering — bottom-up merging of the closest
  clusters, cut at the same K as K-Means for a fair comparison.
- DBSCAN — density-based clustering that does not require K and can mark
  low-density points as noise; `eps` estimated from the 90th percentile of
  5th-nearest-neighbor distances.

**Regression, optional (insurance_pre.csv, target `charges`)**
- Linear Regression, Decision Tree Regressor, Random Forest Regressor,
  Gradient Boosting Regressor.

## 6. Folder structure

```text
ml-algorithms-comparison/
├── data/                       # raw CSVs (gitignored; see data/README.md)
├── notebooks/                  # runnable notebooks mirroring src/ + scripts/
├── src/                        # shared library code (no duplicate logic)
│   ├── config.py                # paths, seed, feature/target lists
│   ├── data_loader.py            # load + schema validation
│   ├── preprocessing.py          # leakage-safe pipelines (fit on train only)
│   ├── exploratory_analysis.py   # dataset overview plot
│   ├── classification.py         # Module A
│   ├── clustering.py             # Module B
│   ├── boosting.py               # Module C
│   ├── regression.py             # Module D (optional)
│   ├── evaluation.py             # shared metric functions, incl. edge cases
│   └── visualization.py          # shared plotting functions
├── results/
│   ├── metrics/                  # *.csv — one row per model
│   ├── predictions/               # *.csv — y_true + per-model y_pred
│   ├── cluster_assignments/       # customer_clusters.csv
│   └── plots/                     # *.png
├── models/                     # best_*.joblib (gitignored)
├── tests/                      # pytest suite (20 tests, see §10)
├── scripts/                    # CLI entry points
└── requirements.txt
```

## 7. Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place `Social_Network_Ads.csv`, `Mall_Customers.csv`, and `insurance_pre.csv`
in `data/` (see `data/README.md`) before running anything.

## 8. Running experiments individually

```bash
python scripts/run_classification.py
python scripts/run_clustering.py
python scripts/run_boosting.py     # also reruns classification for the comparison table
python scripts/run_regression.py   # optional module
```

## 9. Running the complete project

```bash
python scripts/run_all.py
python -m pytest
```

Equivalently, the notebooks in `notebooks/` run the same `src/` code and embed
the plots inline; they were executed top-to-bottom with `jupyter nbconvert
--execute` as part of building this project.

## 10. Evaluation metrics

- **Accuracy** — fraction of correct predictions. Misleading alone under class
  imbalance, so reported alongside precision/recall/F1.
- **Precision** — of predicted positives, how many were actually positive.
- **Recall** — of actual positives, how many were correctly identified.
- **F1-score** — harmonic mean of precision and recall; used as the model
  **selection metric** (see §11) because it balances both error types on this
  moderately imbalanced target.
- **ROC-AUC** — probability the model ranks a random positive above a random
  negative; undefined/`None` when only one class is present or probabilities
  are unavailable (handled explicitly in `src/evaluation.py`).
- **Confusion matrix** — raw counts of TP/FP/TN/FN per model.
- **Silhouette score** — how well-separated clusters are (-1 to 1); undefined
  when fewer than 2 valid (non-noise) clusters exist, handled explicitly
  rather than raised as an error.
- **MAE / RMSE / R²** — mean absolute error, root mean squared error, and
  variance explained, for the regression module.

## 11. Actual results

All metrics below are the literal contents of the generated CSVs in
`results/metrics/` from the most recent `python scripts/run_all.py` run.

### Module A — Classification (`results/metrics/classification_metrics.csv`)

Stratified 80/20 train/test split, seed=42. Model selection uses **mean 5-fold
cross-validated F1-score computed on the training set only** (`cv_f1_mean`),
not the test-set F1 — so the "best" pick is not biased by the one held-out
test split.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (selection) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8125 | 0.8182 | 0.6207 | 0.7059 | 0.9053 | 0.7606 |
| Decision Tree | 0.8625 | 0.7647 | 0.8966 | 0.8254 | 0.8698 | 0.8115 |
| Random Forest | 0.9000 | 0.8387 | 0.8966 | 0.8667 | 0.9266 | 0.8538 |
| **SVM (calibrated)** | 0.9125 | 0.8667 | 0.8966 | 0.8814 | 0.9473 | 0.8696 |
| **KNN** | 0.9000 | 0.8387 | 0.8966 | 0.8667 | 0.9304 | **0.8739** |

**Best model: KNN**, selected by highest mean CV F1-score (0.8739). Note SVM
scored higher on the held-out *test* F1 (0.8814 vs. 0.8667) — this is exactly
why selection is done via cross-validation rather than by peeking at the test
set and picking whichever model happens to win there.

Plots: [classification_comparison.png](results/plots/classification_comparison.png),
[confusion_matrices.png](results/plots/confusion_matrices.png)

### Module B — Clustering (`results/metrics/clustering_metrics.csv`)

Features scaled with `StandardScaler`; K chosen by silhouette score across
K=2..10 (elbow plot included for context, but K-selection is silhouette-driven
since the elbow is visually ambiguous past K=5).

| Model | n_clusters | n_noise_points | Silhouette score |
|---|---|---|---|
| K-Means (K=5) | 5 | 0 | **0.5547** |
| Agglomerative (K=5) | 5 | 0 | 0.5538 |
| DBSCAN (eps=0.468, min_samples=5) | 2 | 11 | 0.3963 |

K-Means and Agglomerative agree closely and both clearly outperform DBSCAN on
silhouette score here. **DBSCAN does not reproduce the well-known 5-blob
structure of this dataset** — it collapses most points into 2 density-connected
clusters and marks 11 points as noise, which is an honest limitation of a
density-based approach on data whose clusters are closer in some regions than
others, not a flaw in the implementation.

**K-Means segment profiles** (mean feature values per cluster):

| Cluster | Annual Income (k$) | Spending Score | Age | Count | Interpretation |
|---|---|---|---|---|---|
| 0 | 55.3 | 49.5 | 42.7 | 81 | Mid income, mid spending — the "average" majority segment |
| 1 | 86.5 | 82.1 | 32.7 | 39 | High income, high spending — ideal target segment |
| 2 | 25.7 | 79.4 | 25.3 | 22 | Low income, high spending — young, impulsive spenders |
| 3 | 88.2 | 17.1 | 41.1 | 35 | High income, low spending — affluent but price-conservative |
| 4 | 26.3 | 20.9 | 45.2 | 23 | Low income, low spending — budget-constrained segment |

Plots: [clustering_elbow.png](results/plots/clustering_elbow.png),
[customer_clusters.png](results/plots/customer_clusters.png)

### Module C — Boosting (`results/metrics/boosting_metrics.csv`)

Same target, same stratified 80/20 split (seed=42), same selection
methodology as Module A.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (selection) |
|---|---|---|---|---|---|---|
| **AdaBoost** | 0.8625 | 0.8214 | 0.7931 | 0.8070 | 0.9378 | **0.8654** |
| Gradient Boosting | 0.9000 | 0.8387 | 0.8966 | 0.8667 | 0.9513 | 0.8371 |
| HistGradientBoosting | 0.9000 | 0.8387 | 0.8966 | 0.8667 | 0.9375 | 0.8443 |

**Best boosting model: AdaBoost**, selected by CV F1 (0.8654) despite having
the lowest test-set accuracy/F1 of the three boosting models — on this small,
400-row dataset, Gradient Boosting and HistGradientBoosting generalize more
inconsistently across folds, and AdaBoost's simplicity worked in its favor
during cross-validation. Gradient Boosting does post the single highest
ROC-AUC of any model trained in this project (0.9513).

**Boosting vs. standard classifiers — combined comparison, sorted by test F1:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM (calibrated) | 0.9125 | 0.8667 | 0.8966 | **0.8814** | 0.9473 |
| Random Forest | 0.9000 | 0.8387 | 0.8966 | 0.8667 | 0.9266 |
| Gradient Boosting | 0.9000 | 0.8387 | 0.8966 | 0.8667 | **0.9513** |
| KNN | 0.9000 | 0.8387 | 0.8966 | 0.8667 | 0.9304 |
| HistGradientBoosting | 0.9000 | 0.8387 | 0.8966 | 0.8667 | 0.9375 |
| Decision Tree | 0.8625 | 0.7647 | 0.8966 | 0.8254 | 0.8698 |
| AdaBoost | 0.8625 | 0.8214 | 0.7931 | 0.8070 | 0.9378 |
| Logistic Regression | 0.8125 | 0.8182 | 0.6207 | 0.7059 | 0.9053 |

**Boosting is not uniformly superior here**: on this dataset, the classic SVM
and Random Forest match or beat every boosting model on test F1, and only
Gradient Boosting edges out the non-boosted field on ROC-AUC. With only 400
rows and 2 real numeric features plus one binary one, there is limited signal
for boosting's sequential error-correction to exploit beyond what Random
Forest's bagging or SVM's margin maximization already capture. Boosting's
advantages (robustness to feature interactions, strong performance on larger
and more complex tabular data) are a better fit for datasets with higher
dimensionality — which is one legitimate limitation of this comparison, not a
deficiency of the boosting implementations.

Plot: [boosting_comparison.png](results/plots/boosting_comparison.png)

### Module D — Regression, optional (`results/metrics/regression_metrics.csv`)

80/20 split (seed=42), selection by mean 5-fold CV R² on training data.

| Model | MAE | RMSE | R² | CV R² (selection) |
|---|---|---|---|---|
| Linear Regression | 4198.11 | 5991.82 | 0.8046 | 0.7237 |
| Decision Tree | 3262.57 | 6546.66 | 0.7668 | 0.6420 |
| Random Forest | 2754.68 | 4843.29 | 0.8723 | 0.8104 |
| **Gradient Boosting** | **2505.84** | **4265.98** | **0.9010** | **0.8330** |

**Best model: Gradient Boosting**, winning on every metric including the
cross-validated selection criterion — unlike the classification and boosting
modules, there is no test/CV disagreement here. `charges` has strong
nonlinear interactions (notably `smoker` × `bmi` × `age`), which tree
ensembles capture far better than plain Linear Regression's R² of 0.80.

Plot: [regression_actual_vs_predicted.png](results/plots/regression_actual_vs_predicted.png)

## 12. Interpretation and limitations

- **No algorithm family wins everywhere.** KNN/SVM lead classification by
  cross-validated F1; AdaBoost — not the more sophisticated gradient-based
  boosters — leads boosting by the same criterion; Gradient Boosting
  dominates regression outright. The right algorithm is dataset- and
  metric-dependent, which is the central, demonstrated finding of this
  project.
- **Model selection vs. test reporting are kept separate.** Every module picks
  its "best" model using training-only cross-validation, then reports test
  metrics for all models for transparency — so the selection is not biased by
  having seen the test set, even though the test-set winner sometimes differs
  from the CV-selected model (see Modules A and C above).
- **DBSCAN's result is a genuine limitation**, not a bug: a density-based
  method is not well suited to this dataset's roughly-equal-density blob
  structure, and the report says so rather than discarding the inconvenient
  result.
- **Dataset size is a real constraint.** 400 rows (classification/boosting)
  and 200 rows (clustering) are small by modern ML standards; metrics here
  should be read as a methodology demonstration, not as benchmarks that would
  necessarily hold at production scale.
- **Regression module is optional** and kept architecturally separate
  (`src/regression.py`, `scripts/run_regression.py`) from the three required
  modules, per the assignment's scope.

## 13. Reproducibility

- Fixed `RANDOM_SEED = 42` in `src/config.py`, used for every train/test split,
  cross-validation fold, and stochastic estimator.
- All preprocessing (scaling, one-hot encoding) is fit inside a scikit-learn
  `Pipeline`/`ColumnTransformer` on the training fold only, then applied to the
  test fold — there is no preprocessing or target leakage.
- Model selection uses cross-validation on the training set, never the test
  set, for every module (see §11 for each module's selection metric).
- Exact package versions are pinned in `requirements.txt`; running
  `pip install -r requirements.txt` reproduces the same environment used to
  generate every number in this README.

## 14. Publishing to GitHub

This repository is not yet connected to a remote. To publish it:

```bash
git init
git add .
git commit -m "Initial commit: ML algorithms comparison project"
git branch -M main
git remote add origin https://github.com/<your-username>/ml-algorithms-comparison.git
git push -u origin main
```

Or, with the GitHub CLI authenticated:

```bash
gh repo create ml-algorithms-comparison --public --source=. --remote=origin --push
```
