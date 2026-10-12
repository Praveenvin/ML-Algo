# Notebooks

Five notebooks, each runnable top-to-bottom from a fresh kernel. All of them read
their dataset(s) from `../data/` via `src/data_loader.py` — see
[`../data/README.md`](../data/README.md) for how to obtain those CSVs. None of the
notebooks require any image file, upload, or URL as input; any `.png` paths you
see in the code are optional **output** charts saved to `../results/plots/`.

## 01_exploratory_data_analysis.ipynb

**Datasets used:** all three —
`Social_Network_Ads.csv`, `Mall_Customers.csv`, `insurance_pre.csv`.

Loads and inspects each dataset (shape, dtypes, distributions) and renders one
combined overview figure. This is the only notebook that touches all three
datasets at once, since its purpose is a first look across the whole project
before diving into any single task.

## 02_classification.ipynb

**Dataset used:** `Social_Network_Ads.csv`

| Column | Type | Role |
|---|---|---|
| `User ID` | numeric | identifier, not used as a feature |
| `Gender` | categorical | feature |
| `Age` | numeric | feature |
| `EstimatedSalary` | numeric | feature |
| `Purchased` | binary (0/1) | **target** |

Trains and compares Logistic Regression, Decision Tree, Random Forest, SVM,
and KNN to predict whether a user purchases a product.

## 03_clustering.ipynb

**Dataset used:** `Mall_Customers.csv`

| Column | Type | Role |
|---|---|---|
| `CustomerID` | numeric | identifier, excluded from clustering |
| `Genre` | categorical | not used in the primary 2D segmentation |
| `Age` | numeric | used for cluster-profile summaries |
| `Annual Income (k$)` | numeric | clustering feature |
| `Spending Score (1-100)` | numeric | clustering feature |

Unsupervised — there is no target column. Segments customers with K-Means,
Agglomerative Clustering, and DBSCAN using `Annual Income (k$)` and
`Spending Score (1-100)`.

## 04_boosting.ipynb

**Dataset used:** `Social_Network_Ads.csv` (same as notebook 02)

Reuses the identical target (`Purchased`) and feature set as the
Classification notebook so that AdaBoost, Gradient Boosting, and
HistGradientBoosting can be compared against those standard classifiers on
equal footing.

## 05_regression_optional.ipynb

**Dataset used:** `insurance_pre.csv`

| Column | Type | Role |
|---|---|---|
| `age` | numeric | feature |
| `sex` | categorical | feature |
| `bmi` | numeric | feature |
| `children` | numeric | feature |
| `smoker` | categorical | feature |
| `charges` | numeric | **target** |

Optional module. Predicts medical insurance `charges` using Linear
Regression, Decision Tree, Random Forest, and Gradient Boosting regressors.

## Summary

| Notebook | Dataset(s) | Target | Task |
|---|---|---|---|
| `01_exploratory_data_analysis.ipynb` | all three | — | EDA |
| `02_classification.ipynb` | `Social_Network_Ads.csv` | `Purchased` | Classification |
| `03_clustering.ipynb` | `Mall_Customers.csv` | none (unsupervised) | Clustering |
| `04_boosting.ipynb` | `Social_Network_Ads.csv` | `Purchased` | Boosting |
| `05_regression_optional.ipynb` | `insurance_pre.csv` | `charges` | Regression (optional) |
