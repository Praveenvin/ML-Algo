# Data

This directory holds the raw CSV datasets used by the project. They are excluded from
version control by `.gitignore` (`data/*.csv`) to keep the repository lightweight and
avoid redistributing third-party data without explicit permission.

| File | Rows | Columns | Used by |
|---|---|---|---|
| `Social_Network_Ads.csv` | 400 | `User ID, Gender, Age, EstimatedSalary, Purchased` | Module A (Classification), Module C (Boosting) |
| `Mall_Customers.csv` | 200 | `CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100)` | Module B (Clustering) |
| `insurance_pre.csv` | 1338 (1337 after de-duplication) | `age, sex, bmi, children, smoker, charges` | Module D (Regression, optional) |

To reproduce the results, place these three files in this directory before running
`scripts/run_all.py` or any individual `scripts/run_*.py` script.
