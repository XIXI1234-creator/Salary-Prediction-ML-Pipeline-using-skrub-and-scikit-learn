# Salary Prediction ML Pipeline · `skrub` + `scikit-learn`

A clean, reproducible end-to-end machine-learning project that predicts annual
employee salaries from job-related features.
The pipeline uses [**skrub**](https://skrub-data.org/) to handle messy
tabular data effortlessly, and [**scikit-learn**](https://scikit-learn.org/)
for modeling and evaluation.

> Built as a portfolio project for a quantitative finance / data science
> student applying to open-source events. Beginner-friendly but
> structured like a serious production project.

---

## Highlights

- **Real-world style dataset** — public-sector salary data (categorical job
  titles, dates, departments, mixed types, missing values).
- **`skrub.tabular_pipeline`** — turns a messy dataframe into a working ML
  pipeline in a single line. No manual `ColumnTransformer` boilerplate.
- **Two models compared** — Ridge regression (linear baseline) vs.
  HistGradientBoostingRegressor.
- **Honest evaluation** — MAE, RMSE, R² + 5-fold cross-validation.
- **Visual diagnostics** — salary distribution, salary-by-department,
  salary vs. tenure, predicted-vs-actual, residual histogram.
- **100% reproducible** — fixed random seeds, pinned dependencies,
  self-contained dataset generator.

---

## Repository structure

```
salary-prediction-ml/
│
├── data/
│   └── employee_salaries.csv         # Generated dataset (9,500 rows)
│
├── notebooks/
│   └── salary_prediction.ipynb       # Main notebook (run me!)
│
├── src/
│   ├── generate_data.py              # Recreates the dataset from scratch
│   └── build_notebook.py             # (Used by maintainers to rebuild the .ipynb)
│
├── reports/
│   └── figures/                      # PNG plots saved by the notebook
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Clone and enter the project

```bash
git clone https://github.com/<your-username>/salary-prediction-ml.git
cd salary-prediction-ml
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python src/generate_data.py
```

This writes `data/employee_salaries.csv` (about 9,500 rows).

### 5. Open the notebook

```bash
jupyter notebook notebooks/salary_prediction.ipynb
```

Run all cells from top to bottom. The notebook is already pre-executed, so
you can also just read it on GitHub.

---

## Methodology

### The dataset
Each row is one employee. Features are:

| Column | Type | Description |
|---|---|---|
| `gender` | categorical | F / M (a few missing) |
| `department` | categorical | 3-letter department code |
| `department_name` | categorical | Full department name |
| `division` | categorical | Sub-unit within department |
| `assignment_category` | categorical | Full-time vs. part-time |
| `employee_position_title` | high-cardinality categorical | Job title (~20 unique) |
| `date_first_hired` | date string | First hire date (MM/DD/YYYY) |
| `year_first_hired` | integer | Year only |
| `current_annual_salary` | numeric | **Target** (in USD) |

### The pipeline

```
raw dataframe ──► skrub.TableVectorizer ──► sklearn estimator
                  (auto-encodes categoricals,
                   parses dates, imputes NaNs,
                   scales numerics if needed)
```

Two estimators are evaluated:

1. **Ridge regression** — linear baseline with L2 regularization.
2. **HistGradientBoostingRegressor** — gradient-boosted trees.

### Metrics

| Metric | What it tells you |
|---|---|
| **MAE**  | Average absolute error in dollars |
| **RMSE** | Penalizes large errors more than MAE |
| **R²**   | Fraction of salary variance explained (1.0 = perfect) |

---

## Example results

On a held-out 20% test set (random seed 42):

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Ridge                       | ~7,500 | ~9,990 | 0.88 |
| HistGradientBoostingRegressor | ~7,000 | ~9,250 | **0.90** |

5-fold cross-validated R² for HistGradientBoosting: **0.91 ± 0.003**.

The model explains roughly 90% of salary variance with a typical error
around $7,000 — a strong baseline for a project of this size.

---

## Possible improvements

- Hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`).
- Log-transform the target (`np.log1p`) to symmetrize the loss.
- Use `skrub.GapEncoder` to extract latent topics from job titles.
- Try `XGBoost` or `LightGBM`.
- Quantile regression for salary intervals instead of point predictions.
- Add a SHAP explanation step for feature importance.

---

## Reproducibility notes

- All randomness uses `random_state = 42`.
- The dataset is regenerated deterministically by `src/generate_data.py`.
- Dependencies are pinned with lower bounds in `requirements.txt`.
- The notebook is committed in *executed* form so plots and tables show on
  GitHub without running anything.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [**skrub**](https://skrub-data.org/) — for the genuinely useful
  `TableVectorizer` and `tabular_pipeline`.
- The dataset schema is inspired by the public
  [Montgomery County, MD employee salary records](https://data.montgomerycountymd.gov/).
