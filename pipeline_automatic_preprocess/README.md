# Automatic Preprocessing Pipeline

A modular, scikit-learn-compatible preprocessing and classification pipeline for tabular datasets. The project automates the most repetitive steps of an ML workflow — imputation, encoding, and scaling — through a single configurable object, so that raw DataFrames can be fed directly into a classifier without any manual transformation.

---

## Project Structure

```
pipeline_automatic_preprocess/
├── main.ipynb          # Main notebook: experiments, comparisons, and predictions
├── tools/
│   ├── preprocessing.py  # Train / validation / test splitting utilities
│   └── transform.py      # Core pipeline classes and factory functions
├── data/
│   └── Titanic-Dataset.csv
└── Tests/
    ├── README.md         # Dataset descriptions
    ├── test_pipeline.py  # Pytest test suite
    └── data/             # Four AI-generated synthetic datasets
```

---

## How It Works

### 1. `tools/preprocessing.py`

| Function                        | Description                                                                                         |
| ------------------------------- | --------------------------------------------------------------------------------------------------- |
| `train_val_test_split(df, ...)` | Splits a DataFrame into train / validation / test sets (60 / 20 / 20) with optional stratification. |
| `remove_labels(df, label_name)` | Separates features `X` and target `y` from a DataFrame.                                             |

### 2. `tools/transform.py`

#### `DataFramePreparer`
The central transformer. Inherits from `BaseEstimator` and `TransformerMixin`, making it fully compatible with scikit-learn pipelines.

It internally builds a `ColumnTransformer` with two sub-pipelines:

- **Numeric pipeline** → `SimpleImputer` → Scaler (`RobustScaler` or `StandardScaler`)
- **Categorical pipeline** → `SimpleImputer(most_frequent)` → Encoder (`OneHotEncoder` or `OrdinalEncoder`)

The output is always a clean, fully numeric `pandas.DataFrame`.

| Parameter           | Options                        | Description                               |
| ------------------- | ------------------------------ | ----------------------------------------- |
| `numeric_strategy`  | `"mean"`, `"median"`, `"mode"` | Imputation strategy for numeric columns   |
| `category_strategy` | `"ONE-HOT"`, `"LABEL"`         | Encoding strategy for categorical columns |
| `scaler_strategy`   | `"ROBUST"`, `"STANDARD"`       | Scaling strategy for numeric columns      |

#### `build_full_pipeline(...)`
Factory function that chains `DataFramePreparer` + `LogisticRegression` into a single end-to-end `Pipeline`. Accepts all `DataFramePreparer` parameters plus any `LogisticRegression` keyword arguments.

```python
from tools.transform import build_full_pipeline

pipeline = build_full_pipeline(
    numeric_strategy="mean",
    category_strategy="ONE-HOT",
    scaler_strategy="ROBUST"
)
pipeline.fit(X_train, y_train)
pipeline.predict(X_new)   # Raw, unprocessed DataFrame — no manual preprocessing needed
```

---

## Scaler Reference

| Scaler           | Formula                           | Best When                               |
| ---------------- | --------------------------------- | --------------------------------------- |
| `RobustScaler`   | $\frac{(x − \text{median})}{IQR}$ | Data contains outliers                  |
| `StandardScaler` | $\frac{(x − \mu)}{\sigma}$        | Features follow a Gaussian distribution |


---

## Running the Tests

```powershell
python -m pytest Tests\test_pipeline.py -v
```

See [`Tests/README.md`](Tests/README.md) for details on each synthetic dataset.
