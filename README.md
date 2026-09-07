# Machine Learning Course — Yachay Tech University

Code activities and projects developed for the Machine Learning course at Yachay Tech University. Each module explores a different aspect of the ML workflow, from data preprocessing to model evaluation.

---

## Repository Structure

```
Machine-learning-course-Yachay/
└── pipeline_automatic_preprocess/   # Automatic preprocessing & classification pipeline
```

---

## Modules

### [`pipeline_automatic_preprocess/`](pipeline_automatic_preprocess/README.md)

A modular, scikit-learn-compatible pipeline that automates the full preprocessing workflow for tabular datasets — imputation, encoding, and scaling — through a single configurable object. Raw `pandas` DataFrames can be fed directly into a `LogisticRegression` classifier without any manual transformation.

**Key components:**

| File | Description |
| ---- | ----------- |
| `tools/transform.py` | `DataFramePreparer` transformer and `build_full_pipeline` factory |
| `tools/preprocessing.py` | Train / val / test split and label separation utilities |
| `main.ipynb` | Experiments: 6 preprocessing configurations compared by Accuracy, Precision, Recall, and F1-score |
| `Tests/` | Pytest suite with 4 AI-generated synthetic datasets covering missing values, all-numeric, outliers, and binary categorical edge cases |

---

## Tech Stack

- **Language:** Python 3
- **Core libraries:** scikit-learn, pandas, NumPy
- **Testing:** pytest
