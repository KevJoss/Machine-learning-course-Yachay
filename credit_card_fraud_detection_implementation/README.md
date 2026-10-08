# Credit Card Fraud Detection — Exploratory Analysis & Classification

Exploratory data analysis and binary classification on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains 284,807 transactions made by European cardholders in September 2013, of which only 492 (~0.17%) are fraudulent.

---

## Contents

| File | Description |
| ---- | ----------- |
| `main.ipynb` | Full EDA, feature selection analysis, and Logistic Regression experiments |

---

## What the notebook covers

### 1. Initial Data Inspection
- Dataset dimensions (rows, columns)
- Variable types and data types summary
- Missing values analysis
- Duplicate rows detection

### 2. Exploratory Data Analysis (EDA)

**Class distribution**
- Count plot with class proportions to visualize the severe class imbalance (~99.8% legitimate vs ~0.2% fraud)

**Feature distributions**
- KDE plots for all 28 PCA components (V1–V28), split by class — to identify which features best separate fraud from legitimate transactions
- Box plots for `Amount` and key PCA features per class

**Temporal analysis**
- KDE plot of transaction frequency over time, separated by class
- X-axis converted from raw seconds to hour-of-day format (AM/PM) to detect temporal fraud patterns

**Correlation analysis**
- Heatmap of the correlation matrix across all V features

**Scatter plots**
- Scatter plot of selected feature pairs (V4 vs V7) colored by class

### 3. Feature Selection Insight
Visual exploration was used to identify which PCA components (e.g., `V14`, `V17`, `V12`) show the clearest separation between classes — the foundation for the 2-feature experiment below.

### 4. Logistic Regression — 2 Selected Features
- Features: `V4`, `V7`
- Stratified train/test split (80/20, `random_state=15`)
- `class_weight='balanced'` to handle class imbalance
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 5. Logistic Regression — All Features
- Same experimental conditions as above
- Uses all available features (V1-V28, Time, Amount)

### 6. Results Comparison

| Model | Features | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | 2 selected features | - | - | - | - |
| Logistic Regression | All features | - | - | - | - |

**Key takeaway:** In imbalanced fraud detection, **Recall** is the most critical metric — a false negative (missed fraud) has far greater real-world cost than a false positive.

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** scikit-learn, pandas, NumPy, matplotlib, seaborn
