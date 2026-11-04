# Menolyte Core — Churn Prediction, Optimization and Explainable AI

Machine Learning course project (Yachay Tech) on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset. It predicts which customers of an internet/telecom provider are likely to leave, explains why, and turns
the predictions into a retention tool for mid-sized ISPs: **Menolyte Core**.

**Author:** Kevin Josué Sánchez Salazar

## Contents

| Path | Description |
|---|---|
| `telco_churn_optimization_xai.ipynb` | Main deliverable: EDA, pipelines, model comparison, hyperparameter optimization (Grid, Random, TPE), multi-objective optimization (NSGA-II), SHAP and LIME, final test evaluation, business figures and conclusions. |
| `data/` | Telco Customer Churn CSV (7,043 customers, 21 columns). |
| `models/final_svm_pipeline.joblib` | Final fitted pipeline (preprocessing + SVM), exported by the notebook. |
| `models/app_results.joblib` | Results the app displays (metrics, optimization results, Pareto front, test predictions, SHAP background). |
| `streamlit_app/app.py` | Streamlit app: EDA, Model Performance, Prediction & Explainability, Business Application. |
| `streamlit_app/requirements.txt` | Pinned dependencies for the app. |
| `figures/` | Business figures (English) exported by the notebook; `figures/es/` holds Spanish versions for the pitch slides. |

## How to run

Requires Python 3.14 (the versions in `requirements.txt` were tested with it).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r streamlit_app/requirements.txt
```

**Streamlit app** (loads the exported model; nothing is retrained). Run it from this folder:

```bash
streamlit run streamlit_app/app.py
```

**Notebook.** It also needs Optuna and Jupyter. Running it end to end takes about 15 minutes (SHAP with an SVM is the
slowest part) and regenerates `models/` and `figures/`:

```bash
pip install optuna==5.0.0 jupyter
jupyter notebook telco_churn_optimization_xai.ipynb
```

**Deployed app:** https://ai-business-optimization.streamlit.app

## Methodology

- **Leakage control:** stratified 80/20 train/test split; all preprocessing (`StandardScaler`, `OneHotEncoder`) inside a
  `Pipeline`/`ColumnTransformer`; the test set is used once, after model selection and optimization.
- **Primary metric: F2.** A missed churner loses all of the customer's future revenue, while an unnecessary retention
  contact is cheap, so recall weighs twice as much as precision.
- **Models compared** (5-fold stratified CV, `class_weight="balanced"`): Logistic Regression, Random Forest and SVM (RBF).
- **Hyperparameter optimization** of the SVM (`C`, `gamma`) with Grid Search, Random Search and TPE (Optuna), with the
  same folds and a budget of 36 configurations each.
- **Multi-objective optimization:** NSGA-II maximizing F2 and minimizing inference time; Pareto front analysis.
- **Explainability:** SHAP (global and local) and LIME on the same customers, compared with Random Forest importance.

## Main results

| | Value |
|---|---|
| Selected model | SVM (RBF), Grid Search hyperparameters |
| Test F2 / Recall / ROC-AUC | 0.714 / 0.808 / 0.835 (close to cross-validation: no overfitting) |
| Contacting the riskiest 20 % of customers | reaches 50 % of the customers who leave |
| Actual churn in the high-risk segment | 72 % (vs 10 % in the low-risk segment) |
| Main churn drivers (SHAP and LIME agree) | contract type, tenure, monthly charge, no online security / technical support |

All figures are computed in the notebook. Explanations show association, not causation.

## Business proposal: Menolyte Core

A SaaS platform for mid-sized and regional ISPs without their own data science team: every month it scores the
customer base, ranks customers by churn risk, explains each case and suggests a retention action.
Hypothetical launch pricing: **$490/month** (up to 10,000 customers) **+ 10 % of the revenue retained** during the
first year. Roadmap: a multi-agent system (orchestrator, analyst agent for the manager, WhatsApp campaign agent),
with human approval of every campaign and customer consent as required by Ecuador's data protection law (LOPDP).

## Use of Artificial Intelligence

The code and markdown of this project were generated with the assistance of the Claude agent (Claude Code, Opus 5.5
model, *High* effort), used as a tool to generate code and to understand concepts, figures and results. All technical
and business decisions, and the full understanding of the work, belong to the student.
