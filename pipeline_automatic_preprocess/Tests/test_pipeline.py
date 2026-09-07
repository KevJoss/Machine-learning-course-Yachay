import sys
import os
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from transform import build_full_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASETS = [
    ("dataset_missing_values.csv",    "mean",   "ONE-HOT",  "ROBUST"),
    ("dataset_missing_values.csv",    "median", "LABEL",    "STANDARD"),
    ("dataset_all_numeric.csv",       "mean",   "ONE-HOT",  "ROBUST"),
    ("dataset_all_numeric.csv",       "median", "ONE-HOT",  "STANDARD"),
    ("dataset_outliers.csv",          "median", "ONE-HOT",  "ROBUST"),
    ("dataset_outliers.csv",          "mean",   "LABEL",    "STANDARD"),
    ("dataset_binary_categorical.csv","mode",   "ONE-HOT",  "ROBUST"),
    ("dataset_binary_categorical.csv","median", "LABEL",    "STANDARD"),
]


def run_pipeline(csv_file, numeric_strategy, category_strategy, scaler_strategy):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_full_pipeline(
        numeric_strategy=numeric_strategy,
        category_strategy=category_strategy,
        scaler_strategy=scaler_strategy,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    return acc, f1


@pytest.mark.parametrize("csv_file,numeric_strategy,category_strategy,scaler_strategy", DATASETS)
def test_pipeline_runs(csv_file, numeric_strategy, category_strategy, scaler_strategy):
    acc, f1 = run_pipeline(csv_file, numeric_strategy, category_strategy, scaler_strategy)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1  <= 1.0


@pytest.mark.parametrize("csv_file,numeric_strategy,category_strategy,scaler_strategy", DATASETS)
def test_pipeline_no_exception(csv_file, numeric_strategy, category_strategy, scaler_strategy):
    try:
        run_pipeline(csv_file, numeric_strategy, category_strategy, scaler_strategy)
    except Exception as e:
        pytest.fail(f"Pipeline raised an exception: {e}")


def test_invalid_numeric_strategy():
    with pytest.raises(ValueError):
        build_full_pipeline(numeric_strategy="sum")


def test_invalid_category_strategy():
    with pytest.raises(ValueError):
        build_full_pipeline(numeric_strategy="mean", category_strategy="BINARY")


def test_invalid_scaler_strategy():
    with pytest.raises(ValueError):
        build_full_pipeline(numeric_strategy="mean", scaler_strategy="MINMAX")
