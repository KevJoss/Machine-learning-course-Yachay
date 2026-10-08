"""
utils.py — Contrato común del proyecto CREMA-D (Ensemble + Weighted Soft Voting)

Uso en Colab:
    import sys
    sys.path.append('/content/drive/MyDrive/CREMA-D_Grupo/common')
    import utils as U

Este módulo NO se modifica sin avisar al grupo. Si necesitas una función nueva,
escríbela en tu propio notebook o propón el cambio en el grupo.
"""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

# ----------------------------------------------------------------------------
# Constantes del contrato
# ----------------------------------------------------------------------------
CLASSES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]   # ORDEN OBLIGATORIO
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PROB_COLS = [f"prob_{c}" for c in CLASSES]
SPLITS = ["train", "val", "test"]

SEED = 42
SR = 16000

PROJECT = Path(os.environ.get("CREMAD_PROJECT",
                              "/content/drive/MyDrive/CREMA-D_Grupo"))
SPLITS_PATH = PROJECT / "splits" / "splits.csv"
PROBS_DIR = PROJECT / "probs"
FROZEN_CONFIG_PATH = PROJECT / "results" / "frozen_config.json"


SPLITS_MD5 = '1dc42b05f19ea3059a3f9a4bf01ebc97'  # hash oficial de splits.csv


# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Fija la semilla en random, numpy y torch (si está instalado)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def file_md5(path) -> str:
    """Huella digital de un archivo. Cambia si cambia un solo byte."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------
# Split oficial
# ----------------------------------------------------------------------------
def load_splits(path=SPLITS_PATH, check_hash: bool = True) -> pd.DataFrame:
    """Carga splits.csv y verifica columnas y (si está definido) su hash."""
    df = pd.read_csv(path)
    required = {"sample_id", "path", "actor_id", "emotion", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"A splits.csv le faltan columnas: {missing}")
    if check_hash and SPLITS_MD5 is not None:
        current = file_md5(path)
        if current != SPLITS_MD5:
            raise ValueError(
                "splits.csv NO coincide con la versión oficial del grupo.\n"
                f"  esperado: {SPLITS_MD5}\n  obtenido: {current}")
    return df


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Devuelve una partición ('train', 'val' o 'test') con índice limpio."""
    if split not in SPLITS:
        raise ValueError(f"split debe ser uno de {SPLITS}")
    return df[df["split"] == split].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Etiquetas y orden de clases
# ----------------------------------------------------------------------------
def labels_to_idx(labels) -> np.ndarray:
    """['ANG', 'SAD', ...] -> [0, 5, ...] siguiendo CLASSES."""
    return np.array([CLASS_TO_IDX[str(l)] for l in labels])


def idx_to_labels(idx) -> np.ndarray:
    """[0, 5, ...] -> ['ANG', 'SAD', ...] siguiendo CLASSES."""
    return np.array([CLASSES[int(i)] for i in idx])


def _to_labels(y) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        return idx_to_labels(y)
    return y.astype(str)


def reorder_proba(proba, model_classes) -> np.ndarray:
    """
    Reordena las columnas de predict_proba al orden de CLASSES.

    model_classes: el orden que usa tu modelo (por ejemplo model.classes_ en
    scikit-learn). Acepta etiquetas ('ANG', ...) o índices (0..5 según CLASSES).
    """
    model_classes = list(model_classes)
    if all(isinstance(c, (int, np.integer)) for c in model_classes):
        model_classes = [CLASSES[int(c)] for c in model_classes]
    model_classes = [str(c) for c in model_classes]
    if sorted(model_classes) != sorted(CLASSES):
        raise ValueError(f"Las clases del modelo {model_classes} no son {CLASSES}")
    order = [model_classes.index(c) for c in CLASSES]
    return np.asarray(proba)[:, order]


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    """Las 5 métricas que exige la tarea. Acepta etiquetas o índices."""
    yt, yp = _to_labels(y_true), _to_labels(y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        yt, yp, labels=CLASSES, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(yt, yp),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
        "weighted_f1": f1_score(yt, yp, labels=CLASSES,
                                average="weighted", zero_division=0),
    }


def metrics_from_probs(y_true, probs) -> dict:
    """Métricas a partir de una matriz de probabilidades (N, 6)."""
    return compute_metrics(y_true, idx_to_labels(np.argmax(probs, axis=1)))


def plot_confusion_matrix(y_true, y_pred, title: str = "",
                          normalize: bool = True, ax=None):
    """
    Matriz de confusión en el orden de CLASSES.
    normalize=True divide cada fila por su total: la diagonal es el recall
    de cada clase.
    """
    yt, yp = _to_labels(y_true), _to_labels(y_pred)
    cm = confusion_matrix(yt, yp, labels=CLASSES)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    threshold = shown.max() / 2
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > threshold else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return ax


# ----------------------------------------------------------------------------
# Probabilidades: guardar y cargar con validación
# ----------------------------------------------------------------------------
def _check_probs(probs: np.ndarray) -> None:
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES):
        raise ValueError(f"probs debe tener forma (N, 6); tiene {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("probs contiene NaN o infinitos")
    if (probs < -1e-6).any():
        raise ValueError("probs tiene valores negativos: ¿son logits? "
                         "Aplica softmax antes de guardar.")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        raise ValueError(f"Las filas no suman 1 (min={sums.min():.4f}, "
                         f"max={sums.max():.4f}). ¿Olvidaste el softmax?")


def save_probs(sample_ids, y_true, probs, model_name: str, split: str,
               splits_df: pd.DataFrame | None = None) -> Path:
    """
    Guarda probs/{split}/probs_{split}_{model_name}.csv con el formato común.

    Verifica: forma (N, 6), filas que suman 1, sample_id únicos, que los
    sample_id sean EXACTAMENTE los de esa partición y que las etiquetas
    coincidan con splits.csv. Para split='test' exige que el ensemble ya
    esté congelado (results/frozen_config.json).
    """
    if split not in ("val", "test"):
        raise ValueError("Solo se guardan probabilidades de 'val' o 'test'")
    if split == "test" and not FROZEN_CONFIG_PATH.exists():
        raise PermissionError(
            "No se pueden generar probabilidades de Test antes de congelar el "
            f"ensemble. Falta {FROZEN_CONFIG_PATH}.")

    probs = np.asarray(probs, dtype=float)
    _check_probs(probs)
    sample_ids = pd.Series(sample_ids, dtype=str)
    y_true = _to_labels(y_true)
    if not (len(sample_ids) == len(y_true) == len(probs)):
        raise ValueError("sample_ids, y_true y probs tienen longitudes distintas")
    if not sample_ids.is_unique:
        raise ValueError("Hay sample_id repetidos")

    splits_df = load_splits() if splits_df is None else splits_df
    ref = get_split(splits_df, split).set_index("sample_id")["emotion"]
    if set(sample_ids) != set(ref.index):
        faltan = len(set(ref.index) - set(sample_ids))
        sobran = len(set(sample_ids) - set(ref.index))
        raise ValueError(f"Los sample_id no coinciden con la partición '{split}': "
                         f"faltan {faltan}, sobran {sobran}")
    if (ref.loc[sample_ids].values != y_true).any():
        raise ValueError("Las etiquetas no coinciden con splits.csv")

    df = pd.DataFrame(probs, columns=PROB_COLS)
    df.insert(0, "true_label", y_true)
    df.insert(0, "sample_id", sample_ids.values)
    df = df.sort_values("sample_id").reset_index(drop=True)

    out = PROBS_DIR / split / f"probs_{split}_{model_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    m = metrics_from_probs(df["true_label"], df[PROB_COLS].values)
    print(f"Guardado {out}  ({len(df)} filas)  "
          f"acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
    return out


def load_probs(model_name: str, split: str = "val") -> pd.DataFrame:
    """Carga y valida probs/{split}/probs_{split}_{model_name}.csv."""
    path = PROBS_DIR / split / f"probs_{split}_{model_name}.csv"
    df = pd.read_csv(path)
    expected = ["sample_id", "true_label"] + PROB_COLS
    if list(df.columns) != expected:
        raise ValueError(f"{path.name}: columnas {list(df.columns)}, "
                         f"se esperaban {expected}")
    _check_probs(df[PROB_COLS].values)
    return df


# ----------------------------------------------------------------------------
# Inferencia sobre un audio nuevo
# ----------------------------------------------------------------------------
def validate_predict_fn(fn, wav_path, name: str = "") -> np.ndarray:
    """
    Comprueba que predict_proba_M{n}(wav_path) devuelve 6 probabilidades
    válidas en el orden de CLASSES.
    """
    out = np.asarray(fn(wav_path), dtype=float)
    if out.shape != (len(CLASSES),):
        raise ValueError(f"{name}: debe devolver forma (6,), devolvió {out.shape}")
    _check_probs(out[None, :])
    print(f"{name or fn.__name__} OK -> {CLASSES[int(out.argmax())]}  "
          + "  ".join(f"{c}={p:.2f}" for c, p in zip(CLASSES, out)))
    return out