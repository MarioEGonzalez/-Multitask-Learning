# -*- coding: utf-8 -*-
"""
07_robustness_checks_tabnet_weighted_and_negative_control.py

Reproducibility script for the two robustness checks reported in
Section 3.6 ("Robustness Checks: Class-Weighted Baseline and Negative
Control") of the manuscript:

  [1] TABNET WITH CLASS WEIGHTS               -> responds to Reviewer 3, Comment 2
  [2] MULTITASK NEGATIVE CONTROL (label shuffle) -> responds to Reviewer 3, Comment 8

This script reuses EXACTLY the same architecture, hyperparameters, and
cached Top-K=4096 folds produced by 01_crear_datasets_representaciones_cv5.py
and used by 02_entrenar_modelos_y_generar_resultados.py. No data-pipeline
step is regenerated here.

Note on scope: this file intentionally contains ONLY the two experiments
above. The Random Forest / Boosting / CNN1D / RNN baselines live in
04_baselines_adicionales_revision.py, and the capacity-matched single-task
MLP baseline lives in 06_singletask_matched_mlp.py. An SVM-RBF baseline
was also explored during revision but is deliberately not included here
(nor reported in the manuscript).

Outputs (subfolders of 08_experimentos_revision, matching the same
directory layout already used for the other revision scripts):
  .../08_experimentos_revision/01_tabnet_weighted        (Section 3.6, Table 13)
  .../08_experimentos_revision/02_multitask_negcontrol    (Section 3.6, Table 14, Fig. 24)
  .../08_experimentos_revision/resumen_robustness_checks_seccion_3_6.csv  (consolidated)
"""

import os
import sys
import gc
import time
import random
import subprocess
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =============================================================================
# 0. IMPORT TABNET (same as 02_...)
# =============================================================================

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    print("pytorch-tabnet is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet"])
    from pytorch_tabnet.tab_model import TabNetClassifier

# =============================================================================
# 1. GENERAL CONFIGURATION (identical to 02_entrenar_modelos_y_generar_resultados.py)
# =============================================================================

SEED = 42

PROJECT_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold")
REPRO_ROOT = PROJECT_ROOT / "reproducible"

CV_ROOT = REPRO_ROOT / "04_cv5_topk4096"

OUT_REVISION = REPRO_ROOT / "08_experimentos_revision"
OUT_TABNET_W = OUT_REVISION / "01_tabnet_weighted"
OUT_MT_NEGCTRL = OUT_REVISION / "02_multitask_negcontrol"

N_SPLITS = 5
INPUT_DIM = 4096
N_CLASSES_JACKET = 5
N_CLASSES_ROTOR = 5

REPRESENTATIONS = {
    "fft": {"name": "FFT log-magnitude", "folder": "01_FFT_log_magnitude_cv5_topk4096"},
    "stft": {"name": "STFT log-power", "folder": "02_STFT_log_power_cv5_topk4096"},
    "welch": {"name": "Welch PSD log-power", "folder": "03_Welch_PSD_log_power_cv5_topk4096"},
}

N_CLASSES_BY_DATASET = {"jacket": N_CLASSES_JACKET, "rotor": N_CLASSES_ROTOR}

# ---- ON/OFF flags ----
RUN_TABNET_WEIGHTED = True         # [1] R3#2
RUN_MULTITASK_NEGCONTROL = True    # [2] R3#8

# ---- TabNet hyperparameters (identical to 02_...) ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

TABNET_PARAMS = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-4,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 40, "gamma": 0.7},
    "mask_type": "entmax",
    "seed": SEED,
    "device_name": DEVICE_NAME,
    "verbose": 10,
}
TABNET_MAX_EPOCHS = 120
TABNET_PATIENCE = 0
TABNET_BATCH_SIZE = 512
TABNET_VIRTUAL_BATCH_SIZE = 128

# ---- Multitask hyperparameters (identical to 02_...) ----
BATCH_SIZE_JACKET = 256
BATCH_SIZE_ROTOR = 128
MT_EPOCHS = 120
MT_LR = 1e-3
MT_WEIGHT_DECAY = 1e-5
MT_DROPOUT = 0.25
MT_ALPHA_JACKET = 1.0
MT_BETA_ROTOR = 1.0
MT_SCHEDULER_STEP_SIZE = 40
MT_SCHEDULER_GAMMA = 0.7

BATCH_SIZE_BY_DATASET = {"jacket": BATCH_SIZE_JACKET, "rotor": BATCH_SIZE_ROTOR}


# =============================================================================
# 2. SHARED UTILITIES (copied from 01_/02_)
# =============================================================================

def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_header(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def print_subheader(text: str) -> None:
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def clean_class_name(name) -> str:
    s = str(name)
    replacements = {
        "class_0": "Healthy",
        "class_1": "Class 1",
        "class_2": "Class 2",
        "class_3": "Class 3",
        "class_4": "Class 4",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def save_figure(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def validate_fold_arrays(name: str, X_train, X_test, y_train, y_test) -> None:
    print(f"{name}: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")
    if X_train.shape[1] != INPUT_DIM:
        raise ValueError(f"{name}: X_train does not have {INPUT_DIM} features.")
    if X_test.shape[1] != INPUT_DIM:
        raise ValueError(f"{name}: X_test does not have {INPUT_DIM} features.")
    if len(X_train) != len(y_train):
        raise ValueError(f"{name}: X_train/y_train length mismatch.")
    if len(X_test) != len(y_test):
        raise ValueError(f"{name}: X_test/y_test length mismatch.")
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError(f"{name}: NaN values found.")
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        raise ValueError(f"{name}: Inf values found.")


def fold_file(dataset: str, rep_key: str, fold_id: int) -> Path:
    folder = CV_ROOT / REPRESENTATIONS[rep_key]["folder"]
    path = folder / f"{dataset}_{rep_key}_fold{fold_id}_topk4096.npz"
    if not path.exists():
        candidates = sorted(folder.glob(f"{dataset}*{rep_key}*fold{fold_id}*topk4096*.npz"))
        if not candidates:
            candidates = sorted(folder.glob(f"{dataset}*fold{fold_id}*topk4096*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No fold file found for dataset={dataset}, rep={rep_key}, fold={fold_id} in {folder}")
        path = candidates[0]
    return path


def load_fold(dataset: str, rep_key: str, fold_id: int) -> dict:
    path = fold_file(dataset, rep_key, fold_id)
    data = np.load(path, allow_pickle=True)

    required = ["X_train", "X_test", "y_train", "y_test", "class_names"]
    for key in required:
        if key not in data.files:
            raise KeyError(f"Missing key {key} in {path}")

    X_train = data["X_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    class_names = np.array([clean_class_name(x) for x in data["class_names"]], dtype=object)

    validate_fold_arrays(f"{dataset}_{rep_key}_fold{fold_id}", X_train, X_test, y_train, y_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "class_names": class_names,
    }


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (n_classes * counts)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def get_class_weights_dict(y):
    """Same formula as the manuscript's class-weighting scheme
    (w_{t,c} = n_t / (K_t * n_{t,c})); used here as the `weights=`
    argument passed to TabNetClassifier.fit()."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    weights = {}
    for c, n in zip(classes, counts):
        weights[int(c)] = float(total / (n_classes * n))
    return weights


def create_loader(X, y, batch_size, shuffle=True, seed=42) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                       num_workers=0, generator=generator)


def compute_metrics_row(y_true, y_pred, model, dataset, representation, fold_id=None) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "representation": representation,
        "fold_id": fold_id if fold_id is not None else "global_oof",
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_confusion_matrix(cm, class_names, title, out_base: Path, normalize: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sums, out=np.zeros_like(cm_plot), where=row_sums != 0)
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot, fmt = cm, "d"
        vmax = max(float(np.max(cm_plot)), 1.0)

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax if normalize else vmax * 1.25)
    ax.set_title(title, color="black", pad=18)
    ax.set_xlabel("Predicted label", color="black", labelpad=12)
    ax.set_ylabel("True label", color="black", labelpad=12)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", color="black")
    ax.set_yticklabels(class_names, color="black")

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            text_value = format(cm_plot[i, j], fmt) if normalize else format(int(cm_plot[i, j]), fmt)
            ax.text(j, i, text_value, ha="center", va="center", color="black", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="black")

    plt.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


def save_classification_outputs(out_dir: Path, model_name: str, dataset: str, rep_key: str,
                                 y_true_all: list, y_pred_all: list, fold_rows: list, class_names) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.concatenate(y_true_all).astype(np.int64)
    y_pred = np.concatenate(y_pred_all).astype(np.int64)
    representation = REPRESENTATIONS[rep_key]["name"]

    global_row = compute_metrics_row(y_true, y_pred, model=model_name, dataset=dataset,
                                      representation=representation, fold_id=None)

    pd.DataFrame(fold_rows).to_csv(out_dir / f"{model_name}_{dataset}_{rep_key}_metrics_per_fold.csv",
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame([global_row]).to_csv(out_dir / f"{model_name}_{dataset}_{rep_key}_metrics_global_oof.csv",
                                       index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=[f"True {c}" for c in class_names],
                 columns=[f"Pred {c}" for c in class_names]).to_csv(
        out_dir / f"{model_name}_{dataset}_{rep_key}_confusion_matrix_global.csv", encoding="utf-8-sig")

    report_txt = classification_report(y_true, y_pred, labels=np.arange(len(class_names)),
                                        target_names=list(class_names), zero_division=0)
    (out_dir / f"{model_name}_{dataset}_{rep_key}_classification_report.txt").write_text(report_txt, encoding="utf-8")

    plot_confusion_matrix(cm, class_names, title=f"Global CM - {model_name} - {dataset.capitalize()} - {representation}",
                           out_base=out_dir / f"{model_name}_{dataset}_{rep_key}_confusion_matrix_global", normalize=False)
    plot_confusion_matrix(cm, class_names, title=f"Normalized CM - {model_name} - {dataset.capitalize()} - {representation}",
                           out_base=out_dir / f"{model_name}_{dataset}_{rep_key}_confusion_matrix_global_normalized", normalize=True)

    return global_row


def evaluate_model_single_head(model, X, y, task=None, batch_size: int = 512):
    model.eval()
    loader = create_loader(X, y, batch_size=batch_size, shuffle=False, seed=SEED)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb, task) if task is not None else model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    return np.concatenate(trues).astype(np.int64), np.concatenate(preds).astype(np.int64)


# =============================================================================
# 3. ARCHITECTURE (MultiTaskMLP, identical to 02_..., needed for [2])
# =============================================================================

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim=4096, n_classes_jacket=5, n_classes_rotor=5, dropout=0.25):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.head_jacket = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes_jacket))
        self.head_rotor = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes_rotor))

    def forward(self, x, task: str):
        h = self.shared(x)
        if task == "jacket":
            return self.head_jacket(h)
        if task == "rotor":
            return self.head_rotor(h)
        raise ValueError(f"Unrecognized task: {task}")

    def encode(self, x):
        return self.shared(x)


# =============================================================================
# [1] TABNET WITH CLASS WEIGHTS  ->  Reviewer 3, Comment 2
# =============================================================================

def train_tabnet_weighted_single_case(dataset: str, rep_key: str) -> dict:
    print_header(f"TABNET WEIGHTED - {dataset.upper()} - {REPRESENTATIONS[rep_key]['name']}")
    model_name = "TabNet_weighted"
    case_out = OUT_TABNET_W / f"{dataset}_{rep_key}"
    models_dir = case_out / "modelos"
    models_dir.mkdir(parents=True, exist_ok=True)

    y_true_all, y_pred_all, fold_rows = [], [], []
    class_names_ref = None

    for fold_id in range(1, N_SPLITS + 1):
        print_subheader(f"TabNet weighted {dataset}_{rep_key} fold {fold_id}")
        fold = load_fold(dataset, rep_key, fold_id)
        X_train, X_test = fold["X_train"], fold["X_test"]
        y_train, y_test = fold["y_train"], fold["y_test"]
        class_names_ref = fold["class_names"]

        weights = get_class_weights_dict(y_train)

        clf = TabNetClassifier(**TABNET_PARAMS)
        batch_size = min(TABNET_BATCH_SIZE, len(X_train))
        virtual_batch_size = min(TABNET_VIRTUAL_BATCH_SIZE, batch_size)

        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_test, y_test)], eval_name=["test"], eval_metric=["accuracy"],
            max_epochs=TABNET_MAX_EPOCHS, patience=TABNET_PATIENCE,
            batch_size=batch_size, virtual_batch_size=virtual_batch_size,
            num_workers=0, drop_last=False,
            weights=weights,
        )

        y_pred = clf.predict(X_test).astype(np.int64)
        fold_row = compute_metrics_row(y_test, y_pred, model=model_name, dataset=dataset,
                                        representation=REPRESENTATIONS[rep_key]["name"], fold_id=fold_id)
        fold_rows.append(fold_row)
        y_true_all.append(y_test); y_pred_all.append(y_pred)

        clf.save_model(str(models_dir / f"tabnet_weighted_{dataset}_{rep_key}_fold{fold_id}"))
        pd.DataFrame([fold_row]).to_csv(case_out / f"tabnet_weighted_{dataset}_{rep_key}_fold{fold_id}_metrics.csv",
                                         index=False, encoding="utf-8-sig")

        del clf, X_train, X_test, y_train, y_test, y_pred
        gc.collect()

    return save_classification_outputs(case_out, model_name, dataset, rep_key,
                                        y_true_all, y_pred_all, fold_rows, class_names_ref)


def run_all_tabnet_weighted() -> pd.DataFrame:
    print_header("EXPERIMENT [1] - TABNET WITH CLASS WEIGHTS (Reviewer 3, Comment 2)")
    rows = []
    for rep_key in ["fft", "stft", "welch"]:
        for dataset in ["jacket", "rotor"]:
            rows.append(train_tabnet_weighted_single_case(dataset, rep_key))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABNET_W / "metrics_global_tabnet_weighted_all_cases.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# [2] MULTITASK NEGATIVE CONTROL (shuffled training labels)  ->  Reviewer 3, Comment 8
# =============================================================================

def train_multitask_negcontrol_case(rep_key: str, shuffle_task: str) -> pd.DataFrame:
    """shuffle_task in {'jacket','rotor'}: that task is trained with shuffled
    (noise) labels, the other keeps real labels. TEST always uses real labels
    for both tasks -- this measures whether the task with real labels
    degrades when the auxiliary task's training signal is pure noise."""
    print_header(f"MULTITASK NEGATIVE CONTROL - {REPRESENTATIONS[rep_key]['name']} - shuffle={shuffle_task}")
    tag = f"Multitask_negctrl_shuffle_{shuffle_task}"
    rep_out = OUT_MT_NEGCTRL / f"{rep_key}_shuffle_{shuffle_task}"
    models_dir = rep_out / "modelos"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    y_true_oof_j, y_pred_oof_j = [], []
    y_true_oof_r, y_pred_oof_r = [], []
    fold_rows_j, fold_rows_r = [], []
    class_names_j = class_names_r = None

    for fold_id in range(1, N_SPLITS + 1):
        print_subheader(f"Negctrl {rep_key} shuffle={shuffle_task} fold {fold_id}")
        fj = load_fold("jacket", rep_key, fold_id)
        fr = load_fold("rotor", rep_key, fold_id)

        Xj_train, Xj_test = fj["X_train"], fj["X_test"]
        yj_train, yj_test = fj["y_train"], fj["y_test"]
        Xr_train, Xr_test = fr["X_train"], fr["X_test"]
        yr_train, yr_test = fr["y_train"], fr["y_test"]
        class_names_j, class_names_r = fj["class_names"], fr["class_names"]

        rng = np.random.RandomState(SEED + fold_id)
        if shuffle_task == "jacket":
            yj_train = yj_train[rng.permutation(len(yj_train))]
        elif shuffle_task == "rotor":
            yr_train = yr_train[rng.permutation(len(yr_train))]
        else:
            raise ValueError("shuffle_task must be 'jacket' or 'rotor'")

        train_loader_j = create_loader(Xj_train, yj_train, BATCH_SIZE_JACKET, shuffle=True, seed=SEED + fold_id)
        train_loader_r = create_loader(Xr_train, yr_train, BATCH_SIZE_ROTOR, shuffle=True, seed=SEED + 100 + fold_id)

        model = MultiTaskMLP(input_dim=INPUT_DIM, n_classes_jacket=N_CLASSES_JACKET,
                              n_classes_rotor=N_CLASSES_ROTOR, dropout=MT_DROPOUT).to(DEVICE)

        criterion_j = nn.CrossEntropyLoss(weight=class_weights(yj_train, N_CLASSES_JACKET))
        criterion_r = nn.CrossEntropyLoss(weight=class_weights(yr_train, N_CLASSES_ROTOR))

        optimizer = torch.optim.Adam(model.parameters(), lr=MT_LR, weight_decay=MT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=MT_SCHEDULER_STEP_SIZE, gamma=MT_SCHEDULER_GAMMA)

        for epoch in range(1, MT_EPOCHS + 1):
            model.train()
            epoch_loss, n_steps = 0.0, 0
            iter_j, iter_r = cycle(train_loader_j), cycle(train_loader_r)
            max_steps = max(len(train_loader_j), len(train_loader_r))

            for _ in range(max_steps):
                xb_j, yb_j = next(iter_j)
                xb_r, yb_r = next(iter_r)
                xb_j, yb_j = xb_j.to(DEVICE), yb_j.to(DEVICE)
                xb_r, yb_r = xb_r.to(DEVICE), yb_r.to(DEVICE)

                optimizer.zero_grad()
                loss_j = criterion_j(model(xb_j, "jacket"), yb_j)
                loss_r = criterion_r(model(xb_r, "rotor"), yb_r)
                loss = MT_ALPHA_JACKET * loss_j + MT_BETA_ROTOR * loss_r
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()); n_steps += 1

            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == MT_EPOCHS:
                print(f"Fold {fold_id} | Epoch {epoch:03d}/{MT_EPOCHS} | loss={epoch_loss / max(n_steps,1):.6f}")

        # Evaluation ALWAYS uses real test labels.
        yj_true, yj_pred = evaluate_model_single_head(model, Xj_test, yj_test, task="jacket")
        yr_true, yr_pred = evaluate_model_single_head(model, Xr_test, yr_test, task="rotor")

        row_j = compute_metrics_row(yj_true, yj_pred, model=tag, dataset="jacket",
                                     representation=REPRESENTATIONS[rep_key]["name"], fold_id=fold_id)
        row_r = compute_metrics_row(yr_true, yr_pred, model=tag, dataset="rotor",
                                     representation=REPRESENTATIONS[rep_key]["name"], fold_id=fold_id)
        fold_rows_j.append(row_j); fold_rows_r.append(row_r)
        all_rows.extend([row_j, row_r])
        y_true_oof_j.append(yj_true); y_pred_oof_j.append(yj_pred)
        y_true_oof_r.append(yr_true); y_pred_oof_r.append(yr_pred)

        torch.save(model.state_dict(), models_dir / f"{tag}_{rep_key}_fold{fold_id}.pt")
        del model, optimizer, scheduler, Xj_train, Xj_test, Xr_train, Xr_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_j = save_classification_outputs(rep_out, tag, "jacket", rep_key, y_true_oof_j, y_pred_oof_j, fold_rows_j, class_names_j)
    global_r = save_classification_outputs(rep_out, tag, "rotor", rep_key, y_true_oof_r, y_pred_oof_r, fold_rows_r, class_names_r)

    df_global = pd.DataFrame([global_j, global_r])
    df_global.to_csv(rep_out / f"{tag}_{rep_key}_metrics_global_oof.csv", index=False, encoding="utf-8-sig")
    return df_global


def run_all_multitask_negcontrol() -> pd.DataFrame:
    print_header("EXPERIMENT [2] - MULTITASK NEGATIVE CONTROL (Reviewer 3, Comment 8)")
    dfs = []
    for rep_key in ["fft", "stft", "welch"]:
        for shuffle_task in ["jacket", "rotor"]:
            dfs.append(train_multitask_negcontrol_case(rep_key, shuffle_task))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT_MT_NEGCTRL / "metrics_global_multitask_negcontrol_all.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    set_global_seed(SEED)

    if not CV_ROOT.exists():
        raise FileNotFoundError(
            f"Fold directory not found:\n{CV_ROOT}\n"
            "Make sure 01_crear_datasets_representaciones_cv5.py has already been run."
        )

    OUT_REVISION.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_summaries = []

    if RUN_TABNET_WEIGHTED:
        df_tabnet_w = run_all_tabnet_weighted()
        df_tabnet_w["family"] = "TabNet_weighted"
        all_summaries.append(df_tabnet_w)

    if RUN_MULTITASK_NEGCONTROL:
        df_negctrl = run_all_multitask_negcontrol()
        df_negctrl["family"] = "Multitask_negcontrol"
        all_summaries.append(df_negctrl)

    if all_summaries:
        df_master = pd.concat(all_summaries, ignore_index=True)
        df_master.to_csv(OUT_REVISION / "resumen_robustness_checks_seccion_3_6.csv",
                          index=False, encoding="utf-8-sig")

        elapsed = time.time() - t0
        print_header(f"ROBUSTNESS CHECKS (SECTION 3.6) FINISHED IN {elapsed/60:.1f} MINUTES")
        print("\nGlobal summary (accuracy, balanced_accuracy, f1_macro, f1_weighted):")
        print(df_master[["family", "model", "dataset", "representation", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]].to_string(index=False))
        print("\nConsolidated file saved at:")
        print(OUT_REVISION / "resumen_robustness_checks_seccion_3_6.csv")