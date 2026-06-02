02_train_models_and_generate_results.py

Reproducible pipeline to train and evaluate:

1. Single-task TabNet for jacket/rotor and FFT/STFT/Welch.
2. PyTorch multitask model for FFT/STFT/Welch.
3. Per-fold metrics, global OOF metrics, confusion matrices, figures, and final summary.

  pip install numpy pandas scikit-learn matplotlib openpyxl torch pytorch-tabnet
"""

import os
import gc
import sys
import json
import time
import random
import shutil
import platform
import subprocess
from pathlib import Path
from datetime import datetime
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
# 0. IMPORTAR TABNET
# =============================================================================

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    print("pytorch-tabnet no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet"])
    from pytorch_tabnet.tab_model import TabNetClassifier


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

SEED = 42

PROJECT_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold")
REPRO_ROOT = PROJECT_ROOT / "reproducible"

CV_ROOT = REPRO_ROOT / "04_cv5_topk4096"

OUT_TABNET = REPRO_ROOT / "05_resultados_tabnet"
OUT_MULTITASK = REPRO_ROOT / "06_resultados_multitask"
OUT_COMPARISON = REPRO_ROOT / "07_resultados_comparativos"
OUT_LOGS = REPRO_ROOT / "99_logs"

OVERWRITE_RESULTS = True

N_SPLITS = 5
INPUT_DIM = 4096
N_CLASSES_JACKET = 5
N_CLASSES_ROTOR = 5

REPRESENTATIONS = {
    "fft": {
        "name": "FFT log-magnitude",
        "folder": "01_FFT_log_magnitude_cv5_topk4096",
    },
    "stft": {
        "name": "STFT log-power",
        "folder": "02_STFT_log_power_cv5_topk4096",
    },
    "welch": {
        "name": "Welch PSD log-power",
        "folder": "03_Welch_PSD_log_power_cv5_topk4096",
    },
}

# Activar/desactivar.
RUN_TABNET = True
RUN_MULTITASK = True

# TabNet.
TABNET_PARAMS = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-4,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": {
        "lr": 2e-2,
        "weight_decay": 1e-5,
    },
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {
        "step_size": 40,
        "gamma": 0.7,
    },
    "mask_type": "entmax",
    "seed": SEED,
    "verbose": 10,
}
TABNET_MAX_EPOCHS = 120
TABNET_PATIENCE = 0
TABNET_BATCH_SIZE = 512
TABNET_VIRTUAL_BATCH_SIZE = 128
TABNET_NUM_WORKERS = 0
TABNET_DROP_LAST = False

# Multitask.
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


# =============================================================================
# 2. REPRODUCIBILIDAD Y DISPOSITIVO
# =============================================================================

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
TABNET_PARAMS["device_name"] = DEVICE_NAME


# =============================================================================
# 3. ESTILO DE FIGURAS
# =============================================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})

PASTEL_COLORS = [
    "#8ecae6",
    "#bde0fe",
    "#a8dadc",
    "#cdb4db",
    "#ffc8dd",
    "#ffafcc",
    "#ffd6a5",
    "#fdffb6",
    "#caffbf",
    "#9bf6ff",
]


# =============================================================================
# 4. UTILIDADES
# =============================================================================

def print_header(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def print_subheader(text: str) -> None:
    print("\n" + "-" * 100)
    print(text)
    print("-" * 100)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def setup_output_dirs() -> None:
    if OVERWRITE_RESULTS:
        for d in [OUT_TABNET, OUT_MULTITASK, OUT_COMPARISON]:
            if d.exists():
                shutil.rmtree(d)

    for d in [OUT_TABNET, OUT_MULTITASK, OUT_COMPARISON, OUT_LOGS]:
        d.mkdir(parents=True, exist_ok=True)


def get_environment_info() -> dict:
    return {
        "created_at": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(DEVICE),
        "seed": SEED,
    }


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
        raise ValueError(f"{name}: X_train no tiene {INPUT_DIM} características.")
    if X_test.shape[1] != INPUT_DIM:
        raise ValueError(f"{name}: X_test no tiene {INPUT_DIM} características.")
    if len(X_train) != len(y_train):
        raise ValueError(f"{name}: X_train/y_train no coinciden.")
    if len(X_test) != len(y_test):
        raise ValueError(f"{name}: X_test/y_test no coinciden.")
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError(f"{name}: hay NaN.")
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        raise ValueError(f"{name}: hay Inf.")


def fold_file(dataset: str, rep_key: str, fold_id: int) -> Path:
    folder = CV_ROOT / REPRESENTATIONS[rep_key]["folder"]
    path = folder / f"{dataset}_{rep_key}_fold{fold_id}_topk4096.npz"

    if not path.exists():
        # Búsqueda flexible.
        candidates = sorted(folder.glob(f"{dataset}*{rep_key}*fold{fold_id}*topk4096*.npz"))
        if not candidates:
            candidates = sorted(folder.glob(f"{dataset}*fold{fold_id}*topk4096*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No se encontró fold file para dataset={dataset}, rep={rep_key}, fold={fold_id} en {folder}")
        path = candidates[0]

    return path


def load_fold(dataset: str, rep_key: str, fold_id: int) -> dict:
    path = fold_file(dataset, rep_key, fold_id)
    data = np.load(path, allow_pickle=True)

    required = ["X_train", "X_test", "y_train", "y_test", "class_names"]
    for key in required:
        if key not in data.files:
            raise KeyError(f"Falta clave {key} en {path}")

    X_train = data["X_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    class_names = np.array([clean_class_name(x) for x in data["class_names"]], dtype=object)

    validate_fold_arrays(f"{dataset}_{rep_key}_fold{fold_id}", X_train, X_test, y_train, y_test)

    return {
        "path": path,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_names": class_names,
    }


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
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = max(float(np.max(cm_plot)), 1.0)

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax if normalize else vmax * 1.25)
    ax.set_title(title, color="black", pad=18)
    ax.set_xlabel("Predicted label", color="black", labelpad=12)
    ax.set_ylabel("True label", color="black", labelpad=12)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", color="black")
    ax.set_yticklabels(class_names, color="black")

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            if normalize:
                text_value = format(cm_plot[i, j], fmt)
            else:
                text_value = format(int(cm_plot[i, j]), fmt)

            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="black")

    plt.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


def save_classification_outputs(
    out_dir: Path,
    model_name: str,
    dataset: str,
    rep_key: str,
    y_true_all: list,
    y_pred_all: list,
    fold_rows: list,
    class_names,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.concatenate(y_true_all).astype(np.int64)
    y_pred = np.concatenate(y_pred_all).astype(np.int64)

    representation = REPRESENTATIONS[rep_key]["name"]

    global_row = compute_metrics_row(
        y_true,
        y_pred,
        model=model_name,
        dataset=dataset,
        representation=representation,
        fold_id=None,
    )

    df_folds = pd.DataFrame(fold_rows)
    df_global = pd.DataFrame([global_row])

    df_folds.to_csv(out_dir / f"{model_name}_{dataset}_{rep_key}_metrics_per_fold.csv", index=False, encoding="utf-8-sig")
    df_global.to_csv(out_dir / f"{model_name}_{dataset}_{rep_key}_metrics_global_oof.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    pd.DataFrame(
        cm,
        index=[f"True {c}" for c in class_names],
        columns=[f"Pred {c}" for c in class_names],
    ).to_csv(out_dir / f"{model_name}_{dataset}_{rep_key}_confusion_matrix_global.csv", encoding="utf-8-sig")

    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_class": [class_names[i] for i in y_true],
        "predicted_class": [class_names[i] for i in y_pred],
    }).to_csv(out_dir / f"{model_name}_{dataset}_{rep_key}_oof_predictions.csv", index=False, encoding="utf-8-sig")

    report_txt = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        zero_division=0,
    )
    (out_dir / f"{model_name}_{dataset}_{rep_key}_classification_report.txt").write_text(report_txt, encoding="utf-8")

    plot_confusion_matrix(
        cm,
        class_names,
        title=f"Global Confusion Matrix - {model_name} - {dataset.capitalize()} - {representation}",
        out_base=out_dir / f"{model_name}_{dataset}_{rep_key}_confusion_matrix_global",
        normalize=False,
    )

    plot_confusion_matrix(
        cm,
        class_names,
        title=f"Normalized Confusion Matrix - {model_name} - {dataset.capitalize()} - {representation}",
        out_base=out_dir / f"{model_name}_{dataset}_{rep_key}_confusion_matrix_global_normalized",
        normalize=True,
    )

    return global_row


# =============================================================================
# 5. TABNET SINGLE-TASK
# =============================================================================

def train_tabnet_single_case(dataset: str, rep_key: str) -> dict:
    print_header(f"TABNET SINGLE-TASK - {dataset.upper()} - {REPRESENTATIONS[rep_key]['name']}")

    model_name = "Single-task TabNet"
    case_out = OUT_TABNET / f"{dataset}_{rep_key}"
    models_dir = case_out / "modelos"
    models_dir.mkdir(parents=True, exist_ok=True)

    y_true_all = []
    y_pred_all = []
    fold_rows = []
    class_names_ref = None

    for fold_id in range(1, N_SPLITS + 1):
        print_subheader(f"TabNet {dataset}_{rep_key} fold {fold_id}")
        fold = load_fold(dataset, rep_key, fold_id)

        X_train = fold["X_train"]
        X_test = fold["X_test"]
        y_train = fold["y_train"]
        y_test = fold["y_test"]
        class_names = fold["class_names"]
        class_names_ref = class_names

        clf = TabNetClassifier(**TABNET_PARAMS)

        clf.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_test, y_test)],
            eval_name=["test"],
            eval_metric=["accuracy"],
            max_epochs=TABNET_MAX_EPOCHS,
            patience=TABNET_PATIENCE,
            batch_size=TABNET_BATCH_SIZE,
            virtual_batch_size=TABNET_VIRTUAL_BATCH_SIZE,
            num_workers=TABNET_NUM_WORKERS,
            drop_last=TABNET_DROP_LAST,
        )

        y_pred = clf.predict(X_test).astype(np.int64)

        fold_row = compute_metrics_row(
            y_test,
            y_pred,
            model=model_name,
            dataset=dataset,
            representation=REPRESENTATIONS[rep_key]["name"],
            fold_id=fold_id,
        )
        fold_rows.append(fold_row)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        clf.save_model(str(models_dir / f"tabnet_{dataset}_{rep_key}_fold{fold_id}"))

        pd.DataFrame([fold_row]).to_csv(
            case_out / f"tabnet_{dataset}_{rep_key}_fold{fold_id}_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del clf, X_train, X_test, y_train, y_test, y_pred
        gc.collect()

    global_row = save_classification_outputs(
        out_dir=case_out,
        model_name="TabNet",
        dataset=dataset,
        rep_key=rep_key,
        y_true_all=y_true_all,
        y_pred_all=y_pred_all,
        fold_rows=fold_rows,
        class_names=class_names_ref,
    )

    return global_row


def run_all_tabnet() -> pd.DataFrame:
    print_header("PASO 1 - ENTRENAR TABNET SINGLE-TASK")

    rows = []

    for rep_key in ["fft", "stft", "welch"]:
        for dataset in ["jacket", "rotor"]:
            row = train_tabnet_single_case(dataset, rep_key)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABNET / "metrics_global_tabnet_all_cases.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_TABNET / "metrics_global_tabnet_all_cases.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="TabNet_Global_OOF", index=False)

    return df


# =============================================================================
# 6. MULTITASK PYTORCH
# =============================================================================

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim=4096, n_classes_jacket=5, n_classes_rotor=5, dropout=0.25):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.head_jacket = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes_jacket),
        )

        self.head_rotor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes_rotor),
        )

    def forward(self, x, task: str):
        h = self.shared(x)
        if task == "jacket":
            return self.head_jacket(h)
        if task == "rotor":
            return self.head_rotor(h)
        raise ValueError(f"Tarea no reconocida: {task}")

    def encode(self, x):
        return self.shared(x)


def create_loader(X, y, batch_size, shuffle=True, seed=42) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (n_classes * counts)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def evaluate_multitask_model(model: MultiTaskMLP, X, y, task: str, batch_size: int = 512):
    model.eval()
    loader = create_loader(X, y, batch_size=batch_size, shuffle=False, seed=SEED)

    preds = []
    trues = []
    latents = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb, task)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            h = model.encode(xb).cpu().numpy()

            preds.append(pred)
            trues.append(yb.numpy())
            latents.append(h)

    return (
        np.concatenate(trues).astype(np.int64),
        np.concatenate(preds).astype(np.int64),
        np.vstack(latents).astype(np.float32),
    )


def train_multitask_representation(rep_key: str) -> pd.DataFrame:
    print_header(f"MULTITASK - {REPRESENTATIONS[rep_key]['name']}")

    rep_out = OUT_MULTITASK / rep_key
    models_dir = rep_out / "modelos"
    latents_dir = rep_out / "latents"
    models_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    y_true_oof_j = []
    y_pred_oof_j = []
    y_true_oof_r = []
    y_pred_oof_r = []

    fold_rows_j = []
    fold_rows_r = []

    class_names_j = None
    class_names_r = None

    for fold_id in range(1, N_SPLITS + 1):
        print_subheader(f"Multitask {rep_key} fold {fold_id}")

        fj = load_fold("jacket", rep_key, fold_id)
        fr = load_fold("rotor", rep_key, fold_id)

        Xj_train, Xj_test = fj["X_train"], fj["X_test"]
        yj_train, yj_test = fj["y_train"], fj["y_test"]
        Xr_train, Xr_test = fr["X_train"], fr["X_test"]
        yr_train, yr_test = fr["y_train"], fr["y_test"]

        class_names_j = fj["class_names"]
        class_names_r = fr["class_names"]

        train_loader_j = create_loader(Xj_train, yj_train, BATCH_SIZE_JACKET, shuffle=True, seed=SEED + fold_id)
        train_loader_r = create_loader(Xr_train, yr_train, BATCH_SIZE_ROTOR, shuffle=True, seed=SEED + 100 + fold_id)

        model = MultiTaskMLP(
            input_dim=INPUT_DIM,
            n_classes_jacket=N_CLASSES_JACKET,
            n_classes_rotor=N_CLASSES_ROTOR,
            dropout=MT_DROPOUT,
        ).to(DEVICE)

        criterion_j = nn.CrossEntropyLoss(weight=class_weights(yj_train, N_CLASSES_JACKET))
        criterion_r = nn.CrossEntropyLoss(weight=class_weights(yr_train, N_CLASSES_ROTOR))

        optimizer = torch.optim.Adam(model.parameters(), lr=MT_LR, weight_decay=MT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=MT_SCHEDULER_STEP_SIZE,
            gamma=MT_SCHEDULER_GAMMA,
        )

        history = []

        for epoch in range(1, MT_EPOCHS + 1):
            model.train()

            epoch_loss = 0.0
            n_steps = 0

            iter_j = cycle(train_loader_j)
            iter_r = cycle(train_loader_r)
            max_steps = max(len(train_loader_j), len(train_loader_r))

            for _ in range(max_steps):
                xb_j, yb_j = next(iter_j)
                xb_r, yb_r = next(iter_r)

                xb_j = xb_j.to(DEVICE)
                yb_j = yb_j.to(DEVICE)
                xb_r = xb_r.to(DEVICE)
                yb_r = yb_r.to(DEVICE)

                optimizer.zero_grad()

                logits_j = model(xb_j, "jacket")
                logits_r = model(xb_r, "rotor")

                loss_j = criterion_j(logits_j, yb_j)
                loss_r = criterion_r(logits_r, yb_r)

                loss = MT_ALPHA_JACKET * loss_j + MT_BETA_ROTOR * loss_r

                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_steps += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_steps, 1)
            history.append({
                "fold_id": fold_id,
                "epoch": epoch,
                "loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if epoch == 1 or epoch % 10 == 0 or epoch == MT_EPOCHS:
                print(f"Fold {fold_id} | Epoch {epoch:03d}/{MT_EPOCHS} | loss={avg_loss:.6f}")

        pd.DataFrame(history).to_csv(
            rep_out / f"multitask_{rep_key}_fold{fold_id}_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        yj_true, yj_pred, Zj = evaluate_multitask_model(model, Xj_test, yj_test, "jacket")
        yr_true, yr_pred, Zr = evaluate_multitask_model(model, Xr_test, yr_test, "rotor")

        row_j = compute_metrics_row(
            yj_true,
            yj_pred,
            model=f"Multitask {REPRESENTATIONS[rep_key]['name']} MLP",
            dataset="jacket",
            representation=REPRESENTATIONS[rep_key]["name"],
            fold_id=fold_id,
        )
        row_r = compute_metrics_row(
            yr_true,
            yr_pred,
            model=f"Multitask {REPRESENTATIONS[rep_key]['name']} MLP",
            dataset="rotor",
            representation=REPRESENTATIONS[rep_key]["name"],
            fold_id=fold_id,
        )

        fold_rows_j.append(row_j)
        fold_rows_r.append(row_r)
        all_rows.extend([row_j, row_r])

        y_true_oof_j.append(yj_true)
        y_pred_oof_j.append(yj_pred)
        y_true_oof_r.append(yr_true)
        y_pred_oof_r.append(yr_pred)

        torch.save(model.state_dict(), models_dir / f"multitask_{rep_key}_fold{fold_id}.pt")

        np.savez_compressed(
            latents_dir / f"multitask_{rep_key}_fold{fold_id}_latents_test.npz",
            Z_jacket=Zj.astype(np.float32),
            y_jacket=yj_true.astype(np.int64),
            pred_jacket=yj_pred.astype(np.int64),
            Z_rotor=Zr.astype(np.float32),
            y_rotor=yr_true.astype(np.int64),
            pred_rotor=yr_pred.astype(np.int64),
        )

        del model, optimizer, scheduler, Xj_train, Xj_test, Xr_train, Xr_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Guardar OOF por dataset.
    global_j = save_classification_outputs(
        out_dir=rep_out,
        model_name="Multitask",
        dataset="jacket",
        rep_key=rep_key,
        y_true_all=y_true_oof_j,
        y_pred_all=y_pred_oof_j,
        fold_rows=fold_rows_j,
        class_names=class_names_j,
    )
    global_r = save_classification_outputs(
        out_dir=rep_out,
        model_name="Multitask",
        dataset="rotor",
        rep_key=rep_key,
        y_true_all=y_true_oof_r,
        y_pred_all=y_pred_oof_r,
        fold_rows=fold_rows_r,
        class_names=class_names_r,
    )

    df_fold = pd.DataFrame(all_rows)
    df_global = pd.DataFrame([global_j, global_r])

    df_fold.to_csv(rep_out / f"multitask_{rep_key}_metrics_per_fold.csv", index=False, encoding="utf-8-sig")
    df_global.to_csv(rep_out / f"multitask_{rep_key}_metrics_global_oof.csv", index=False, encoding="utf-8-sig")

    return df_global


def run_all_multitask() -> pd.DataFrame:
    print_header("PASO 2 - ENTRENAR MODELOS MULTITASK")

    dfs = []
    for rep_key in ["fft", "stft", "welch"]:
        df_rep = train_multitask_representation(rep_key)
        dfs.append(df_rep)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT_MULTITASK / "metrics_global_multitask_all_representations.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_MULTITASK / "metrics_global_multitask_all_representations.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Multitask_Global_OOF", index=False)

    return df


# =============================================================================
# 7. COMPARACIÓN FINAL
# =============================================================================

def build_final_comparison(tabnet_df: pd.DataFrame | None, multitask_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Construye las tablas comparativas finales del artículo.

    Salidas principales:
      - comparison_global_oof_all_models.csv/xlsx:
          Tabla larga con todos los modelos.

      - table_single_task_tabnet_article.csv/xlsx:
          Tabla tipo artículo para modelos single-task TabNet.

      - table_multitask_article.csv/xlsx:
          Tabla tipo artículo para modelos multitask.

      - comparison_single_vs_multitask_pivot.csv/xlsx:
          Tabla comparativa lado a lado:
          Single-task vs Multitask por dataset y representación.

      - comparison_single_vs_multitask_delta.csv/xlsx:
          Diferencias:
          Delta = Multitask - Single-task.

      - resumen_global_oof_all_models.txt:
          Resumen textual completo.
    """

    print_header("PASO 3 - COMPARACIÓN FINAL SINGLE-TASK VS MULTITASK")

    frames = []

    if tabnet_df is not None and len(tabnet_df) > 0:
        df_t = tabnet_df.copy()
        df_t["family"] = "Single-task"
        df_t["model_short"] = "TabNet"
        frames.append(df_t)

    if multitask_df is not None and len(multitask_df) > 0:
        df_m = multitask_df.copy()
        df_m["family"] = "Multitask"
        df_m["model_short"] = "Multitask MLP"
        frames.append(df_m)

    if not frames:
        raise RuntimeError("No hay resultados para comparar.")

    df = pd.concat(frames, ignore_index=True)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "precision_weighted",
        "recall_weighted",
    ]

    preferred_cols = [
        "family",
        "model_short",
        "model",
        "dataset",
        "representation",
        "n_samples",
    ] + metric_cols

    available_cols = [c for c in preferred_cols if c in df.columns]
    df_out = df[available_cols].copy()

    # Orden del artículo.
    rep_order = ["FFT log-magnitude", "STFT log-power", "Welch PSD log-power"]
    dataset_order = ["jacket", "rotor"]
    family_order = ["Single-task", "Multitask"]

    df_out["representation"] = pd.Categorical(
        df_out["representation"],
        categories=rep_order,
        ordered=True
    )
    df_out["dataset"] = pd.Categorical(
        df_out["dataset"],
        categories=dataset_order,
        ordered=True
    )
    df_out["family"] = pd.Categorical(
        df_out["family"],
        categories=family_order,
        ordered=True
    )

    df_out = df_out.sort_values(
        ["family", "representation", "dataset"]
    ).reset_index(drop=True)

    for col in ["representation", "dataset", "family"]:
        df_out[col] = df_out[col].astype(str)

    OUT_COMPARISON.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. Tabla larga completa
    # ========================================================
    long_csv = OUT_COMPARISON / "comparison_global_oof_all_models.csv"
    long_xlsx = OUT_COMPARISON / "comparison_global_oof_all_models.xlsx"

    df_out.to_csv(long_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(long_xlsx, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="All_Global_OOF", index=False)

    # ========================================================
    # 2. Tablas tipo artículo separadas
    # ========================================================
    article_cols = [
        "dataset",
        "representation",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
    ]

    if tabnet_df is not None and len(tabnet_df) > 0:
        tabnet_article = tabnet_df[article_cols].copy()
        tabnet_article = tabnet_article.sort_values(
            ["representation", "dataset"]
        ).reset_index(drop=True)

        tabnet_article.to_csv(
            OUT_COMPARISON / "table_single_task_tabnet_article.csv",
            index=False,
            encoding="utf-8-sig"
        )

        with pd.ExcelWriter(
            OUT_COMPARISON / "table_single_task_tabnet_article.xlsx",
            engine="openpyxl"
        ) as writer:
            tabnet_article.to_excel(writer, sheet_name="Single_task_TabNet", index=False)

    if multitask_df is not None and len(multitask_df) > 0:
        mt_article = multitask_df[article_cols].copy()
        mt_article = mt_article.sort_values(
            ["representation", "dataset"]
        ).reset_index(drop=True)

        mt_article.to_csv(
            OUT_COMPARISON / "table_multitask_article.csv",
            index=False,
            encoding="utf-8-sig"
        )

        with pd.ExcelWriter(
            OUT_COMPARISON / "table_multitask_article.xlsx",
            engine="openpyxl"
        ) as writer:
            mt_article.to_excel(writer, sheet_name="Multitask", index=False)

    # ========================================================
    # 3. Tabla comparativa lado a lado
    # ========================================================
    core = df_out[
        [
            "family",
            "dataset",
            "representation",
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
        ]
    ].copy()

    pivot = core.pivot_table(
        index=["dataset", "representation"],
        columns="family",
        values=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
        aggfunc="first",
        observed=False,
    )

    # Aplanar columnas: accuracy_Single-task, accuracy_Multitask, etc.
    pivot.columns = [f"{metric}_{family}" for metric, family in pivot.columns]
    pivot = pivot.reset_index()

    # Ordenar columnas de forma clara.
    ordered_pivot_cols = ["dataset", "representation"]
    for m in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        for fam in ["Single-task", "Multitask"]:
            col = f"{m}_{fam}"
            if col in pivot.columns:
                ordered_pivot_cols.append(col)

    pivot = pivot[ordered_pivot_cols]

    # Deltas Multitask - Single-task.
    delta = pivot[["dataset", "representation"]].copy()

    for m in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        col_single = f"{m}_Single-task"
        col_multi = f"{m}_Multitask"
        if col_single in pivot.columns and col_multi in pivot.columns:
            delta[f"delta_{m}_Multitask_minus_Single"] = pivot[col_multi] - pivot[col_single]

    pivot.to_csv(
        OUT_COMPARISON / "comparison_single_vs_multitask_pivot.csv",
        index=False,
        encoding="utf-8-sig"
    )

    delta.to_csv(
        OUT_COMPARISON / "comparison_single_vs_multitask_delta.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with pd.ExcelWriter(
        OUT_COMPARISON / "comparison_single_vs_multitask_pivot.xlsx",
        engine="openpyxl"
    ) as writer:
        pivot.to_excel(writer, sheet_name="Single_vs_Multitask", index=False)
        delta.to_excel(writer, sheet_name="Deltas", index=False)

    # ========================================================
    # 4. Tabla resumida lista para el artículo
    # ========================================================
    comparison_article = pivot.copy()

    rename_map = {
        "dataset": "Dataset",
        "representation": "Representation",
        "accuracy_Single-task": "Single-task Accuracy",
        "accuracy_Multitask": "Multitask Accuracy",
        "balanced_accuracy_Single-task": "Single-task Balanced Accuracy",
        "balanced_accuracy_Multitask": "Multitask Balanced Accuracy",
        "f1_macro_Single-task": "Single-task Macro-F1",
        "f1_macro_Multitask": "Multitask Macro-F1",
        "f1_weighted_Single-task": "Single-task Weighted-F1",
        "f1_weighted_Multitask": "Multitask Weighted-F1",
    }

    comparison_article = comparison_article.rename(columns=rename_map)

    comparison_article.to_csv(
        OUT_COMPARISON / "table_comparison_single_vs_multitask_article.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with pd.ExcelWriter(
        OUT_COMPARISON / "table_comparison_single_vs_multitask_article.xlsx",
        engine="openpyxl"
    ) as writer:
        comparison_article.to_excel(writer, sheet_name="Article_Comparison", index=False)

    # ========================================================
    # 5. Resumen TXT
    # ========================================================
    summary_text = []
    summary_text.append("RESUMEN GLOBAL OOF - ARTÍCULO MULTITASK")
    summary_text.append("=" * 100)
    summary_text.append("")
    summary_text.append("1. Tabla larga global")
    summary_text.append("-" * 100)
    summary_text.append(df_out.to_string(index=False))
    summary_text.append("")
    summary_text.append("2. Comparación Single-task vs Multitask")
    summary_text.append("-" * 100)
    summary_text.append(pivot.to_string(index=False))
    summary_text.append("")
    summary_text.append("3. Deltas Multitask - Single-task")
    summary_text.append("-" * 100)
    summary_text.append(delta.to_string(index=False))
    summary_text.append("")
    summary_text.append("Notas:")
    summary_text.append("- Todas las métricas son globales OOF agregadas sobre los 5 folds.")
    summary_text.append("- Top-K y StandardScaler fueron ajustados únicamente con el train de cada fold.")
    summary_text.append("- Para rotor, los folds respetan grupos experimentales.")
    summary_text.append("- Para jacket, cada muestra se trata como grupo independiente.")
    summary_text.append("")
    summary_text.append("Archivos principales generados:")
    summary_text.append(str(long_csv))
    summary_text.append(str(long_xlsx))
    summary_text.append(str(OUT_COMPARISON / "comparison_single_vs_multitask_pivot.csv"))
    summary_text.append(str(OUT_COMPARISON / "comparison_single_vs_multitask_delta.csv"))
    summary_text.append(str(OUT_COMPARISON / "table_comparison_single_vs_multitask_article.csv"))

    (OUT_COMPARISON / "resumen_global_oof_all_models.txt").write_text(
        "\n".join(summary_text),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("TABLA COMPARATIVA SINGLE-TASK VS MULTITASK")
    print("=" * 100)
    print(pivot.to_string(index=False))

    print("\n" + "=" * 100)
    print("DELTAS MULTITASK - SINGLE-TASK")
    print("=" * 100)
    print(delta.to_string(index=False))

    return df_out


def write_run_config() -> None:
    config = {
        "created_at": datetime.now().isoformat(),
        "environment": get_environment_info(),
        "paths": {
            "repro_root": str(REPRO_ROOT),
            "cv_root": str(CV_ROOT),
            "out_tabnet": str(OUT_TABNET),
            "out_multitask": str(OUT_MULTITASK),
            "out_comparison": str(OUT_COMPARISON),
        },
        "tabnet": {
            "run": RUN_TABNET,
            "params": {
                "n_d": 32,
                "n_a": 32,
                "n_steps": 5,
                "gamma": 1.5,
                "lambda_sparse": 1e-4,
                "optimizer": "Adam",
                "lr": 2e-2,
                "weight_decay": 1e-5,
                "scheduler": "StepLR",
                "scheduler_step_size": 40,
                "scheduler_gamma": 0.7,
                "mask_type": "entmax",
                "max_epochs": TABNET_MAX_EPOCHS,
                "batch_size": TABNET_BATCH_SIZE,
                "virtual_batch_size": TABNET_VIRTUAL_BATCH_SIZE,
            },
        },
        "multitask": {
            "run": RUN_MULTITASK,
            "framework": "PyTorch",
            "architecture": {
                "input_dim": INPUT_DIM,
                "shared_encoder": [1024, 256, 64],
                "head_hidden": 32,
                "n_classes_jacket": N_CLASSES_JACKET,
                "n_classes_rotor": N_CLASSES_ROTOR,
                "dropout": MT_DROPOUT,
            },
            "training": {
                "epochs": MT_EPOCHS,
                "lr": MT_LR,
                "weight_decay": MT_WEIGHT_DECAY,
                "batch_size_jacket": BATCH_SIZE_JACKET,
                "batch_size_rotor": BATCH_SIZE_ROTOR,
                "scheduler": "StepLR",
                "scheduler_step_size": MT_SCHEDULER_STEP_SIZE,
                "scheduler_gamma": MT_SCHEDULER_GAMMA,
                "loss": "Weighted cross-entropy",
                "alpha_jacket": MT_ALPHA_JACKET,
                "beta_rotor": MT_BETA_ROTOR,
            },
        },
    }

    save_json(OUT_LOGS / "script02_run_configuration.json", config)


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    print_header("INICIO SCRIPT 02 - MODELOS + RESULTADOS")
    print("REPRO_ROOT:", REPRO_ROOT)
    print("DEVICE:", DEVICE)

    setup_output_dirs()
    write_run_config()

    if not CV_ROOT.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de folds:\n{CV_ROOT}\n"
            "Primero ejecuta 01_crear_datasets_representaciones_cv5.py"
        )

    tabnet_df = None
    multitask_df = None

    start = time.time()

    if RUN_TABNET:
        tabnet_df = run_all_tabnet()

    if RUN_MULTITASK:
        multitask_df = run_all_multitask()

    comparison_df = build_final_comparison(tabnet_df, multitask_df)

    elapsed = time.time() - start

    final_summary = {
        "status": "ok",
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": float(elapsed),
        "comparison_csv": str(OUT_COMPARISON / "comparison_global_oof_all_models.csv"),
        "tabnet_ran": RUN_TABNET,
        "multitask_ran": RUN_MULTITASK,
    }
    save_json(OUT_LOGS / "script02_final_summary.json", final_summary)

    print_header("SCRIPT 02 TERMINADO CORRECTAMENTE")
    print("Resultados comparativos:")
    print(OUT_COMPARISON / "comparison_global_oof_all_models.csv")
    print()
    print(comparison_df)


if __name__ == "__main__":
    main()
