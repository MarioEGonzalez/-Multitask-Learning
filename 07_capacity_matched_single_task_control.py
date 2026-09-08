# -*- coding: utf-8 -*-
r"""
03_capacity_matched_single_task_control.py

Control single-task con arquitectura/capacidad esencialmente equivalente al encoder
del modelo multitarea, diseñado específicamente para responder al Comentario 7
del revisor.

Objetivo:
- Entrenar por separado Jacket y Rotor.
- Usar EXACTAMENTE los folds CV5 ya preprocesados.
- Usar las mismas 4096 features Top-K ya seleccionadas por fold.
- Usar los mismos datos ya estandarizados por fold.
- Mantener la arquitectura del encoder:
    4096 -> 1024 -> 256 -> 64
  con BatchNorm + ReLU, Dropout=0.25 en las dos primeras capas.
- Usar la misma cabeza de clasificación:
    64 -> 32 -> 5
- Weighted CrossEntropy con la misma fórmula.
- Adam lr=1e-3, weight_decay=1e-5.
- StepLR step_size=40, gamma=0.7.
- 120 epochs, sin early stopping.
- Batch size Jacket=256, Rotor=128.
- Mantener el mismo número de optimizer steps por epoch que el esquema multitarea
  original, usando max(len(loader_jacket), len(loader_rotor)).
  Esto evita favorecer al control Rotor con un presupuesto de actualización distinto.
- Comparar fold a fold contra los resultados multitarea ya existentes.

Entrada:
C:\Users\Usuario\Documents\5. Multi Task k fold\reproducible\04_cv5_topk4096

Resultados multitarea existentes:
C:\Users\Usuario\Documents\5. Multi Task k fold\reproducible\06_resultados_multitask

Nueva salida:
C:\Users\Usuario\Documents\5. Multi Task k fold\reproducible\08_capacity_matched_single_task

IMPORTANTE:
Este script NO regenera Top-K, StandardScaler ni folds. Consume exactamente
los archivos creados previamente, garantizando comparación sobre folds idénticos.

Uso desde JupyterLab:
%run "C:/Users/Usuario/Documents/5. Multi Task k fold/reproducible/01_scripts/03_capacity_matched_single_task_control.py"
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

import os
import gc
import json
import math
import random
import platform
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
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

SEED = 42

PROJECT_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold")
REPRO_ROOT = PROJECT_ROOT / "reproducible"

CV_ROOT = REPRO_ROOT / "04_cv5_topk4096"
MT_ROOT = REPRO_ROOT / "06_resultados_multitask"

OUT_ROOT = REPRO_ROOT / "08_capacity_matched_single_task"
OUT_MODELS = OUT_ROOT / "models"
OUT_HISTORIES = OUT_ROOT / "histories"
OUT_PREDICTIONS = OUT_ROOT / "predictions"
OUT_METRICS = OUT_ROOT / "metrics"
OUT_COMPARISON = OUT_ROOT / "comparison_with_multitask"
OUT_FIGURES = OUT_ROOT / "figures"
OUT_LOGS = OUT_ROOT / "logs"

N_SPLITS = 5
INPUT_DIM = 4096
N_CLASSES = 5

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

DATASETS = ["jacket", "rotor"]

# EXACTAMENTE como el multitask original.
BATCH_SIZE = {
    "jacket": 256,
    "rotor": 128,
}

EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25
SCHEDULER_STEP_SIZE = 40
SCHEDULER_GAMMA = 0.7

# True = igualar el número de optimizer updates/epoch al multitask original.
MATCH_MULTITASK_UPDATE_BUDGET = True

# No borrar resultados originales multitask.
OVERWRITE_CONTROL_OUTPUT = True


# =============================================================================
# 2. REPRODUCIBILIDAD
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


# =============================================================================
# 3. DIRECTORIOS
# =============================================================================

def setup_dirs() -> None:
    dirs = [
        OUT_ROOT,
        OUT_MODELS,
        OUT_HISTORIES,
        OUT_PREDICTIONS,
        OUT_METRICS,
        OUT_COMPARISON,
        OUT_FIGURES,
        OUT_LOGS,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 4. UTILIDADES
# =============================================================================

def print_header(text: str) -> None:
    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)


def print_subheader(text: str) -> None:
    print("\n" + "-" * 110)
    print(text)
    print("-" * 110)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def environment_info() -> dict:
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


# =============================================================================
# 5. LOCALIZACIÓN Y CARGA DE FOLDS EXISTENTES
# =============================================================================

def fold_file(dataset: str, rep_key: str, fold_id: int) -> Path:
    folder = CV_ROOT / REPRESENTATIONS[rep_key]["folder"]

    exact = folder / f"{dataset}_{rep_key}_fold{fold_id}_topk4096.npz"

    if exact.exists():
        return exact

    candidates = sorted(
        folder.glob(
            f"{dataset}*{rep_key}*fold{fold_id}*topk4096*.npz"
        )
    )

    if not candidates:
        candidates = sorted(
            folder.glob(
                f"{dataset}*fold{fold_id}*topk4096*.npz"
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"No se encontró fold para dataset={dataset}, "
            f"rep={rep_key}, fold={fold_id}\n"
            f"Carpeta buscada:\n{folder}"
        )

    return candidates[0]


def load_fold(dataset: str, rep_key: str, fold_id: int) -> dict:
    path = fold_file(dataset, rep_key, fold_id)

    data = np.load(path, allow_pickle=True)

    required = [
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "class_names",
    ]

    for key in required:
        if key not in data.files:
            raise KeyError(
                f"Falta la clave '{key}' en:\n{path}"
            )

    X_train = data["X_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)

    class_names = np.array(
        [
            clean_class_name(x)
            for x in data["class_names"]
        ],
        dtype=object,
    )

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"{path.name}: X_train/X_test deben ser matrices 2D."
        )

    if X_train.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{path.name}: X_train tiene {X_train.shape[1]} "
            f"features; se esperaban {INPUT_DIM}."
        )

    if X_test.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{path.name}: X_test tiene {X_test.shape[1]} "
            f"features; se esperaban {INPUT_DIM}."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            f"{path.name}: X_train/y_train no coinciden."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"{path.name}: X_test/y_test no coinciden."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(
            f"{path.name}: X_train contiene NaN o Inf."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            f"{path.name}: X_test contiene NaN o Inf."
        )

    return {
        "path": path,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_names": class_names,
    }


# =============================================================================
# 6. ARQUITECTURAS
# =============================================================================

class CapacityMatchedSingleTaskMLP(nn.Module):
    """
    Single-task counterpart del modelo multitarea.

    Encoder idéntico:
        4096 -> 1024 -> 256 -> 64

    Head idéntico a cada head multitarea:
        64 -> 32 -> 5
    """

    def __init__(
        self,
        input_dim: int = 4096,
        n_classes: int = 5,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
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

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)

    def encode(self, x):
        return self.encoder(x)


class MultiTaskArchitectureForAudit(nn.Module):
    """
    Solo para auditar número de parámetros.
    No se entrena en este script.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        n_classes_jacket: int = 5,
        n_classes_rotor: int = 5,
        dropout: float = 0.25,
    ):
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


def count_parameters(model: nn.Module) -> int:
    return int(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


def parameter_audit() -> dict:
    st_model = CapacityMatchedSingleTaskMLP(
        INPUT_DIM,
        N_CLASSES,
        DROPOUT,
    )

    mt_model = MultiTaskArchitectureForAudit(
        INPUT_DIM,
        N_CLASSES,
        N_CLASSES,
        DROPOUT,
    )

    st_total = count_parameters(st_model)
    mt_total = count_parameters(mt_model)

    difference = mt_total - st_total
    difference_pct_vs_mt = (
        100.0 * difference / mt_total
    )

    result = {
        "single_task_total_parameters": st_total,
        "multitask_total_parameters": mt_total,
        "absolute_difference": difference,
        "difference_percent_of_multitask": difference_pct_vs_mt,
        "single_task_percent_of_multitask": (
            100.0 * st_total / mt_total
        ),
        "note": (
            "The encoder and active classification head are identical. "
            "The only parameter-count difference is the second task head "
            "present in the multitask network but absent in a genuine "
            "single-task model."
        ),
    }

    del st_model, mt_model

    return result


# =============================================================================
# 7. DATALOADER Y PESOS DE CLASE
# =============================================================================

def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def class_weights(
    y: np.ndarray,
    n_classes: int = 5,
) -> torch.Tensor:
    """
    Misma fórmula del multitask:
        w_c = n / (K * n_c)
    """

    counts = np.bincount(
        y,
        minlength=n_classes,
    ).astype(np.float32)

    total = float(
        counts.sum()
    )

    weights = total / (
        n_classes * counts
    )

    weights = np.nan_to_num(
        weights,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


# =============================================================================
# 8. MÉTRICAS
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# =============================================================================
# 9. EVALUACIÓN
# =============================================================================

def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
):

    model.eval()

    loader = create_loader(
        X,
        y,
        batch_size=batch_size,
        shuffle=False,
        seed=SEED,
    )

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(DEVICE)

            logits = model(xb)

            pred = torch.argmax(
                logits,
                dim=1,
            ).cpu().numpy()

            y_true_all.append(
                yb.numpy()
            )

            y_pred_all.append(
                pred
            )

    y_true = np.concatenate(
        y_true_all
    ).astype(np.int64)

    y_pred = np.concatenate(
        y_pred_all
    ).astype(np.int64)

    return y_true, y_pred


# =============================================================================
# 10. PRESUPUESTO DE ACTUALIZACIONES DEL MULTITASK
# =============================================================================

def multitask_steps_per_epoch(
    rep_key: str,
    fold_id: int,
) -> dict:
    """
    Calcula exactamente cuántos mini-batch steps hacía el multitask original:
        max(len(loader_jacket), len(loader_rotor))
    """

    fj = load_fold(
        "jacket",
        rep_key,
        fold_id,
    )

    fr = load_fold(
        "rotor",
        rep_key,
        fold_id,
    )

    n_j = len(
        fj["X_train"]
    )

    n_r = len(
        fr["X_train"]
    )

    steps_j = math.ceil(
        n_j / BATCH_SIZE["jacket"]
    )

    steps_r = math.ceil(
        n_r / BATCH_SIZE["rotor"]
    )

    max_steps = max(
        steps_j,
        steps_r,
    )

    del fj, fr

    return {
        "n_train_jacket": n_j,
        "n_train_rotor": n_r,
        "steps_jacket": steps_j,
        "steps_rotor": steps_r,
        "multitask_max_steps": max_steps,
    }


# =============================================================================
# 11. ENTRENAMIENTO SINGLE-TASK MATCHED
# =============================================================================

def train_one_case(
    dataset: str,
    rep_key: str,
    fold_id: int,
) -> dict:

    print_subheader(
        f"CONTROL SINGLE-TASK MATCHED | "
        f"{dataset.upper()} | "
        f"{REPRESENTATIONS[rep_key]['name']} | "
        f"Fold {fold_id}"
    )

    fold = load_fold(
        dataset,
        rep_key,
        fold_id,
    )

    X_train = fold["X_train"]
    X_test = fold["X_test"]
    y_train = fold["y_train"]
    y_test = fold["y_test"]
    class_names = fold["class_names"]

    # --------------------------------------------------------
    # Presupuesto de steps idéntico al multitask original
    # --------------------------------------------------------

    budget = multitask_steps_per_epoch(
        rep_key,
        fold_id,
    )

    target_loader_seed = (
        SEED + fold_id
        if dataset == "jacket"
        else SEED + 100 + fold_id
    )

    train_loader = create_loader(
        X_train,
        y_train,
        batch_size=BATCH_SIZE[dataset],
        shuffle=True,
        seed=target_loader_seed,
    )

    natural_steps = len(
        train_loader
    )

    if MATCH_MULTITASK_UPDATE_BUDGET:
        steps_per_epoch = budget[
            "multitask_max_steps"
        ]
    else:
        steps_per_epoch = natural_steps

    # --------------------------------------------------------
    # Reinicialización determinista del control
    # --------------------------------------------------------

    set_seed(SEED)

    model = CapacityMatchedSingleTaskMLP(
        input_dim=INPUT_DIM,
        n_classes=N_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    weights = class_weights(
        y_train,
        N_CLASSES,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA,
    )

    history = []

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        iterator = cycle(
            train_loader
        )

        epoch_loss = 0.0

        for _ in range(
            steps_per_epoch
        ):

            xb, yb = next(
                iterator
            )

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += float(
                loss.item()
            )

        scheduler.step()

        avg_loss = (
            epoch_loss
            / steps_per_epoch
        )

        history.append({
            "dataset": dataset,
            "representation": rep_key,
            "fold_id": fold_id,
            "epoch": epoch,
            "loss": avg_loss,
            "lr": float(
                optimizer.param_groups[0]["lr"]
            ),
            "natural_steps": natural_steps,
            "matched_steps": steps_per_epoch,
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"loss={avg_loss:.6f} | "
                f"steps={steps_per_epoch}"
            )

    # --------------------------------------------------------
    # Guardar historia
    # --------------------------------------------------------

    history_path = (
        OUT_HISTORIES
        / f"capacity_matched_{dataset}_{rep_key}_fold{fold_id}_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Evaluar fold
    # --------------------------------------------------------

    y_true, y_pred = evaluate_model(
        model,
        X_test,
        y_test,
        batch_size=512,
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    row = {
        "model": "Capacity-matched single-task MLP",
        "dataset": dataset,
        "representation_key": rep_key,
        "representation": REPRESENTATIONS[rep_key]["name"],
        "fold_id": fold_id,
        "n_train": int(
            len(y_train)
        ),
        "n_test": int(
            len(y_test)
        ),
        "batch_size": int(
            BATCH_SIZE[dataset]
        ),
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "StepLR",
        "scheduler_step_size": SCHEDULER_STEP_SIZE,
        "scheduler_gamma": SCHEDULER_GAMMA,
        "dropout": DROPOUT,
        "loss": "Weighted cross-entropy",
        "natural_steps_per_epoch": int(
            natural_steps
        ),
        "matched_steps_per_epoch": int(
            steps_per_epoch
        ),
        "multitask_steps_jacket": int(
            budget["steps_jacket"]
        ),
        "multitask_steps_rotor": int(
            budget["steps_rotor"]
        ),
        "multitask_max_steps": int(
            budget["multitask_max_steps"]
        ),
        **metrics,
    }

    # --------------------------------------------------------
    # Guardar predicciones
    # --------------------------------------------------------

    pred_path = (
        OUT_PREDICTIONS
        / f"capacity_matched_{dataset}_{rep_key}_fold{fold_id}_predictions.csv"
    )

    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_class": [
            class_names[i]
            for i in y_true
        ],
        "predicted_class": [
            class_names[i]
            for i in y_pred
        ],
    }).to_csv(
        pred_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Guardar modelo
    # --------------------------------------------------------

    model_path = (
        OUT_MODELS
        / f"capacity_matched_{dataset}_{rep_key}_fold{fold_id}.pt"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    # --------------------------------------------------------
    # Reporte clasificación
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(
            len(class_names)
        ),
        target_names=list(
            class_names
        ),
        zero_division=0,
    )

    report_path = (
        OUT_METRICS
        / f"capacity_matched_{dataset}_{rep_key}_fold{fold_id}_classification_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"Fold {fold_id} | "
        f"Accuracy={metrics['accuracy']:.4f} | "
        f"BA={metrics['balanced_accuracy']:.4f} | "
        f"Macro-F1={metrics['f1_macro']:.4f}"
    )

    del (
        model,
        optimizer,
        scheduler,
        criterion,
        weights,
        train_loader,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "row": row,
        "y_true": y_true,
        "y_pred": y_pred,
        "class_names": class_names,
    }


# =============================================================================
# 12. GLOBAL OOF DEL CONTROL
# =============================================================================

def summarize_global_oof(
    all_fold_results: list,
) -> pd.DataFrame:

    rows = []

    for rep_key in REPRESENTATIONS:
        for dataset in DATASETS:

            selected = [
                r
                for r in all_fold_results
                if (
                    r["row"]["dataset"] == dataset
                    and
                    r["row"]["representation_key"] == rep_key
                )
            ]

            if len(selected) != N_SPLITS:
                raise ValueError(
                    f"Se esperaban {N_SPLITS} folds para "
                    f"{dataset}-{rep_key}; encontrados {len(selected)}."
                )

            y_true = np.concatenate(
                [
                    r["y_true"]
                    for r in selected
                ]
            )

            y_pred = np.concatenate(
                [
                    r["y_pred"]
                    for r in selected
                ]
            )

            metrics = compute_metrics(
                y_true,
                y_pred,
            )

            row = {
                "model": "Capacity-matched single-task MLP",
                "dataset": dataset,
                "representation_key": rep_key,
                "representation": REPRESENTATIONS[rep_key]["name"],
                "fold_id": "global_oof",
                "n_samples": int(
                    len(y_true)
                ),
                **metrics,
            }

            rows.append(
                row
            )

            # OOF predictions.
            out_pred = (
                OUT_PREDICTIONS
                / f"capacity_matched_{dataset}_{rep_key}_OOF_predictions.csv"
            )

            pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred,
            }).to_csv(
                out_pred,
                index=False,
                encoding="utf-8-sig",
            )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUT_METRICS
        / "capacity_matched_metrics_global_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return df


# =============================================================================
# 13. RESUMEN MEAN ± STD ENTRE FOLDS
# =============================================================================

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
]


def summarize_folds(
    df_fold: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        dataset,
        rep_key
    ), group in df_fold.groupby(
        [
            "dataset",
            "representation_key",
        ]
    ):

        row = {
            "dataset": dataset,
            "representation_key": rep_key,
            "representation": REPRESENTATIONS[rep_key]["name"],
            "n_folds": int(
                len(group)
            ),
        }

        for metric in METRIC_COLUMNS:

            values = group[
                metric
            ].astype(float)

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 14. LEER RESULTADOS MULTITASK EXISTENTES
# =============================================================================

def load_existing_multitask_fold_metrics() -> pd.DataFrame:

    dfs = []

    for rep_key in REPRESENTATIONS:

        path = (
            MT_ROOT
            / rep_key
            / f"multitask_{rep_key}_metrics_per_fold.csv"
        )

        if not path.exists():

            candidates = sorted(
                (
                    MT_ROOT
                    / rep_key
                ).glob(
                    f"*{rep_key}*metrics_per_fold*.csv"
                )
            )

            if not candidates:
                raise FileNotFoundError(
                    "No se encontraron las métricas por fold "
                    f"del multitask para {rep_key}.\n"
                    f"Esperado:\n{path}"
                )

            path = candidates[0]

        df = pd.read_csv(
            path
        )

        if "representation_key" not in df.columns:
            df[
                "representation_key"
            ] = rep_key

        dfs.append(
            df
        )

    result = pd.concat(
        dfs,
        ignore_index=True,
    )

    result[
        "dataset"
    ] = result[
        "dataset"
    ].astype(str).str.lower()

    result[
        "fold_id"
    ] = pd.to_numeric(
        result[
            "fold_id"
        ],
        errors="coerce",
    ).astype(int)

    return result


# =============================================================================
# 15. COMPARACIÓN FOLD A FOLD: MTL - CONTROL
# =============================================================================

def compare_with_multitask(
    df_control_fold: pd.DataFrame,
    df_mtl_fold: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    control = df_control_fold.copy()
    mtl = df_mtl_fold.copy()

    keys = [
        "dataset",
        "representation_key",
        "fold_id",
    ]

    control_keep = (
        keys
        +
        METRIC_COLUMNS
    )

    mtl_keep = (
        keys
        +
        METRIC_COLUMNS
    )

    control = control[
        control_keep
    ].copy()

    mtl = mtl[
        mtl_keep
    ].copy()

    control = control.rename(
        columns={
            m: f"{m}_single_task_matched"
            for m in METRIC_COLUMNS
        }
    )

    mtl = mtl.rename(
        columns={
            m: f"{m}_multitask"
            for m in METRIC_COLUMNS
        }
    )

    paired = pd.merge(
        control,
        mtl,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    for metric in METRIC_COLUMNS:

        paired[
            f"delta_{metric}_MTL_minus_ST"
        ] = (
            paired[
                f"{metric}_multitask"
            ]
            -
            paired[
                f"{metric}_single_task_matched"
            ]
        )

    paired.to_csv(
        OUT_COMPARISON
        / "paired_fold_capacity_matched_vs_multitask.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Resumen descriptivo
    # --------------------------------------------------------

    summary_rows = []

    for (
        dataset,
        rep_key
    ), group in paired.groupby(
        [
            "dataset",
            "representation_key",
        ]
    ):

        row = {
            "dataset": dataset,
            "representation_key": rep_key,
            "representation": REPRESENTATIONS[rep_key]["name"],
            "n_folds": int(
                len(group)
            ),
        }

        for metric in METRIC_COLUMNS:

            st = group[
                f"{metric}_single_task_matched"
            ].astype(float)

            mt = group[
                f"{metric}_multitask"
            ].astype(float)

            delta = (
                mt - st
            )

            row[
                f"{metric}_ST_mean"
            ] = float(
                st.mean()
            )

            row[
                f"{metric}_ST_std"
            ] = float(
                st.std(
                    ddof=1
                )
            )

            row[
                f"{metric}_MTL_mean"
            ] = float(
                mt.mean()
            )

            row[
                f"{metric}_MTL_std"
            ] = float(
                mt.std(
                    ddof=1
                )
            )

            row[
                f"{metric}_delta_mean_MTL_minus_ST"
            ] = float(
                delta.mean()
            )

            row[
                f"{metric}_delta_std"
            ] = float(
                delta.std(
                    ddof=1
                )
            )

            row[
                f"{metric}_MTL_better_folds"
            ] = int(
                np.sum(
                    delta > 0
                )
            )

            row[
                f"{metric}_equal_folds"
            ] = int(
                np.sum(
                    np.isclose(
                        delta,
                        0.0,
                        atol=1e-12,
                    )
                )
            )

            row[
                f"{metric}_ST_better_folds"
            ] = int(
                np.sum(
                    delta < 0
                )
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_COMPARISON
        / "summary_capacity_matched_vs_multitask.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return paired, summary


# =============================================================================
# 16. FIGURAS DESCRIPTIVAS POR FOLD
# =============================================================================

def make_paired_figures(
    paired: pd.DataFrame,
) -> None:
    """
    Figuras descriptivas únicamente.
    NO realizan inferencia estadística.
    """

    for dataset in DATASETS:

        df_d = paired[
            paired["dataset"] == dataset
        ].copy()

        if df_d.empty:
            continue

        # -----------------------------------------------
        # Macro-F1
        # -----------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(14, 4.5),
            sharey=True,
        )

        for ax, rep_key in zip(
            axes,
            REPRESENTATIONS.keys(),
        ):

            g = df_d[
                df_d["representation_key"] == rep_key
            ].sort_values(
                "fold_id"
            )

            x = g[
                "fold_id"
            ].to_numpy()

            y_st = g[
                "f1_macro_single_task_matched"
            ].to_numpy()

            y_mt = g[
                "f1_macro_multitask"
            ].to_numpy()

            for i in range(
                len(x)
            ):
                ax.plot(
                    [0, 1],
                    [
                        y_st[i],
                        y_mt[i],
                    ],
                    marker="o",
                    alpha=0.65,
                )

            ax.set_xticks(
                [0, 1]
            )

            ax.set_xticklabels(
                [
                    "Matched\nsingle-task",
                    "Multitask",
                ]
            )

            ax.set_title(
                REPRESENTATIONS[rep_key]["name"]
            )

            ax.grid(
                True,
                axis="y",
                alpha=0.25,
            )

        axes[0].set_ylabel(
            "Macro-F1"
        )

        fig.suptitle(
            f"{dataset.capitalize()}: fold-wise capacity-matched control vs multitask"
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.93]
        )

        fig.savefig(
            OUT_FIGURES
            / f"{dataset}_paired_macro_f1_capacity_matched_vs_multitask.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            OUT_FIGURES
            / f"{dataset}_paired_macro_f1_capacity_matched_vs_multitask.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        # -----------------------------------------------
        # Balanced accuracy
        # -----------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(14, 4.5),
            sharey=True,
        )

        for ax, rep_key in zip(
            axes,
            REPRESENTATIONS.keys(),
        ):

            g = df_d[
                df_d["representation_key"] == rep_key
            ].sort_values(
                "fold_id"
            )

            x = g[
                "fold_id"
            ].to_numpy()

            y_st = g[
                "balanced_accuracy_single_task_matched"
            ].to_numpy()

            y_mt = g[
                "balanced_accuracy_multitask"
            ].to_numpy()

            for i in range(
                len(x)
            ):
                ax.plot(
                    [0, 1],
                    [
                        y_st[i],
                        y_mt[i],
                    ],
                    marker="o",
                    alpha=0.65,
                )

            ax.set_xticks(
                [0, 1]
            )

            ax.set_xticklabels(
                [
                    "Matched\nsingle-task",
                    "Multitask",
                ]
            )

            ax.set_title(
                REPRESENTATIONS[rep_key]["name"]
            )

            ax.grid(
                True,
                axis="y",
                alpha=0.25,
            )

        axes[0].set_ylabel(
            "Balanced accuracy"
        )

        fig.suptitle(
            f"{dataset.capitalize()}: fold-wise capacity-matched control vs multitask"
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.93]
        )

        fig.savefig(
            OUT_FIGURES
            / f"{dataset}_paired_balanced_accuracy_capacity_matched_vs_multitask.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            OUT_FIGURES
            / f"{dataset}_paired_balanced_accuracy_capacity_matched_vs_multitask.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )


# =============================================================================
# 17. AUDITORÍA DE FOLDS
# =============================================================================

def audit_identical_folds() -> pd.DataFrame:

    rows = []

    for rep_key in REPRESENTATIONS:
        for fold_id in range(
            1,
            N_SPLITS + 1
        ):
            for dataset in DATASETS:

                fold = load_fold(
                    dataset,
                    rep_key,
                    fold_id,
                )

                rows.append({
                    "dataset": dataset,
                    "representation_key": rep_key,
                    "fold_id": fold_id,
                    "fold_file": str(
                        fold["path"]
                    ),
                    "n_train": int(
                        len(
                            fold["y_train"]
                        )
                    ),
                    "n_test": int(
                        len(
                            fold["y_test"]
                        )
                    ),
                    "input_dim": int(
                        fold["X_train"].shape[1]
                    ),
                    "train_class_counts": json.dumps(
                        {
                            int(k): int(v)
                            for k, v in zip(
                                *np.unique(
                                    fold["y_train"],
                                    return_counts=True,
                                )
                            )
                        }
                    ),
                    "test_class_counts": json.dumps(
                        {
                            int(k): int(v)
                            for k, v in zip(
                                *np.unique(
                                    fold["y_test"],
                                    return_counts=True,
                                )
                            )
                        }
                    ),
                })

                del fold

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUT_LOGS
        / "identical_fold_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return df


# =============================================================================
# 18. CONFIGURACIÓN DE REPRODUCIBILIDAD
# =============================================================================

def write_configuration(
    parameter_info: dict,
) -> None:

    config = {
        "purpose": (
            "Reviewer Comment 7: capacity-matched single-task control."
        ),
        "created_at": datetime.now().isoformat(),
        "environment": environment_info(),
        "input": {
            "cv_root": str(
                CV_ROOT
            ),
            "existing_multitask_results": str(
                MT_ROOT
            ),
            "input_dim": INPUT_DIM,
            "representations": REPRESENTATIONS,
            "datasets": DATASETS,
            "same_precomputed_folds_as_multitask": True,
            "top_k_and_scaler_recomputed": False,
        },
        "architecture": {
            "encoder": [
                4096,
                1024,
                256,
                64,
            ],
            "batch_norm": True,
            "activation": "ReLU",
            "dropout": DROPOUT,
            "dropout_layers": [
                "after 1024",
                "after 256",
            ],
            "classification_head": [
                64,
                32,
                5,
            ],
        },
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "early_stopping": False,
            "optimizer": "Adam",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "StepLR",
            "scheduler_step_size": SCHEDULER_STEP_SIZE,
            "scheduler_gamma": SCHEDULER_GAMMA,
            "loss": "Weighted cross-entropy",
            "class_weight_formula": "w_c = n / (K * n_c)",
            "batch_size_jacket": BATCH_SIZE["jacket"],
            "batch_size_rotor": BATCH_SIZE["rotor"],
            "match_multitask_optimizer_steps_per_epoch": MATCH_MULTITASK_UPDATE_BUDGET,
        },
        "parameter_audit": parameter_info,
        "output_root": str(
            OUT_ROOT
        ),
    }

    save_json(
        OUT_LOGS
        / "capacity_matched_run_configuration.json",
        config,
    )


# =============================================================================
# 19. MAIN
# =============================================================================

def main() -> None:

    setup_dirs()

    print_header(
        "CAPACITY-MATCHED SINGLE-TASK CONTROL - REVIEWER COMMENT 7"
    )

    print(
        "DEVICE:",
        DEVICE
    )

    print(
        "CV_ROOT:",
        CV_ROOT
    )

    print(
        "MT_ROOT:",
        MT_ROOT
    )

    print(
        "OUT_ROOT:",
        OUT_ROOT
    )

    if not CV_ROOT.exists():
        raise FileNotFoundError(
            f"No existe:\n{CV_ROOT}"
        )

    if not MT_ROOT.exists():
        raise FileNotFoundError(
            f"No existe la carpeta con resultados multitask:\n{MT_ROOT}"
        )

    # --------------------------------------------------------
    # Auditoría de capacidad
    # --------------------------------------------------------

    parameter_info = parameter_audit()

    print_header(
        "AUDITORÍA DE PARÁMETROS"
    )

    print(
        json.dumps(
            parameter_info,
            indent=4,
            ensure_ascii=False,
        )
    )

    write_configuration(
        parameter_info
    )

    # --------------------------------------------------------
    # Auditoría de folds
    # --------------------------------------------------------

    fold_audit = audit_identical_folds()

    print_header(
        "AUDITORÍA DE FOLDS"
    )

    print(
        fold_audit[
            [
                "dataset",
                "representation_key",
                "fold_id",
                "n_train",
                "n_test",
                "input_dim",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------

    all_results = []

    for rep_key in [
        "fft",
        "stft",
        "welch",
    ]:

        print_header(
            f"REPRESENTACIÓN: {REPRESENTATIONS[rep_key]['name']}"
        )

        for dataset in [
            "jacket",
            "rotor",
        ]:

            for fold_id in range(
                1,
                N_SPLITS + 1
            ):

                result = train_one_case(
                    dataset=dataset,
                    rep_key=rep_key,
                    fold_id=fold_id,
                )

                all_results.append(
                    result
                )

    # --------------------------------------------------------
    # Métricas por fold
    # --------------------------------------------------------

    df_control_fold = pd.DataFrame(
        [
            r["row"]
            for r in all_results
        ]
    )

    df_control_fold.to_csv(
        OUT_METRICS
        / "capacity_matched_metrics_per_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Resumen folds
    # --------------------------------------------------------

    df_control_fold_summary = summarize_folds(
        df_control_fold
    )

    df_control_fold_summary.to_csv(
        OUT_METRICS
        / "capacity_matched_metrics_mean_std_across_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Global OOF
    # --------------------------------------------------------

    df_global = summarize_global_oof(
        all_results
    )

    # --------------------------------------------------------
    # Comparación con MTL existente
    # --------------------------------------------------------

    df_mtl_fold = load_existing_multitask_fold_metrics()

    paired, comparison_summary = compare_with_multitask(
        df_control_fold,
        df_mtl_fold,
    )

    # --------------------------------------------------------
    # Figuras descriptivas
    # --------------------------------------------------------

    make_paired_figures(
        paired
    )

    # --------------------------------------------------------
    # Excel final
    # --------------------------------------------------------

    excel_path = (
        OUT_ROOT
        / "capacity_matched_control_complete_results.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        df_control_fold.to_excel(
            writer,
            sheet_name="Control_Per_Fold",
            index=False,
        )

        df_control_fold_summary.to_excel(
            writer,
            sheet_name="Control_Mean_Std",
            index=False,
        )

        df_global.to_excel(
            writer,
            sheet_name="Control_Global_OOF",
            index=False,
        )

        paired.to_excel(
            writer,
            sheet_name="Paired_MTL_vs_Control",
            index=False,
        )

        comparison_summary.to_excel(
            writer,
            sheet_name="Comparison_Summary",
            index=False,
        )

        fold_audit.to_excel(
            writer,
            sheet_name="Fold_Audit",
            index=False,
        )

        pd.DataFrame(
            [
                parameter_info
            ]
        ).to_excel(
            writer,
            sheet_name="Parameter_Audit",
            index=False,
        )

    # --------------------------------------------------------
    # TXT final
    # --------------------------------------------------------

    summary_txt = []

    summary_txt.append(
        "CAPACITY-MATCHED SINGLE-TASK CONTROL"
    )

    summary_txt.append(
        "=" * 110
    )

    summary_txt.append(
        ""
    )

    summary_txt.append(
        "PURPOSE"
    )

    summary_txt.append(
        "Direct response to Reviewer Comment 7."
    )

    summary_txt.append(
        ""
    )

    summary_txt.append(
        "PARAMETER AUDIT"
    )

    summary_txt.append(
        json.dumps(
            parameter_info,
            indent=2,
            ensure_ascii=False,
        )
    )

    summary_txt.append(
        ""
    )

    summary_txt.append(
        "CONTROL: MEAN ± STD ACROSS FOLDS"
    )

    summary_txt.append(
        df_control_fold_summary.to_string(
            index=False
        )
    )

    summary_txt.append(
        ""
    )

    summary_txt.append(
        "CONTROL: GLOBAL OOF"
    )

    summary_txt.append(
        df_global.to_string(
            index=False
        )
    )

    summary_txt.append(
        ""
    )

    summary_txt.append(
        "PAIRED DESCRIPTIVE COMPARISON: MTL - MATCHED SINGLE-TASK"
    )

    summary_txt.append(
        comparison_summary.to_string(
            index=False
        )
    )

    summary_txt.append(
        ""
    )

    summary_txt.append(
        "IMPORTANT STATISTICAL NOTE"
    )

    summary_txt.append(
        "No inferential p-values are computed in this script. "
        "The reviewer separately questioned significance testing "
        "based on only five correlated CV folds. Formal inference "
        "will be handled in the dedicated response to Reviewer Comment 4."
    )

    (
        OUT_ROOT
        / "README_capacity_matched_control_results.txt"
    ).write_text(
        "\n".join(
            summary_txt
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Pantalla
    # --------------------------------------------------------

    print_header(
        "RESULTADOS CONTROL - GLOBAL OOF"
    )

    print(
        df_global[
            [
                "dataset",
                "representation",
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
            ]
        ].to_string(
            index=False
        )
    )

    print_header(
        "COMPARACIÓN DESCRIPTIVA MTL - CONTROL MATCHED"
    )

    cols_show = [
        "dataset",
        "representation",
        "f1_macro_ST_mean",
        "f1_macro_MTL_mean",
        "f1_macro_delta_mean_MTL_minus_ST",
        "balanced_accuracy_ST_mean",
        "balanced_accuracy_MTL_mean",
        "balanced_accuracy_delta_mean_MTL_minus_ST",
    ]

    available = [
        c
        for c in cols_show
        if c in comparison_summary.columns
    ]

    print(
        comparison_summary[
            available
        ].to_string(
            index=False
        )
    )

    print_header(
        "SCRIPT TERMINADO"
    )

    print(
        "Resultados guardados en:"
    )

    print(
        OUT_ROOT
    )

    print(
        "\nExcel principal:"
    )

    print(
        excel_path
    )


if __name__ == "__main__":
    main()
