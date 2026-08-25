# -*- coding: utf-8 -*-
"""
Baselines adicionales para la revisión (Major Revision) - script maestro
==================================================================================
Script AUTOCONTENIDO para el repositorio público (reproducibilidad, ver
respuesta a Comentario 24 / Data Availability Statement, y Comentario 13b).

Consolida, en un solo archivo con banderas ON/OFF, TODO el código usado para
generar los resultados de baselines adicionales reportados en la revisión:

  [1] BASELINES CLÁSICOS: Random Forest y Boosting (XGBoost, con fallback a
      GradientBoosting de sklearn si XGBoost no está instalado)  -> R1#13a
  [2] CNN 1D EXPLORATORIO (Top-K reordenado por frecuencia)
  [3] RNN (LSTM bidireccional) EXPLORATORIO (Top-K reordenado por frecuencia)

Nota importante: el baseline SVM-RBF que también se corrió durante la
exploración de [1] se dejó deliberadamente fuera de este script y de los
resultados reportados en el manuscrito; no se incluye aquí por consistencia
con esa decisión.

Todos los experimentos reutilizan EXACTAMENTE los mismos folds Top-K=4096 ya
cacheados por 01_crear_datasets_representaciones_cv5.py (no regeneran nada
del pipeline de datos) y el mismo protocolo general (seed=42, mismos batch
sizes por dataset) que el resto de los modelos single-task del repositorio.

Salida (subcarpetas de 08_experimentos_revision, idénticas a las ya usadas
para generar los resultados reportados):
  .../08_experimentos_revision/04_sklearn_baselines   (Random Forest, Boosting)
  .../08_experimentos_revision/05_cnn1d_exploratorio  (CNN 1D)
  .../08_experimentos_revision/06_rnn_exploratorio    (RNN)
  .../08_experimentos_revision/resumen_baselines_adicionales.csv  (consolidado)
"""

import os
import gc
import time
import random
from pathlib import Path

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Boosting: XGBoost si está disponible, si no GradientBoosting de sklearn.
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
    print("XGBoost detectado: se usará XGBClassifier para el baseline de boosting.")
except ImportError:
    USE_XGBOOST = False
    print("XGBoost no está instalado: se usará GradientBoostingClassifier de sklearn.")


# =============================================================================
# 1. CONFIGURACIÓN GENERAL (idéntica a la de los scripts que generaron los
#    resultados reportados)
# =============================================================================

SEED = 42

PROJECT_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold")
REPRO_ROOT = PROJECT_ROOT / "reproducible"
CV_ROOT = REPRO_ROOT / "04_cv5_topk4096"

OUT_REVISION = REPRO_ROOT / "08_experimentos_revision"
OUT_SKLEARN = OUT_REVISION / "04_sklearn_baselines"
OUT_CNN1D = OUT_REVISION / "05_cnn1d_exploratorio"
OUT_RNN = OUT_REVISION / "06_rnn_exploratorio"

N_SPLITS = 5
INPUT_DIM = 4096
N_CLASSES_JACKET = 5
N_CLASSES_ROTOR = 5
N_CLASSES_BY_DATASET = {"jacket": N_CLASSES_JACKET, "rotor": N_CLASSES_ROTOR}

REPRESENTATIONS = {
    "fft": {"name": "FFT log-magnitude", "folder": "01_FFT_log_magnitude_cv5_topk4096"},
    "stft": {"name": "STFT log-power", "folder": "02_STFT_log_power_cv5_topk4096"},
    "welch": {"name": "Welch PSD log-power", "folder": "03_Welch_PSD_log_power_cv5_topk4096"},
}

BATCH_SIZE_JACKET = 256
BATCH_SIZE_ROTOR = 128
BATCH_SIZE_BY_DATASET = {"jacket": BATCH_SIZE_JACKET, "rotor": BATCH_SIZE_ROTOR}

MT_EPOCHS = 120
MT_LR = 1e-3
MT_WEIGHT_DECAY = 1e-5
MT_DROPOUT = 0.25
MT_SCHEDULER_STEP_SIZE = 40
MT_SCHEDULER_GAMMA = 0.7

# Reagrupamiento del vector 4096 en (n_steps, step_features) para el RNN.
RNN_N_STEPS = 64
RNN_STEP_FEATURES = INPUT_DIM // RNN_N_STEPS  # 64
RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 1

# ---- Banderas ON/OFF ----
RUN_SKLEARN_BASELINES = True   # Random Forest + Boosting (XGBoost/GradientBoosting) -> R1#13a
RUN_CNN1D = True
RUN_RNN = True


def set_global_seed(seed: int = 42) -> None:
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


set_global_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 2. UTILIDADES COMPARTIDAS (idénticas a las usadas en 02_.../los scripts de
#    experimentos de la revisión; se consolidan aquí una sola vez)
# =============================================================================

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
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "class_names": class_names,
    }


def load_fold_freq_ordered(dataset: str, rep_key: str, fold_id: int) -> dict:
    """Igual que load_fold(), pero reordena las 4096 columnas Top-K de vuelta
    a orden de frecuencia ascendente usando 'selected_idx' guardado en el
    .npz, en vez del orden por varianza descendente con el que se guardaron.
    La usan tanto el CNN 1D como el RNN (no el baseline sklearn, que usa el
    orden Top-K original)."""
    path = fold_file(dataset, rep_key, fold_id)
    data = np.load(path, allow_pickle=True)
    fold = load_fold(dataset, rep_key, fold_id)
    selected_idx = data["selected_idx"].astype(np.int64)
    freq_order = np.argsort(selected_idx)  # permutación: varianza -> frecuencia
    fold["X_train"] = fold["X_train"][:, freq_order]
    fold["X_test"] = fold["X_test"][:, freq_order]
    return fold


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (n_classes * counts)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def get_class_weights_dict(y):
    """La fórmula total/(n_classes*count) es matemáticamente la misma que usa
    sklearn con class_weight='balanced', así que Random Forest abajo usa
    'balanced' directamente y Boosting usa este dict como sample_weight."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    weights = {}
    for c, n in zip(classes, counts):
        weights[int(c)] = float(total / (n_classes * n))
    return weights


def sample_weights_from_dict(y, weights_dict):
    return np.array([weights_dict[int(yi)] for yi in y], dtype=np.float64)


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
# 3. BASELINES CLÁSICOS: RANDOM FOREST + BOOSTING  ->  R1#13a
#    (SVM-RBF, corrido durante la exploración inicial, se excluye
#    deliberadamente: no se reporta en el manuscrito.)
# =============================================================================

def run_classical_baselines_case(dataset: str, rep_key: str) -> pd.DataFrame:
    print_header(f"BASELINES CLÁSICOS (RF + Boosting) - {dataset.upper()} - {REPRESENTATIONS[rep_key]['name']}")
    case_out = OUT_SKLEARN / f"{dataset}_{rep_key}"
    case_out.mkdir(parents=True, exist_ok=True)

    fold_rows = {"RandomForest": [], "Boosting": []}
    y_true_all = {"RandomForest": [], "Boosting": []}
    y_pred_all = {"RandomForest": [], "Boosting": []}
    class_names_ref = None
    boosting_name = "XGBoost" if USE_XGBOOST else "GradientBoosting"

    for fold_id in range(1, N_SPLITS + 1):
        print_subheader(f"Baselines clásicos {dataset}_{rep_key} fold {fold_id}")
        fold = load_fold(dataset, rep_key, fold_id)
        X_train, X_test = fold["X_train"], fold["X_test"]
        y_train, y_test = fold["y_train"], fold["y_test"]
        class_names_ref = fold["class_names"]

        sw_dict = get_class_weights_dict(y_train)
        sample_weight = sample_weights_from_dict(y_train, sw_dict)

        # Random Forest.
        rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced")
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        # Boosting (XGBoost si está, si no GradientBoosting).
        if USE_XGBOOST:
            bst = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                                 random_state=SEED, n_jobs=-1, eval_metric="mlogloss")
            bst.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            bst = GradientBoostingClassifier(n_estimators=300, random_state=SEED)
            bst.fit(X_train, y_train, sample_weight=sample_weight)
        pred_bst = bst.predict(X_test)

        for model_key, y_pred in [("RandomForest", pred_rf), ("Boosting", pred_bst)]:
            fold_row = compute_metrics_row(y_test, y_pred, model=model_key if model_key != "Boosting" else boosting_name,
                                            dataset=dataset, representation=REPRESENTATIONS[rep_key]["name"], fold_id=fold_id)
            fold_rows[model_key].append(fold_row)
            y_true_all[model_key].append(y_test)
            y_pred_all[model_key].append(y_pred)

        del rf, bst, X_train, X_test, y_train, y_test
        gc.collect()

    global_rows = []
    for model_key in ["RandomForest", "Boosting"]:
        model_name = boosting_name if model_key == "Boosting" else model_key
        global_rows.append(save_classification_outputs(case_out, model_name, dataset, rep_key,
                                                         y_true_all[model_key], y_pred_all[model_key],
                                                         fold_rows[model_key], class_names_ref))
    return pd.DataFrame(global_rows)


def run_all_classical_baselines() -> pd.DataFrame:
    print_header("EXPERIMENTO [1] - BASELINES CLÁSICOS: RANDOM FOREST + BOOSTING (R1#13a)")
    dfs = []
    for rep_key in ["fft", "stft", "welch"]:
        for dataset in ["jacket", "rotor"]:
            dfs.append(run_classical_baselines_case(dataset, rep_key))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT_SKLEARN / "metrics_global_classical_baselines_all.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# 4A. CNN 1D EXPLORATORIO
# =============================================================================

class CNN1DClassifier(nn.Module):
    """CNN 1D simple sobre el vector Top-K reordenado por frecuencia.
    Arquitectura exploratoria (no es 'capacidad igualada' en sentido estricto,
    es otra familia de red neuronal distinta al MLP del multitask)."""

    def __init__(self, input_len=4096, n_classes=5, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),  # 4096 -> 1024

            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),  # 1024 -> 256

            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),  # 256 -> 64

            nn.AdaptiveAvgPool1d(1),  # -> (batch, 128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, input_len)
        h = self.features(x)
        return self.classifier(h)


def train_cnn1d_case(dataset: str, rep_key: str) -> dict:
    print_header(f"CNN 1D EXPLORATORIO - {dataset.upper()} - {REPRESENTATIONS[rep_key]['name']}")
    tag = "CNN1D_freqordered"
    case_out = OUT_CNN1D / f"{dataset}_{rep_key}"
    models_dir = case_out / "modelos"
    models_dir.mkdir(parents=True, exist_ok=True)

    n_classes = N_CLASSES_BY_DATASET[dataset]
    batch_size = BATCH_SIZE_BY_DATASET[dataset]

    y_true_all, y_pred_all, fold_rows = [], [], []
    class_names_ref = None

    for fold_id in range(1, N_SPLITS + 1):
        print_subheader(f"CNN1D {dataset}_{rep_key} fold {fold_id}")
        fold = load_fold_freq_ordered(dataset, rep_key, fold_id)
        X_train, X_test = fold["X_train"], fold["X_test"]
        y_train, y_test = fold["y_train"], fold["y_test"]
        class_names_ref = fold["class_names"]

        train_loader = create_loader(X_train, y_train, batch_size, shuffle=True, seed=SEED + fold_id)

        model = CNN1DClassifier(input_len=INPUT_DIM, n_classes=n_classes, dropout=MT_DROPOUT).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes))
        optimizer = torch.optim.Adam(model.parameters(), lr=MT_LR, weight_decay=MT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=MT_SCHEDULER_STEP_SIZE, gamma=MT_SCHEDULER_GAMMA)

        for epoch in range(1, MT_EPOCHS + 1):
            model.train()
            epoch_loss, n_steps = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()); n_steps += 1
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == MT_EPOCHS:
                print(f"Fold {fold_id} | Epoch {epoch:03d}/{MT_EPOCHS} | loss={epoch_loss / max(n_steps,1):.6f}")

        y_true, y_pred = evaluate_model_single_head(model, X_test, y_test, task=None, batch_size=batch_size)
        fold_row = compute_metrics_row(y_true, y_pred, model=tag, dataset=dataset,
                                        representation=REPRESENTATIONS[rep_key]["name"], fold_id=fold_id)
        fold_rows.append(fold_row)
        y_true_all.append(y_true); y_pred_all.append(y_pred)

        torch.save(model.state_dict(), models_dir / f"{tag}_{dataset}_{rep_key}_fold{fold_id}.pt")
        del model, optimizer, scheduler, X_train, X_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return save_classification_outputs(case_out, tag, dataset, rep_key, y_true_all, y_pred_all, fold_rows, class_names_ref)


def run_all_cnn1d() -> pd.DataFrame:
    print_header("EXPERIMENTO [2] - CNN 1D EXPLORATORIO (reordenado por frecuencia)")
    rows = []
    for rep_key in ["fft", "stft", "welch"]:
        for dataset in ["jacket", "rotor"]:
            rows.append(train_cnn1d_case(dataset, rep_key))
    df = pd.DataFrame(rows)
    OUT_CNN1D.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CNN1D / "metrics_global_cnn1d_all_cases.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# 4B. RNN (LSTM BIDIRECCIONAL) EXPLORATORIO
# =============================================================================

class RNNClassifier(nn.Module):
    """LSTM bidireccional sobre el vector Top-K reordenado por frecuencia,
    reagrupado en (RNN_N_STEPS, RNN_STEP_FEATURES) pasos de tiempo.
    Arquitectura exploratoria (no es 'capacidad igualada' en sentido estricto,
    es otra familia de red neuronal distinta al MLP del multitask)."""

    def __init__(self, n_steps=64, step_features=64, hidden_size=64, num_layers=1,
                 n_classes=5, dropout=0.25):
        super().__init__()
        self.n_steps = n_steps
        self.step_features = step_features
        self.lstm = nn.LSTM(
            input_size=step_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_seq = x.view(batch_size, self.n_steps, self.step_features)
        out, (h_n, c_n) = self.lstm(x_seq)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_final = torch.cat([h_forward, h_backward], dim=1)
        return self.classifier(h_final)


def train_rnn_case(dataset: str, rep_key: str) -> dict:
    print_header(f"RNN EXPLORATORIO - {dataset.upper()} - {REPRESENTATIONS[rep_key]['name']}")
    tag = "RNN_freqordered"
    case_out = OUT_RNN / f"{dataset}_{rep_key}"
    models_dir = case_out / "modelos"
    models_dir.mkdir(parents=True, exist_ok=True)

    n_classes = N_CLASSES_BY_DATASET[dataset]
    batch_size = BATCH_SIZE_BY_DATASET[dataset]

    y_true_all, y_pred_all, fold_rows = [], [], []
    class_names_ref = None

    for fold_id in range(1, N_SPLITS + 1):
        print_subheader(f"RNN {dataset}_{rep_key} fold {fold_id}")
        fold = load_fold_freq_ordered(dataset, rep_key, fold_id)
        X_train, X_test = fold["X_train"], fold["X_test"]
        y_train, y_test = fold["y_train"], fold["y_test"]
        class_names_ref = fold["class_names"]

        train_loader = create_loader(X_train, y_train, batch_size, shuffle=True, seed=SEED + fold_id)

        model = RNNClassifier(n_steps=RNN_N_STEPS, step_features=RNN_STEP_FEATURES,
                               hidden_size=RNN_HIDDEN_SIZE, num_layers=RNN_NUM_LAYERS,
                               n_classes=n_classes, dropout=MT_DROPOUT).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes))
        optimizer = torch.optim.Adam(model.parameters(), lr=MT_LR, weight_decay=MT_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=MT_SCHEDULER_STEP_SIZE, gamma=MT_SCHEDULER_GAMMA)

        for epoch in range(1, MT_EPOCHS + 1):
            model.train()
            epoch_loss, n_steps_done = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()); n_steps_done += 1
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == MT_EPOCHS:
                print(f"Fold {fold_id} | Epoch {epoch:03d}/{MT_EPOCHS} | loss={epoch_loss / max(n_steps_done,1):.6f}")

        y_true, y_pred = evaluate_model_single_head(model, X_test, y_test, task=None, batch_size=batch_size)
        fold_row = compute_metrics_row(y_true, y_pred, model=tag, dataset=dataset,
                                        representation=REPRESENTATIONS[rep_key]["name"], fold_id=fold_id)
        fold_rows.append(fold_row)
        y_true_all.append(y_true); y_pred_all.append(y_pred)

        torch.save(model.state_dict(), models_dir / f"{tag}_{dataset}_{rep_key}_fold{fold_id}.pt")
        del model, optimizer, scheduler, X_train, X_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return save_classification_outputs(case_out, tag, dataset, rep_key, y_true_all, y_pred_all, fold_rows, class_names_ref)


def run_all_rnn() -> pd.DataFrame:
    print_header("EXPERIMENTO [3] - RNN (LSTM BIDIRECCIONAL) EXPLORATORIO (reordenado por frecuencia)")
    rows = []
    for rep_key in ["fft", "stft", "welch"]:
        for dataset in ["jacket", "rotor"]:
            rows.append(train_rnn_case(dataset, rep_key))
    df = pd.DataFrame(rows)
    OUT_RNN.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_RNN / "metrics_global_rnn_all_cases.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# 5. MAIN - correr esta celda completa
# =============================================================================

if __name__ == "__main__":
    t0 = time.time()

    if not CV_ROOT.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de folds:\n{CV_ROOT}\n"
            "Verifica que 01_crear_datasets_representaciones_cv5.py ya se corrió antes."
        )

    OUT_REVISION.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    if RUN_SKLEARN_BASELINES:
        df_classical = run_all_classical_baselines()
        df_classical["family"] = "classical_baselines"
        all_summaries.append(df_classical)

    if RUN_CNN1D:
        df_cnn1d = run_all_cnn1d()
        df_cnn1d["family"] = "CNN1D_freqordered"
        all_summaries.append(df_cnn1d)

    if RUN_RNN:
        df_rnn = run_all_rnn()
        df_rnn["family"] = "RNN_freqordered"
        all_summaries.append(df_rnn)

    if all_summaries:
        df_master = pd.concat(all_summaries, ignore_index=True)
        df_master.to_csv(OUT_REVISION / "resumen_baselines_adicionales.csv", index=False, encoding="utf-8-sig")

        elapsed = time.time() - t0
        print_header(f"BASELINES ADICIONALES TERMINADOS EN {elapsed/60:.1f} MINUTOS")
        print("\nResumen global (accuracy, balanced_accuracy, f1_macro, f1_weighted):")
        print(df_master[["family", "model", "dataset", "representation", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]].to_string(index=False))
        print("\nArchivo consolidado guardado en:")
        print(OUT_REVISION / "resumen_baselines_adicionales.csv")