01_create_datasets_representations_cv5.py
Reproducible pipeline for the article:
"Improving Vibration-Based Classification in a Wind Turbine Jacket Structure and Rotor
through Multitask Learning and Spectral Representations"
This script creates:
1. Multiclass base dataset for the jacket structure.
2. Multiclass base dataset for the rotor.
3. FFT log-magnitude, STFT log-power, and Welch PSD log-power representations.
4. CV5 folds with Top-K=4096 and StandardScaler fitted ONLY on the training set of each fold.
5. Configuration and summary files for traceability.

Requirements:
pip install numpy pandas scipy h5py scikit-learn joblib tqdm

"""
import os
import gc
import json
import shutil
import random
import platform
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import stft, welch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


# =============================================================================
# 0. CONFIGURACIÓN GENERAL
# =============================================================================

SEED = 42

PROJECT_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold")
REPRO_ROOT = PROJECT_ROOT / "reproducible"

# Rutas reales de los datos originales.
# Ajusta estas rutas únicamente si mueves los archivos originales.
JACKET_SOURCE_DIR = Path(r"C:\Users\Usuario\Documents\5. Multi Task\NPZ_dataset_base estructura")
JACKET_MAT_PATH = JACKET_SOURCE_DIR / "datacompletos_sin_escalar.mat"
JACKET_LABELS_PATH = JACKET_SOURCE_DIR / "Ywind.csv"

ROTOR_SOURCE_DIR = Path(r"C:\Users\Usuario\Documents\5. Multi Task\NPZ_dataset_rotor")
ROTOR_HEALTHY_DIR = ROTOR_SOURCE_DIR / "DatosSanos_Entrenamiento"
ROTOR_FAULTS_DIR = ROTOR_SOURCE_DIR / "DatosFallos"

# Si True, borra la carpeta reproducible antes de regenerar todo.
CLEAN_OUTPUT = True

# Parámetros globales.
N_SPLITS = 5
TOP_K = 4096

# Jacket.
JACKET_N_SAMPLES = 5740
JACKET_L = 2417
JACKET_CHANNELS = 24
JACKET_FEATURES = JACKET_L * JACKET_CHANNELS
JACKET_FS = 275.0

# Rotor.
ROTOR_FS = 1706.66
ROTOR_L = 3200
ROTOR_CHANNELS = 24
ROTOR_TOTAL_SAMPLES_PER_EXPERIMENT = 102400
ROTOR_WINDOWS_PER_EXPERIMENT = 32

# STFT.
STFT_NPERSEG = 256
STFT_NOVERLAP = 128
STFT_NFFT = 256
STFT_WINDOW = "hann"

# Welch.
WELCH_NPERSEG = 512
WELCH_NOVERLAP = 256
WELCH_NFFT = 512
WELCH_WINDOW = "hann"

EPS = 1e-12

# Salidas.
DIR_RAW_INFO = REPRO_ROOT / "00_datos_originales_info"
DIR_SCRIPTS = REPRO_ROOT / "01_scripts"
DIR_BASE = REPRO_ROOT / "02_datasets_base"
DIR_REP = REPRO_ROOT / "03_representaciones"
DIR_CV5 = REPRO_ROOT / "04_cv5_topk4096"
DIR_LOGS = REPRO_ROOT / "99_logs"

REPRESENTATIONS = {
    "fft": {
        "name": "FFT log-magnitude",
        "folder": "01_FFT_log_magnitude",
    },
    "stft": {
        "name": "STFT log-power",
        "folder": "02_STFT_log_power",
    },
    "welch": {
        "name": "Welch PSD log-power",
        "folder": "03_Welch_PSD_log_power",
    },
}


# =============================================================================
# 1. UTILIDADES
# =============================================================================

def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


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


def ensure_dirs() -> None:
    """
    Crea la estructura reproducible.

    IMPORTANTE:
    No borra REPRO_ROOT completo porque el propio script está dentro de:
        reproducible/01_scripts

    Si CLEAN_OUTPUT=True, solo borra salidas regenerables:
        00_datos_originales_info
        02_datasets_base
        03_representaciones
        04_cv5_topk4096
        08_verificacion_reproducibilidad
        99_logs

    Nunca borra:
        01_scripts
    """
    REPRO_ROOT.mkdir(parents=True, exist_ok=True)
    DIR_SCRIPTS.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUTPUT:
        dirs_to_clean = [
            DIR_RAW_INFO,
            DIR_BASE,
            DIR_REP,
            DIR_CV5,
            REPRO_ROOT / "08_verificacion_reproducibilidad",
            DIR_LOGS,
        ]

        for d in dirs_to_clean:
            if d.exists():
                print(f"Borrando salida anterior: {d}")
                shutil.rmtree(d)

    for d in [
        REPRO_ROOT,
        DIR_RAW_INFO,
        DIR_SCRIPTS,
        DIR_BASE,
        DIR_REP,
        DIR_CV5,
        DIR_LOGS,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def verify_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {description}:\n{path}")


def verify_dir(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {description}:\n{path}")


def array_stats(X: np.ndarray) -> dict:
    return {
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "nan": int(np.isnan(X).sum()) if np.issubdtype(X.dtype, np.number) else None,
        "inf": int(np.isinf(X).sum()) if np.issubdtype(X.dtype, np.number) else None,
        "min": float(np.nanmin(X)) if np.issubdtype(X.dtype, np.number) else None,
        "max": float(np.nanmax(X)) if np.issubdtype(X.dtype, np.number) else None,
        "mean": float(np.nanmean(X)) if np.issubdtype(X.dtype, np.number) else None,
        "std": float(np.nanstd(X)) if np.issubdtype(X.dtype, np.number) else None,
    }


def validate_numeric_matrix(name: str, X: np.ndarray) -> None:
    print(f"{name}: shape={X.shape}, dtype={X.dtype}")
    if np.isnan(X).any():
        raise ValueError(f"{name} contiene NaN.")
    if np.isinf(X).any():
        raise ValueError(f"{name} contiene Inf.")


def read_mat_2d_any_format(path: Path) -> np.ndarray:
    """
    Carga un .mat con una matriz 2D. Funciona para MATLAB clásico y v7.3.
    Retorna la matriz numérica 2D más grande si no encuentra una forma exacta.
    """
    path = Path(path)

    try:
        data = sio.loadmat(path)
        candidates = []
        for key, value in data.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                candidates.append((key, arr))

        if candidates:
            key, arr = max(candidates, key=lambda kv: kv[1].size)
            return np.asarray(arr, dtype=np.float32)

    except Exception:
        pass

    with h5py.File(path, "r") as f:
        candidates = []
        for key in f.keys():
            obj = f[key]
            try:
                arr = np.array(obj)
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                    candidates.append((key, arr))
            except Exception:
                continue

        if not candidates:
            raise ValueError(f"No se encontró una matriz 2D numérica en {path}")

        key, arr = max(candidates, key=lambda kv: kv[1].size)
        return np.asarray(arr, dtype=np.float32)


# =============================================================================
# 2. DATASET BASE JACKET
# =============================================================================

def load_jacket_labels(path: Path, n_expected: int = 5740) -> np.ndarray:
    """
    Lee Ywind.csv y remapea etiquetas a 0,1,2,3,4.
    Soporta CSV con una o varias columnas.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)

    if df.shape[1] == 1:
        y_raw = df.iloc[:, 0].to_numpy()
    else:
        # Escoger una columna con longitud compatible. Si hay varias, se usa la última.
        candidate_cols = []
        for col in df.columns:
            values = df[col].dropna().to_numpy()
            if len(values) == n_expected:
                candidate_cols.append(col)

        col = candidate_cols[-1] if candidate_cols else df.columns[-1]
        y_raw = df[col].to_numpy()

    y_raw = np.asarray(y_raw).reshape(-1)
    y_raw = y_raw[pd.notna(y_raw)]

    if len(y_raw) != n_expected:
        raise ValueError(
            f"El número de etiquetas jacket ({len(y_raw)}) no coincide con {n_expected}."
        )

    # Convertir a numérico si es posible.
    try:
        y_raw_numeric = y_raw.astype(float)
        unique_vals = np.unique(y_raw_numeric)
        mapping = {old: new for new, old in enumerate(sorted(unique_vals))}
        y_multi = np.array([mapping[v] for v in y_raw_numeric], dtype=np.int64)
    except Exception:
        unique_vals = sorted(pd.unique(y_raw).tolist())
        mapping = {old: new for new, old in enumerate(unique_vals)}
        y_multi = np.array([mapping[v] for v in y_raw], dtype=np.int64)

    if len(np.unique(y_multi)) != 5:
        raise ValueError(
            f"Se esperaban 5 clases para jacket, pero se encontraron {len(np.unique(y_multi))}: "
            f"{np.unique(y_multi)}"
        )

    return y_multi.astype(np.int64)


def create_jacket_base_dataset() -> Path:
    print_header("PASO 1A - CREAR DATASET BASE JACKET MULTICLASE")

    verify_file(JACKET_MAT_PATH, "archivo MATLAB de jacket")
    verify_file(JACKET_LABELS_PATH, "archivo Ywind.csv de jacket")

    with h5py.File(JACKET_MAT_PATH, "r") as f:
        if "datacompleto" not in f.keys():
            raise KeyError("No se encontró la variable 'datacompleto' en el .mat de jacket.")
        X = np.array(f["datacompleto"])

    print("Tamaño jacket cargado inicialmente:", X.shape)

    if X.shape == (JACKET_FEATURES, JACKET_N_SAMPLES):
        print("La matriz jacket venía transpuesta. Se aplica X = X.T")
        X = X.T

    if X.shape != (JACKET_N_SAMPLES, JACKET_FEATURES):
        raise ValueError(
            f"Jacket debería tener shape {(JACKET_N_SAMPLES, JACKET_FEATURES)}, "
            f"pero tiene {X.shape}."
        )

    X = X.astype(np.float32)
    y_multi = load_jacket_labels(JACKET_LABELS_PATH, JACKET_N_SAMPLES)

    # Asegurar mapeo: 0 = Healthy, 1-4 = Crack levels.
    # Si el CSV ya está ordenado de forma diferente, este script conserva el remapeo ordinal.
    y_bin = (y_multi != 0).astype(np.uint8)

    class_names = np.array(
        ["Healthy", "Crack_Level_1", "Crack_Level_2", "Crack_Level_3", "Crack_Level_4"],
        dtype=object,
    )

    # Para jacket, el artículo trata cada muestra como grupo independiente.
    groups = np.arange(len(y_multi), dtype=np.int64)

    print("X jacket:", X.shape)
    print("Conteos jacket:", dict(zip(*np.unique(y_multi, return_counts=True))))

    out_path = DIR_BASE / "dataset_base_jacket_grietas_multiclase_sin_escalar.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y_multi=y_multi.astype(np.int64),
        y_bin=y_bin.astype(np.uint8),
        groups=groups.astype(np.int64),
        class_names=class_names,
        fs=np.array([JACKET_FS], dtype=np.float32),
        L=np.array([JACKET_L], dtype=np.int32),
        N=np.array([JACKET_CHANNELS], dtype=np.int32),
        description=np.array([
            "Dataset base jacket multiclase. X=(5740,58008). "
            "Cada muestra se reconstruye como (2417,24). "
            "groups = un grupo por muestra."
        ]),
    )

    save_json(
        DIR_LOGS / "jacket_base_summary.json",
        {
            "path": str(out_path),
            "X": array_stats(X),
            "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_multi, return_counts=True))},
            "class_names": class_names.tolist(),
            "groups_rule": "one sample per group",
        },
    )

    return out_path


# =============================================================================
# 3. DATASET BASE ROTOR
# =============================================================================

def load_rotor_experiment(path: Path) -> np.ndarray:
    X = read_mat_2d_any_format(path)

    if X.shape == (ROTOR_CHANNELS, ROTOR_TOTAL_SAMPLES_PER_EXPERIMENT):
        X = X.T

    if X.shape[1] != ROTOR_CHANNELS and X.shape[0] == ROTOR_CHANNELS:
        X = X.T

    if X.shape[1] != ROTOR_CHANNELS:
        raise ValueError(f"{path.name}: se esperaban {ROTOR_CHANNELS} canales, shape={X.shape}")

    if X.shape[0] < ROTOR_TOTAL_SAMPLES_PER_EXPERIMENT:
        raise ValueError(
            f"{path.name}: se esperaban al menos {ROTOR_TOTAL_SAMPLES_PER_EXPERIMENT} muestras, "
            f"shape={X.shape}"
        )

    X = X[:ROTOR_TOTAL_SAMPLES_PER_EXPERIMENT, :].astype(np.float32)
    return X


def segment_rotor_experiment(X_exp: np.ndarray) -> np.ndarray:
    """
    Convierte un experimento (102400,24) en 32 ventanas (3200,24).
    """
    expected = ROTOR_WINDOWS_PER_EXPERIMENT * ROTOR_L
    if X_exp.shape[0] != expected:
        raise ValueError(f"Experimento rotor tiene {X_exp.shape[0]} muestras; se esperaban {expected}.")
    return X_exp.reshape(ROTOR_WINDOWS_PER_EXPERIMENT, ROTOR_L, ROTOR_CHANNELS).astype(np.float32)


def create_rotor_base_dataset() -> Path:
    print_header("PASO 1B - CREAR DATASET BASE ROTOR MULTICLASE")

    verify_dir(ROTOR_HEALTHY_DIR, "carpeta de datos sanos rotor")
    verify_dir(ROTOR_FAULTS_DIR, "carpeta de fallos rotor")

    healthy_files = []
    for letter in ["A", "B", "C", "D", "E"]:
        for idx in range(3):
            healthy_files.append(ROTOR_HEALTHY_DIR / f"Data_{letter}{idx}.mat")

    fault_files = []
    for level in range(1, 5):
        for idx in range(1, 6):
            fault_files.append((ROTOR_FAULTS_DIR / f"Fallo{level}_{idx}.mat", level))

    for p in healthy_files:
        verify_file(p, f"archivo sano rotor {p.name}")
    for p, _ in fault_files:
        verify_file(p, f"archivo fallo rotor {p.name}")

    X_windows_list = []
    y_list = []
    y_bin_list = []
    groups_list = []
    paths_list = []
    exp_names_list = []

    group_id = 0

    print_subheader("Procesando experimentos sanos rotor")
    for p in tqdm(healthy_files, desc="Rotor healthy"):
        X_exp = load_rotor_experiment(p)
        X_w = segment_rotor_experiment(X_exp)

        X_windows_list.append(X_w)
        y_list.extend([0] * ROTOR_WINDOWS_PER_EXPERIMENT)
        y_bin_list.extend([0] * ROTOR_WINDOWS_PER_EXPERIMENT)
        groups_list.extend([group_id] * ROTOR_WINDOWS_PER_EXPERIMENT)
        paths_list.extend([str(p)] * ROTOR_WINDOWS_PER_EXPERIMENT)
        exp_names_list.extend([p.stem] * ROTOR_WINDOWS_PER_EXPERIMENT)
        group_id += 1

    print_subheader("Procesando experimentos con desbalance rotor")
    for p, level in tqdm(fault_files, desc="Rotor faults"):
        X_exp = load_rotor_experiment(p)
        X_w = segment_rotor_experiment(X_exp)

        X_windows_list.append(X_w)
        y_list.extend([level] * ROTOR_WINDOWS_PER_EXPERIMENT)
        y_bin_list.extend([1] * ROTOR_WINDOWS_PER_EXPERIMENT)
        groups_list.extend([group_id] * ROTOR_WINDOWS_PER_EXPERIMENT)
        paths_list.extend([str(p)] * ROTOR_WINDOWS_PER_EXPERIMENT)
        exp_names_list.extend([p.stem] * ROTOR_WINDOWS_PER_EXPERIMENT)
        group_id += 1

    X_windows = np.concatenate(X_windows_list, axis=0).astype(np.float32)
    y_multi = np.array(y_list, dtype=np.int64)
    y_bin = np.array(y_bin_list, dtype=np.uint8)
    groups = np.array(groups_list, dtype=np.int64)
    paths = np.array(paths_list, dtype=object)
    exp_names = np.array(exp_names_list, dtype=object)

    expected_windows = 35 * ROTOR_WINDOWS_PER_EXPERIMENT
    if X_windows.shape != (expected_windows, ROTOR_L, ROTOR_CHANNELS):
        raise ValueError(f"Rotor debería tener shape {(expected_windows, ROTOR_L, ROTOR_CHANNELS)}, tiene {X_windows.shape}")

    class_names = np.array(
        ["Healthy", "Imbalance_Level_1", "Imbalance_Level_2", "Imbalance_Level_3", "Imbalance_Level_4"],
        dtype=object,
    )

    print("X rotor:", X_windows.shape)
    print("Conteos rotor:", dict(zip(*np.unique(y_multi, return_counts=True))))
    print("Grupos rotor:", len(np.unique(groups)))

    out_path = DIR_BASE / "dataset_base_rotor_desbalance_35exp_15sanos_20fallos.npz"
    np.savez_compressed(
        out_path,
        X_windows=X_windows,
        y_multi=y_multi,
        y_bin=y_bin,
        groups=groups,
        paths=paths,
        exp_names=exp_names,
        class_names=class_names,
        fs=np.array([ROTOR_FS], dtype=np.float32),
        L=np.array([ROTOR_L], dtype=np.int32),
        N=np.array([ROTOR_CHANNELS], dtype=np.int32),
        windows_per_experiment=np.array([ROTOR_WINDOWS_PER_EXPERIMENT], dtype=np.int32),
        description=np.array([
            "Dataset base rotor multiclase. X_windows=(1120,3200,24). "
            "35 experimentos, 32 ventanas por experimento."
        ]),
    )

    save_json(
        DIR_LOGS / "rotor_base_summary.json",
        {
            "path": str(out_path),
            "X_windows": array_stats(X_windows),
            "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_multi, return_counts=True))},
            "n_groups": int(len(np.unique(groups))),
            "class_names": class_names.tolist(),
            "groups_rule": "one group per experiment",
        },
    )

    return out_path


# =============================================================================
# 4. REPRESENTACIONES ESPECTRALES
# =============================================================================

def reconstruct_jacket_to_3d(X_flat: np.ndarray) -> np.ndarray:
    if X_flat.shape[1] != JACKET_FEATURES:
        raise ValueError(f"Jacket X debe tener {JACKET_FEATURES} features, tiene {X_flat.shape[1]}")
    return X_flat.reshape(X_flat.shape[0], JACKET_L, JACKET_CHANNELS).astype(np.float32)


def fft_log_magnitude_dataset(X_3d: np.ndarray) -> np.ndarray:
    n_samples, L, C = X_3d.shape
    n_freq = L // 2 + 1
    X_fft = np.empty((n_samples, n_freq * C), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="FFT"):
        spec = np.fft.rfft(X_3d[i], axis=0)
        feat = np.log10(1.0 + np.abs(spec) + EPS)
        X_fft[i] = feat.reshape(-1).astype(np.float32)

    return X_fft


def stft_log_power_dataset(X_3d: np.ndarray, fs: float):
    n_samples, L, C = X_3d.shape

    f_ref, t_ref, _ = stft(
        X_3d[0, :, 0],
        fs=fs,
        window=STFT_WINDOW,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )

    n_freq = len(f_ref)
    n_time = len(t_ref)
    X_stft = np.empty((n_samples, n_freq * n_time * C), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="STFT"):
        features = []
        for ch in range(C):
            _, _, Zxx = stft(
                X_3d[i, :, ch],
                fs=fs,
                window=STFT_WINDOW,
                nperseg=STFT_NPERSEG,
                noverlap=STFT_NOVERLAP,
                nfft=STFT_NFFT,
                detrend=False,
                return_onesided=True,
                boundary=None,
                padded=False,
            )
            power = np.abs(Zxx) ** 2
            features.append(np.log10(power + EPS).reshape(-1))
        X_stft[i] = np.concatenate(features).astype(np.float32)

    return X_stft, f_ref.astype(np.float32), t_ref.astype(np.float32)


def welch_psd_log_power_dataset(X_3d: np.ndarray, fs: float):
    n_samples, L, C = X_3d.shape

    f_ref, pxx_ref = welch(
        X_3d[0, :, 0],
        fs=fs,
        window=WELCH_WINDOW,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )

    n_freq = len(f_ref)
    X_welch = np.empty((n_samples, n_freq * C), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="Welch"):
        features = []
        for ch in range(C):
            _, pxx = welch(
                X_3d[i, :, ch],
                fs=fs,
                window=WELCH_WINDOW,
                nperseg=WELCH_NPERSEG,
                noverlap=WELCH_NOVERLAP,
                nfft=WELCH_NFFT,
                detrend="constant",
                return_onesided=True,
                scaling="density",
            )
            # Fórmula Welch PSD log-power usada para evitar colapso de varianza:
            # log10(PSD + EPS), consistente con el flujo original Welch.
            features.append(np.log10(pxx + EPS))
        X_welch[i] = np.concatenate(features).astype(np.float32)

    return X_welch, f_ref.astype(np.float32)


def save_representation_npz(
    dataset: str,
    rep_key: str,
    X_rep: np.ndarray,
    y_multi: np.ndarray,
    y_bin: np.ndarray,
    groups: np.ndarray,
    class_names: np.ndarray,
    metadata: dict,
) -> Path:
    rep_folder = DIR_REP / REPRESENTATIONS[rep_key]["folder"]
    rep_folder.mkdir(parents=True, exist_ok=True)

    out_path = rep_folder / f"{dataset}_{rep_key}.npz"

    np.savez_compressed(
        out_path,
        X=X_rep.astype(np.float32),
        y_multi=y_multi.astype(np.int64),
        y_bin=y_bin.astype(np.uint8),
        groups=groups.astype(np.int64),
        class_names=class_names,
        metadata=np.array([json.dumps(metadata, ensure_ascii=False)]),
    )

    return out_path


def generate_all_representations(jacket_base_path: Path, rotor_base_path: Path) -> dict:
    print_header("PASO 2 - GENERAR REPRESENTACIONES FFT, STFT Y WELCH")

    outputs = {}

    # Jacket.
    data_j = np.load(jacket_base_path, allow_pickle=True)
    X_j_3d = reconstruct_jacket_to_3d(data_j["X"].astype(np.float32))
    y_j = data_j["y_multi"]
    yb_j = data_j["y_bin"]
    g_j = data_j["groups"]
    cn_j = data_j["class_names"]

    print_subheader("Representaciones jacket")

    X_fft_j = fft_log_magnitude_dataset(X_j_3d)
    outputs["jacket_fft"] = save_representation_npz(
        "jacket", "fft", X_fft_j, y_j, yb_j, g_j, cn_j,
        {
            "representation": "FFT log-magnitude",
            "formula": "log10(1 + abs(rFFT(x)) + eps)",
            "features": int(X_fft_j.shape[1]),
            "expected_features_article": 29016,
            "fs": JACKET_FS,
        },
    )
    validate_numeric_matrix("jacket FFT", X_fft_j)

    X_stft_j, f_stft_j, t_stft_j = stft_log_power_dataset(X_j_3d, JACKET_FS)
    outputs["jacket_stft"] = save_representation_npz(
        "jacket", "stft", X_stft_j, y_j, yb_j, g_j, cn_j,
        {
            "representation": "STFT log-power",
            "formula": "log10(abs(STFT(x))^2 + eps)",
            "features": int(X_stft_j.shape[1]),
            "expected_features_article": 52632,
            "fs": JACKET_FS,
            "nperseg": STFT_NPERSEG,
            "noverlap": STFT_NOVERLAP,
            "nfft": STFT_NFFT,
            "window": STFT_WINDOW,
            "n_freq": int(len(f_stft_j)),
            "n_time": int(len(t_stft_j)),
        },
    )
    validate_numeric_matrix("jacket STFT", X_stft_j)

    X_welch_j, f_welch_j = welch_psd_log_power_dataset(X_j_3d, JACKET_FS)
    outputs["jacket_welch"] = save_representation_npz(
        "jacket", "welch", X_welch_j, y_j, yb_j, g_j, cn_j,
        {
            "representation": "Welch PSD log-power",
            "formula": "log10(WelchPSD(x) + eps)",
            "features": int(X_welch_j.shape[1]),
            "expected_features_article": 6168,
            "fs": JACKET_FS,
            "nperseg": WELCH_NPERSEG,
            "noverlap": WELCH_NOVERLAP,
            "nfft": WELCH_NFFT,
            "window": WELCH_WINDOW,
            "n_freq": int(len(f_welch_j)),
        },
    )
    validate_numeric_matrix("jacket Welch", X_welch_j)
    if float(np.std(X_welch_j)) == 0.0:
        raise ValueError("Jacket Welch quedó con desviación estándar 0. Revisa la fórmula Welch log-power.")

    del X_fft_j, X_stft_j, X_welch_j, X_j_3d
    gc.collect()

    # Rotor.
    data_r = np.load(rotor_base_path, allow_pickle=True)
    X_r_3d = data_r["X_windows"].astype(np.float32)
    y_r = data_r["y_multi"]
    yb_r = data_r["y_bin"]
    g_r = data_r["groups"]
    cn_r = data_r["class_names"]

    print_subheader("Representaciones rotor")

    X_fft_r = fft_log_magnitude_dataset(X_r_3d)
    outputs["rotor_fft"] = save_representation_npz(
        "rotor", "fft", X_fft_r, y_r, yb_r, g_r, cn_r,
        {
            "representation": "FFT log-magnitude",
            "formula": "log10(1 + abs(rFFT(x)) + eps)",
            "features": int(X_fft_r.shape[1]),
            "expected_features_article": 38424,
            "fs": ROTOR_FS,
        },
    )
    validate_numeric_matrix("rotor FFT", X_fft_r)

    X_stft_r, f_stft_r, t_stft_r = stft_log_power_dataset(X_r_3d, ROTOR_FS)
    outputs["rotor_stft"] = save_representation_npz(
        "rotor", "stft", X_stft_r, y_r, yb_r, g_r, cn_r,
        {
            "representation": "STFT log-power",
            "formula": "log10(abs(STFT(x))^2 + eps)",
            "features": int(X_stft_r.shape[1]),
            "expected_features_article": 74304,
            "fs": ROTOR_FS,
            "nperseg": STFT_NPERSEG,
            "noverlap": STFT_NOVERLAP,
            "nfft": STFT_NFFT,
            "window": STFT_WINDOW,
            "n_freq": int(len(f_stft_r)),
            "n_time": int(len(t_stft_r)),
        },
    )
    validate_numeric_matrix("rotor STFT", X_stft_r)

    X_welch_r, f_welch_r = welch_psd_log_power_dataset(X_r_3d, ROTOR_FS)
    outputs["rotor_welch"] = save_representation_npz(
        "rotor", "welch", X_welch_r, y_r, yb_r, g_r, cn_r,
        {
            "representation": "Welch PSD log-power",
            "formula": "log10(WelchPSD(x) + eps)",
            "features": int(X_welch_r.shape[1]),
            "expected_features_article": 6168,
            "fs": ROTOR_FS,
            "nperseg": WELCH_NPERSEG,
            "noverlap": WELCH_NOVERLAP,
            "nfft": WELCH_NFFT,
            "window": WELCH_WINDOW,
            "n_freq": int(len(f_welch_r)),
        },
    )
    validate_numeric_matrix("rotor Welch", X_welch_r)
    if float(np.std(X_welch_r)) == 0.0:
        raise ValueError("Rotor Welch quedó con desviación estándar 0. Revisa la fórmula Welch log-power.")

    del X_fft_r, X_stft_r, X_welch_r, X_r_3d
    gc.collect()

    save_json(DIR_LOGS / "representations_outputs.json", {k: str(v) for k, v in outputs.items()})
    return outputs


# =============================================================================
# 5. CV5 + TOP-K + STANDARD SCALER
# =============================================================================

def verify_groups_single_label(y: np.ndarray, groups: np.ndarray) -> None:
    for g in np.unique(groups):
        labels = np.unique(y[groups == g])
        if len(labels) != 1:
            raise ValueError(f"El grupo {g} contiene más de una clase: {labels}")


def topk_variance_indices(X_train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    variances = np.var(X_train, axis=0).astype(np.float32)
    feature_indices = np.arange(X_train.shape[1])
    order = np.lexsort((feature_indices, -variances))
    selected = order[:min(k, X_train.shape[1])]
    return selected.astype(np.int64), variances


def create_cv5_for_representation(rep_path: Path, rep_key: str, dataset_name: str) -> list[Path]:
    data = np.load(rep_path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y_multi"].astype(np.int64)
    y_bin = data["y_bin"].astype(np.uint8)
    groups = data["groups"].astype(np.int64)
    class_names = data["class_names"]

    verify_groups_single_label(y, groups)

    out_dir = DIR_CV5 / f"{REPRESENTATIONS[rep_key]['folder']}_cv5_topk4096"
    out_dir.mkdir(parents=True, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    generated_files = []

    print_subheader(f"CV5 {dataset_name.upper()} - {REPRESENTATIONS[rep_key]['name']}")
    print("X:", X.shape, "y:", y.shape, "groups:", len(np.unique(groups)))

    for fold_id, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        y_bin_train = y_bin[train_idx]
        y_bin_test = y_bin[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        # Verificación de fuga por grupos.
        overlap_groups = set(groups_train.tolist()) & set(groups_test.tolist())
        if overlap_groups:
            raise ValueError(f"Fold {fold_id}: fuga de grupos train/test: {len(overlap_groups)} grupos repetidos.")

        selected_idx, variances = topk_variance_indices(X_train_raw, TOP_K)

        X_train_sel = X_train_raw[:, selected_idx]
        X_test_sel = X_test_raw[:, selected_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_sel).astype(np.float32)
        X_test = scaler.transform(X_test_sel).astype(np.float32)

        if X_train.shape[1] != TOP_K or X_test.shape[1] != TOP_K:
            raise ValueError(f"Fold {fold_id}: Top-K no produjo {TOP_K} características.")

        validate_numeric_matrix(f"{dataset_name}_{rep_key}_fold{fold_id}_X_train", X_train)
        validate_numeric_matrix(f"{dataset_name}_{rep_key}_fold{fold_id}_X_test", X_test)

        out_path = out_dir / f"{dataset_name}_{rep_key}_fold{fold_id}_topk4096.npz"

        np.savez_compressed(
            out_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train.astype(np.int64),
            y_test=y_test.astype(np.int64),
            y_bin_train=y_bin_train.astype(np.uint8),
            y_bin_test=y_bin_test.astype(np.uint8),
            train_idx=train_idx.astype(np.int64),
            test_idx=test_idx.astype(np.int64),
            groups_train=groups_train.astype(np.int64),
            groups_test=groups_test.astype(np.int64),
            selected_idx=selected_idx.astype(np.int64),
            feature_variances=variances.astype(np.float32),
            class_names=class_names,
            dataset=np.array([dataset_name]),
            representation=np.array([REPRESENTATIONS[rep_key]["name"]]),
            rep_key=np.array([rep_key]),
            fold_id=np.array([fold_id], dtype=np.int32),
            top_k=np.array([TOP_K], dtype=np.int32),
            seed=np.array([SEED], dtype=np.int32),
            description=np.array([
                "Fold CV5 con Top-K por varianza y StandardScaler. "
                "Top-K y scaler ajustados exclusivamente con X_train del fold."
            ]),
        )

        # Guardar scaler y selector por trazabilidad.
        joblib.dump(
            {
                "selected_idx": selected_idx,
                "feature_variances": variances,
                "top_k": TOP_K,
                "fitted_only_with": "X_train of this fold",
            },
            out_dir / f"{dataset_name}_{rep_key}_fold{fold_id}_selector.joblib",
        )
        joblib.dump(
            scaler,
            out_dir / f"{dataset_name}_{rep_key}_fold{fold_id}_standard_scaler.joblib",
        )

        generated_files.append(out_path)

        print(
            f"Fold {fold_id}: train={X_train.shape}, test={X_test.shape}, "
            f"classes train={dict(zip(*np.unique(y_train, return_counts=True)))}, "
            f"classes test={dict(zip(*np.unique(y_test, return_counts=True)))}"
        )

    return generated_files


def create_all_cv5(representation_outputs: dict) -> dict:
    print_header("PASO 3 - CREAR CV5 + TOP-K=4096 + STANDARD SCALER")

    all_cv_files = {}

    for rep_key in ["fft", "stft", "welch"]:
        for dataset_name in ["jacket", "rotor"]:
            key = f"{dataset_name}_{rep_key}"
            files = create_cv5_for_representation(
                rep_path=representation_outputs[key],
                rep_key=rep_key,
                dataset_name=dataset_name,
            )
            all_cv_files[key] = [str(p) for p in files]
            gc.collect()

    save_json(DIR_LOGS / "cv5_topk4096_outputs.json", all_cv_files)
    return all_cv_files


# =============================================================================
# 6. README Y CONFIGURACIÓN
# =============================================================================

def write_readme() -> None:
    readme = f"""REPRODUCIBILIDAD DEL ARTÍCULO MULTITASK

Artículo:
Improving Vibration-Based Classification in a Wind Turbine Jacket Structure and Rotor
through Multitask Learning and Spectral Representations

Carpeta raíz:
{REPRO_ROOT}

Script 1:
01_crear_datasets_representaciones_cv5.py

Qué genera este script:
1. Dataset base jacket multiclase:
   - 5740 muestras
   - 58008 características
   - 5 clases: Healthy + Crack_Level_1..4
   - groups = una muestra por grupo

2. Dataset base rotor multiclase:
   - 1120 ventanas
   - 35 experimentos
   - 32 ventanas por experimento
   - 5 clases: Healthy + Imbalance_Level_1..4
   - groups = identificador de experimento

3. Representaciones:
   - FFT log-magnitude
   - STFT log-power
   - Welch PSD log-power

4. Validación cruzada:
   - StratifiedGroupKFold
   - 5 folds
   - Top-K por varianza, K=4096
   - StandardScaler ajustado solo con train del fold

Parámetros:
- Seed = {SEED}
- STFT: window={STFT_WINDOW}, nperseg={STFT_NPERSEG}, noverlap={STFT_NOVERLAP}, nfft={STFT_NFFT}
- Welch: window={WELCH_WINDOW}, nperseg={WELCH_NPERSEG}, noverlap={WELCH_NOVERLAP}, nfft={WELCH_NFFT}
- Top-K = {TOP_K}

Orden de ejecución:
1) Ejecutar este script.
2) Ejecutar 02_entrenar_modelos_y_generar_resultados.py
"""
    (REPRO_ROOT / "README_REPRODUCIBILIDAD.txt").write_text(readme, encoding="utf-8")


def save_global_config() -> None:
    config = {
        "article": "Improving Vibration-Based Classification in a Wind Turbine Jacket Structure and Rotor through Multitask Learning and Spectral Representations",
        "created_at": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "project_root": str(PROJECT_ROOT),
        "repro_root": str(REPRO_ROOT),
        "paths": {
            "jacket_mat": str(JACKET_MAT_PATH),
            "jacket_labels": str(JACKET_LABELS_PATH),
            "rotor_healthy_dir": str(ROTOR_HEALTHY_DIR),
            "rotor_faults_dir": str(ROTOR_FAULTS_DIR),
        },
        "jacket": {
            "samples": JACKET_N_SAMPLES,
            "L": JACKET_L,
            "channels": JACKET_CHANNELS,
            "features": JACKET_FEATURES,
            "fs": JACKET_FS,
        },
        "rotor": {
            "experiments": 35,
            "windows": 1120,
            "windows_per_experiment": ROTOR_WINDOWS_PER_EXPERIMENT,
            "L": ROTOR_L,
            "channels": ROTOR_CHANNELS,
            "fs": ROTOR_FS,
        },
        "representations": {
            "fft": "log10(1 + abs(rFFT(x)) + eps)",
            "stft": {
                "formula": "log10(abs(STFT(x))^2 + eps)",
                "window": STFT_WINDOW,
                "nperseg": STFT_NPERSEG,
                "noverlap": STFT_NOVERLAP,
                "nfft": STFT_NFFT,
            },
            "welch": {
                "formula": "log10(WelchPSD(x) + eps)",
                "window": WELCH_WINDOW,
                "nperseg": WELCH_NPERSEG,
                "noverlap": WELCH_NOVERLAP,
                "nfft": WELCH_NFFT,
            },
        },
        "cv": {
            "method": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "top_k": TOP_K,
            "scaler": "StandardScaler fitted only with X_train of each fold",
        },
    }
    save_json(REPRO_ROOT / "config_reproducibilidad.json", config)


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    set_global_seed(SEED)
    ensure_dirs()

    print_header("INICIO DEL SCRIPT 01 - DATASETS + REPRESENTACIONES + CV5")
    print("REPRO_ROOT:", REPRO_ROOT)

    save_global_config()

    jacket_base = create_jacket_base_dataset()
    rotor_base = create_rotor_base_dataset()

    representation_outputs = generate_all_representations(jacket_base, rotor_base)
    cv_outputs = create_all_cv5(representation_outputs)

    write_readme()

    final_summary = {
        "status": "ok",
        "finished_at": datetime.now().isoformat(),
        "jacket_base": str(jacket_base),
        "rotor_base": str(rotor_base),
        "representations": {k: str(v) for k, v in representation_outputs.items()},
        "cv_outputs": cv_outputs,
    }
    save_json(DIR_LOGS / "script01_final_summary.json", final_summary)

    print_header("SCRIPT 01 TERMINADO CORRECTAMENTE")
    print("Carpeta reproducible:")
    print(REPRO_ROOT)
    print("Siguiente paso:")
    print("Ejecutar 02_entrenar_modelos_y_generar_resultados.py")


if __name__ == "__main__":
    main()
