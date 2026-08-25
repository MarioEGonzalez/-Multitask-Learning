# -*- coding: utf-8 -*-
r"""
05_topk_sensitivity_revision.py

Experimento nuevo para la revisión (Major Revision):
Responde al comentario agrupado G4 (Reviewer 1, Comentario 10 / Reviewer 3, Comentario 6):

    "Top-K feature selection uses variance only, which ignores class
    discriminability... Compare against supervised selection and include
    a sensitivity analysis on K."

Qué hace este script:
  1) Para cada combinación dataset x representación (6 en total: jacket/rotor x
     FFT/STFT/Welch), reconstruye EXACTAMENTE el mismo split CV5
     (StratifiedGroupKFold, n_splits=5, shuffle=True, random_state=SEED) que usa
     01_crear_datasets_representaciones_cv5.py, partiendo de las representaciones
     COMPLETAS ya cacheadas en 03_representaciones (antes de Top-K). No se
     recalculan FFT/STFT/Welch desde las señales crudas.
  2) Para cada fold, para cada valor de K en TOPK_VALUES, selecciona features de
     dos formas (ajustadas SOLO con X_train del fold, igual que en el pipeline
     original):
       - "variance"        -> el mismo criterio ya usado en el artículo
                               (topk_variance_indices, idéntico a 01_...).
       - "supervised_anova" -> selección supervisada real: SelectKBest con
                               ANOVA F-value (f_classif) entre cada feature y
                               la etiqueta de clase. Responde directamente a la
                               crítica de "ignora la discriminabilidad de clase".
  3) Entrena, sobre cada selección, un RandomForestClassifier con LA MISMA
     configuración ya reportada en el artículo para el baseline clásico
     (n_estimators=500, class_weight="balanced", random_state=SEED) -- se usa
     RandomForest en vez de TabNet/Multitask por costo computacional: aquí se
     entrenan 6 combinaciones x ~6-7 valores de K x 2 selectores x 5 folds
     (várias centenas de entrenamientos), y RandomForest ya es un baseline
     establecido y reportado en el artículo (Sección 3.6, Table 15).
  4) Guarda un CSV consolidado por fold y un CSV agregado (media +/- std entre
     folds) con accuracy, balanced accuracy y Macro-F1 para cada
     (dataset, representación, K, selector).
  5) Genera una figura por combinación dataset-representación (Macro-F1 vs K,
     una línea por selector, con barras de error = std entre folds) y una
     figura resumen con las 6 combinaciones en una grilla.

Salida principal:
  <REPRO_ROOT>/08_experimentos_revision/05_topk_sensitivity/

Requisitos:
  pip install numpy pandas scikit-learn matplotlib tqdm
"""

import os
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# =============================================================================
# 0. CONFIGURACIÓN GENERAL (idéntica a 01_crear_datasets_representaciones_cv5.py)
# =============================================================================

SEED = 42
N_SPLITS = 5

PROJECT_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold")
REPRO_ROOT = PROJECT_ROOT / "reproducible"

DIR_REP = REPRO_ROOT / "03_representaciones"

OUT_REVISION = REPRO_ROOT / "08_experimentos_revision"
OUT_SENSITIVITY = OUT_REVISION / "05_topk_sensitivity"
OUT_SENSITIVITY.mkdir(parents=True, exist_ok=True)

REPRESENTATIONS = {
    "fft": {"name": "FFT log-magnitude", "folder": "01_FFT_log_magnitude"},
    "stft": {"name": "STFT log-power", "folder": "02_STFT_log_power"},
    "welch": {"name": "Welch PSD log-power", "folder": "03_Welch_PSD_log_power"},
}
DATASETS = ["jacket", "rotor"]

# Barrido de K. Cada valor se recorta automáticamente al número total de
# características disponibles en cada representación (p. ej. Welch solo tiene
# 6168 features en total, así que 8192 no aplica ahí). También se añade
# automáticamente el punto "todas las features" (equivalente a "no selection")
# como el valor más alto de cada barrido.
TOPK_VALUES_BASE = [256, 512, 1024, 2048, 4096, 8192]

# RandomForest: MISMA configuración ya reportada en el artículo
# (Sección 3.6 / Table 15) para el baseline clásico.
RF_PARAMS = dict(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced")

# Si quieres acelerar una primera corrida de prueba, reduce esta lista, p. ej.:
# TOPK_VALUES_BASE = [1024, 4096]
# y/o comenta datasets/representaciones en el loop principal.


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
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def topk_variance_indices(X_train: np.ndarray, k: int) -> np.ndarray:
    """Idéntico a 01_crear_datasets_representaciones_cv5.py::topk_variance_indices."""
    variances = np.var(X_train, axis=0).astype(np.float32)
    feature_indices = np.arange(X_train.shape[1])
    order = np.lexsort((feature_indices, -variances))
    selected = order[:min(k, X_train.shape[1])]
    return selected.astype(np.int64)


def topk_supervised_anova_indices(X_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Selección supervisada: ANOVA F-value (f_classif) entre cada feature y la
    clase, ajustada SOLO con X_train/y_train del fold (igual que la selección
    por varianza). Devuelve los índices de las K features con mayor F-score.
    """
    k_eff = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_eff)
    selector.fit(X_train, y_train)
    scores = selector.scores_.copy()
    # NaN puede aparecer si una feature es constante en train; se manda al final.
    scores = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(-scores, kind="stable")
    return order[:k_eff].astype(np.int64)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def load_full_representation(dataset: str, rep_key: str):
    rep_folder = DIR_REP / REPRESENTATIONS[rep_key]["folder"]
    path = rep_folder / f"{dataset}_{rep_key}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la representación completa en {path}. "
            f"Este script requiere que 01_crear_datasets_representaciones_cv5.py "
            f"ya se haya corrido al menos una vez (03_representaciones cacheado)."
        )
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y_multi"].astype(np.int64)
    groups = data["groups"].astype(np.int64)
    return X, y, groups


def k_sweep_for_combination(dataset: str, rep_key: str) -> pd.DataFrame:
    rep_name = REPRESENTATIONS[rep_key]["name"]
    print_header(f"K-SENSITIVITY - {dataset.upper()} - {rep_name}")

    X, y, groups = load_full_representation(dataset, rep_key)
    n_features_total = X.shape[1]
    print(f"X: {X.shape}, features totales={n_features_total}, clases={len(np.unique(y))}")

    # Barrido de K recortado al total de features disponibles, sin duplicados,
    # más el punto "todas las features" (equivalente a no aplicar selección).
    k_values = sorted(set(min(k, n_features_total) for k in TOPK_VALUES_BASE))
    if n_features_total not in k_values:
        k_values.append(n_features_total)
    k_values = sorted(k_values)
    print(f"Valores de K evaluados (recortados a {n_features_total} features totales): {k_values}")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for k in k_values:
            for selector_name in ["variance", "supervised_anova"]:
                t0 = time.time()

                if selector_name == "variance":
                    sel_idx = topk_variance_indices(X_train_raw, k)
                else:
                    sel_idx = topk_supervised_anova_indices(X_train_raw, y_train, k)

                X_train_sel = X_train_raw[:, sel_idx]
                X_test_sel = X_test_raw[:, sel_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train_sel).astype(np.float32)
                X_test_s = scaler.transform(X_test_sel).astype(np.float32)

                clf = RandomForestClassifier(**RF_PARAMS)
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)

                metrics = compute_metrics(y_test, y_pred)
                elapsed = time.time() - t0

                rows.append({
                    "dataset": dataset,
                    "representation": rep_name,
                    "rep_key": rep_key,
                    "fold_id": fold_id,
                    "k": k,
                    "k_is_all_features": bool(k == n_features_total),
                    "selector": selector_name,
                    "n_features_total": n_features_total,
                    "seconds": round(elapsed, 2),
                    **metrics,
                })

                print(
                    f"  fold={fold_id} K={k:>6d} selector={selector_name:<16s} "
                    f"acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
                    f"f1_macro={metrics['f1_macro']:.4f} ({elapsed:.1f}s)"
                )

        del X_train_raw, X_test_raw
        gc.collect()

    del X, y, groups
    gc.collect()

    return pd.DataFrame(rows)


def make_plot_for_combination(df_combo: pd.DataFrame, dataset: str, rep_key: str) -> None:
    rep_name = REPRESENTATIONS[rep_key]["name"]
    agg = (
        df_combo.groupby(["k", "selector"])["f1_macro"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = {"variance": "#1f77b4", "supervised_anova": "#d62728"}
    labels = {"variance": "Top-K por varianza (usado en el artículo)",
              "supervised_anova": "Top-K supervisado (ANOVA F-value)"}

    for selector_name in ["variance", "supervised_anova"]:
        sub = agg[agg["selector"] == selector_name].sort_values("k")
        ax.errorbar(
            sub["k"], sub["mean"], yerr=sub["std"].fillna(0.0),
            marker="o", capsize=3, linewidth=1.8,
            color=colors[selector_name], label=labels[selector_name],
        )

    ax.axvline(4096, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(4096, ax.get_ylim()[0], "  K=4096 (artículo)", rotation=90,
            va="bottom", ha="left", fontsize=8, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("K (número de características seleccionadas)")
    ax.set_ylabel("Macro-F1 (media $\\pm$ std entre folds)")
    ax.set_title(f"Sensibilidad en K -- {dataset} / {rep_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_base = OUT_SENSITIVITY / f"fig_topk_sensitivity_{dataset}_{rep_key}"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_summary_grid(df_all: pd.DataFrame) -> None:
    combos = [(d, r) for d in DATASETS for r in REPRESENTATIONS.keys()]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    colors = {"variance": "#1f77b4", "supervised_anova": "#d62728"}
    labels = {"variance": "Varianza (artículo)", "supervised_anova": "Supervisado (ANOVA F)"}

    for ax, (dataset, rep_key) in zip(axes.flat, combos):
        df_combo = df_all[(df_all["dataset"] == dataset) & (df_all["rep_key"] == rep_key)]
        agg = df_combo.groupby(["k", "selector"])["f1_macro"].agg(["mean", "std"]).reset_index()
        for selector_name in ["variance", "supervised_anova"]:
            sub = agg[agg["selector"] == selector_name].sort_values("k")
            ax.errorbar(sub["k"], sub["mean"], yerr=sub["std"].fillna(0.0),
                        marker="o", markersize=3, capsize=2, linewidth=1.4,
                        color=colors[selector_name], label=labels[selector_name])
        ax.axvline(4096, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_title(f"{dataset} / {REPRESENTATIONS[rep_key]['name']}", fontsize=10)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="lower right", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    for ax in axes[1, :]:
        ax.set_xlabel("K")

    fig.suptitle("Análisis de sensibilidad en K: Top-K por varianza vs. Top-K supervisado (ANOVA F)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_base = OUT_SENSITIVITY / "fig_topk_sensitivity_summary_grid"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_global_seed(SEED)
    print_header("ANALISIS DE SENSIBILIDAD EN K + COMPARACION CONTRA SELECTOR SUPERVISADO")
    print("Responde a: Reviewer 1 Comentario 10 (P10) / Reviewer 3 Comentario 6 (grupo G4)")

    all_rows = []
    for dataset in DATASETS:
        for rep_key in REPRESENTATIONS.keys():
            df_combo = k_sweep_for_combination(dataset, rep_key)
            all_rows.append(df_combo)

            # Guardado incremental por combinación (por si el proceso se interrumpe).
            df_combo.to_csv(
                OUT_SENSITIVITY / f"topk_sensitivity_{dataset}_{rep_key}_per_fold.csv",
                index=False, encoding="utf-8-sig",
            )
            make_plot_for_combination(df_combo, dataset, rep_key)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(OUT_SENSITIVITY / "resumen_topk_sensitivity_per_fold.csv",
                   index=False, encoding="utf-8-sig")

    # Tabla agregada (media +/- std entre folds), lista para convertir en tabla LaTeX.
    agg = (
        df_all.groupby(["dataset", "representation", "rep_key", "k", "k_is_all_features", "selector"])
        [["accuracy", "balanced_accuracy", "f1_macro"]]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT_SENSITIVITY / "resumen_topk_sensitivity_aggregated.csv",
               index=False, encoding="utf-8-sig")

    make_summary_grid(df_all)

    print_subheader("LISTO")
    print(f"CSV por fold:      {OUT_SENSITIVITY / 'resumen_topk_sensitivity_per_fold.csv'}")
    print(f"CSV agregado:      {OUT_SENSITIVITY / 'resumen_topk_sensitivity_aggregated.csv'}")
    print(f"Figuras:           {OUT_SENSITIVITY}")
    print("\nPor favor comparte AMBOS archivos CSV (per_fold y aggregated) para redactar la respuesta al revisor.")


if __name__ == "__main__":
    main()