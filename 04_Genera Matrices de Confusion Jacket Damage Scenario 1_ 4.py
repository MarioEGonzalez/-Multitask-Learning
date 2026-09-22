# -*- coding: utf-8 -*-
"""Regenera las Figuras 10--19 de Wind con etiquetas Jacket.

No vuelve a entrenar modelos ni altera predicciones: usa los archivos OOF y
los latentes de los experimentos reproducibles ya realizados.


from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

DEFAULT_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold\reproducible")
REP_INFO = {
    "fft": "FFT log-magnitude",
    "stft": "STFT log-power",
    "welch": "Welch PSD log-power",
}
REP_SHORT = {"fft": "FFT", "stft": "STFT", "welch": "Welch PSD"}
JACKET_CLASSES = ["Healthy"] + [f"Damage Scenario {k}" for k in range(1, 5)]
ROTOR_CLASSES = ["Healthy"] + [f"Imbalance L{k}" for k in range(1, 5)]
ROTOR_CLASSES_NORMALIZED = ["Healthy"] + [f"Imbalance_Level_{k}" for k in range(1, 5)]
CLASS_COLORS = ["#8ecae6", "#ffb703", "#a8dadc", "#cdb4db", "#fb8500"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def save_figure(fig, basename: str, pdf_dir: Path, png_dir: Path) -> None:
    fig.savefig(pdf_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(png_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {basename}.pdf")


def load_oof(root: Path, rep: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    path = (root / "06_resultados_multitask" / rep /
            f"Multitask_{dataset}_{rep}_oof_predictions.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Faltan las predicciones originales: {path}")
    df = pd.read_csv(path)
    for column in ("y_true", "y_pred"):
        if column not in df:
            raise ValueError(f"La columna {column!r} no existe en {path}")
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)
    if len(y_true) == 0 or not all(np.isin(arr, np.arange(5)).all() for arr in (y_true, y_pred)):
        raise ValueError(f"Se esperaban clases enteras 0..4 y datos no vacíos: {path}")
    return y_true, y_pred


def confusion_figure(cm: np.ndarray, classes: list[str], title: str,
                     normalized: bool):
    # Un archivo por panel, tal y como se referencia en el manuscrito LaTeX.
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    if normalized:
        denom = cm.sum(axis=1, keepdims=True)
        image_data = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
        im = ax.imshow(image_data, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    else:
        image_data = cm
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest",
                       vmin=0, vmax=max(float(cm.max()), 1) * 1.20)

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, labels=classes, rotation=42, ha="right")
    ax.set_yticks(ticks, labels=classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, pad=13)
    for i in range(5):
        for j in range(5):
            value = f"{image_data[i, j]:.2f}" if normalized else str(int(cm[i, j]))
            ax.text(j, i, value, ha="center", va="center", color="black",
                    fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_confusion_matrices(root: Path, pdf_dir: Path, png_dir: Path) -> None:
    for rep in REP_INFO:
        for dataset in ("jacket", "rotor"):
            y_true, y_pred = load_oof(root, rep, dataset)
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(5))
            for normalized in (False, True):
                if dataset == "jacket":
                    classes = JACKET_CLASSES
                else:
                    classes = ROTOR_CLASSES_NORMALIZED if normalized else ROTOR_CLASSES
                suffix = "_normalized" if normalized else ""
                if normalized:
                    title = (f"Normalized Confusion Matrix - Multitask - "
                             f"{dataset.title()} - {REP_INFO[rep]}")
                else:
                    title = (f"Global Confusion Matrix - Multitask "
                             f"{REP_SHORT[rep]} - {dataset.title()}")
                fig = confusion_figure(cm, classes, title, normalized)
                save_figure(fig, f"cm_multitask_{dataset}_{rep}{suffix}", pdf_dir, png_dir)


def load_latents(root: Path, rep: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    folder = root / "06_resultados_multitask" / rep / "latents"
    z_parts, y_parts = [], []
    for fold in range(1, 6):
        path = folder / f"multitask_{rep}_fold{fold}_latents_test.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Faltan los latentes originales: {path}")
        with np.load(path) as data:
            z_parts.append(data[f"Z_{dataset}"].astype(np.float32))
            y_parts.append(data[f"y_{dataset}"].astype(int))
    z = np.vstack(z_parts)
    y = np.concatenate(y_parts)
    if z.ndim != 2 or z.shape[1] < 3 or len(y) != len(z) or not np.isin(y, np.arange(5)).all():
        raise ValueError(f"Latentes o clases inesperados: {folder}")
    return z, y


def make_pca(root: Path, pdf_dir: Path, png_dir: Path) -> None:
    # El artículo contiene PCA STFT y Welch; FFT no se incluye en esas figuras.
    for rep in ("stft", "welch"):
        for dataset in ("jacket", "rotor"):
            z, y = load_latents(root, rep, dataset)
            # Conserva el cálculo del script original: PCA de los latentes OOF.
            pca = PCA(n_components=3, random_state=42)
            z_pca = pca.fit_transform(z)
            ratios = pca.explained_variance_ratio_ * 100
            classes = JACKET_CLASSES if dataset == "jacket" else ROTOR_CLASSES

            fig, ax = plt.subplots(figsize=(9.5, 7.2))
            for idx, name in enumerate(classes):
                mask = y == idx
                ax.scatter(z_pca[mask, 0], z_pca[mask, 1], s=26, alpha=0.78,
                           color=CLASS_COLORS[idx], edgecolors="black",
                           linewidths=0.25, label=name)
            ax.set_title(f"PCA 2D latent space - {dataset.title()} - {REP_SHORT[rep]}")
            ax.set_xlabel(f"PC1 ({ratios[0]:.2f}%)")
            ax.set_ylabel(f"PC2 ({ratios[1]:.2f}%)")
            ax.grid(alpha=0.25)
            ax.legend(frameon=True, loc="best")
            fig.tight_layout()
            save_figure(fig, f"fig_pca2d_latent_{dataset}_{rep}", pdf_dir, png_dir)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            for idx, name in enumerate(classes):
                mask = y == idx
                ax.scatter(z_pca[mask, 0], z_pca[mask, 1], z_pca[mask, 2],
                           s=22, alpha=0.78, color=CLASS_COLORS[idx],
                           edgecolors="black", linewidths=0.20, label=name)
            ax.set_title(f"PCA 3D latent space - {dataset.title()} - {REP_SHORT[rep]}", pad=20)
            ax.set_xlabel(f"PC1 ({ratios[0]:.2f}%)", labelpad=10)
            ax.set_ylabel(f"PC2 ({ratios[1]:.2f}%)", labelpad=10)
            ax.set_zlabel(f"PC3 ({ratios[2]:.2f}%)", labelpad=10)
            ax.legend(frameon=True, loc="best")
            fig.tight_layout()
            save_figure(fig, f"fig_pca3d_latent_{dataset}_{rep}", pdf_dir, png_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recrear figuras 10--19 con Damage Scenario 1--4")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Ruta de la carpeta reproducible con 06_resultados_multitask")
    parser.add_argument("--out", type=Path, default=None,
                        help="Carpeta de salida opcional (por defecto 09_figuras_resultados_articulo/figuras_revisor3)")
    args = parser.parse_args()
    root = args.root
    out = args.out or root / "09_figuras_resultados_articulo" / "figuras_revisor3"
    pdf_dir, png_dir = out / "pdf", out / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    print("Regenerando matrices de confusión desde predicciones OOF:")
    make_confusion_matrices(root, pdf_dir, png_dir)
    print("Regenerando PCA STFT y Welch desde latentes OOF:")
    make_pca(root, pdf_dir, png_dir)
    print(f"Listo: 20 PDF y 20 PNG en {out}")
    print("Copia los 20 PDF de la subcarpeta pdf/ a Figures/ en Overleaf.")


if __name__ == "__main__":
    main()
