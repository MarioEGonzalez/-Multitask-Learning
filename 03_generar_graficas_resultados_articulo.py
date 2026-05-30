# -*- coding: utf-8 -*-
r"""
03_generar_graficas_resultados_articulo.py

Genera las gráficas del apartado de resultados del artículo multitask a partir de
los resultados producidos por los scripts reproducibles 01 y 02.

Entrada esperada:
    C:\Users\Usuario\Documents\5. Multi Task k fold\reproducible

Salidas:
    reproducible\09_figuras_resultados_articulo

Figuras generadas:
    1) Matrices de confusión globales multitask para FFT, STFT y Welch.
    2) Comparación single-task vs multitask para Jacket.
    3) Comparación single-task vs multitask para Rotor.
    4) Barras de mejora absoluta Multitask - Single-task.
    5) PCA 2D y PCA 3D del espacio latente multitask por representación y dataset.
    6) Excel con datos fuente de las figuras.

Requisitos:
    pip install numpy pandas matplotlib scikit-learn openpyxl
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


# =============================================================================
# 0. CONFIGURACIÓN GENERAL
# =============================================================================

REPRO_ROOT = Path(r"C:\Users\Usuario\Documents\5. Multi Task k fold\reproducible")

RESULTS_TABNET = REPRO_ROOT / "05_resultados_tabnet"
RESULTS_MULTITASK = REPRO_ROOT / "06_resultados_multitask"
RESULTS_COMPARISON = REPRO_ROOT / "07_resultados_comparativos"

OUT_ROOT = REPRO_ROOT / "09_figuras_resultados_articulo"
OUT_PNG = OUT_ROOT / "png"
OUT_PDF = OUT_ROOT / "pdf"
OUT_CSV = OUT_ROOT / "csv_datos_fuente"
OUT_XLSX = OUT_ROOT / "excel_datos_fuente"

for d in [OUT_ROOT, OUT_PNG, OUT_PDF, OUT_CSV, OUT_XLSX]:
    d.mkdir(parents=True, exist_ok=True)


REP_INFO = {
    "fft": {"label": "FFT", "long": "FFT log-magnitude", "folder": "fft"},
    "stft": {"label": "STFT", "long": "STFT log-power", "folder": "stft"},
    "welch": {"label": "Welch", "long": "Welch PSD log-power", "folder": "welch"},
}

DATASET_INFO = {
    "jacket": {
        "title": "Jacket",
        "class_names": ["Healthy", "Crack L1", "Crack L2", "Crack L3", "Crack L4"],
    },
    "rotor": {
        "title": "Rotor",
        "class_names": ["Healthy", "Imbalance L1", "Imbalance L2", "Imbalance L3", "Imbalance L4"],
    },
}

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced\nAccuracy",
    "f1_macro": "Macro-F1",
    "f1_weighted": "Weighted-F1",
}

# Colores claros, adecuados para artículo.
REP_COLORS = {
    "FFT log-magnitude": "#8ecae6",
    "STFT log-power": "#cdb4db",
    "Welch PSD log-power": "#a8dadc",
}

CLASS_COLORS = ["#8ecae6", "#ffb703", "#a8dadc", "#cdb4db", "#fb8500"]

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
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


# =============================================================================
# 1. UTILIDADES
# =============================================================================

def print_header(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def save_figure(fig, basename: str) -> None:
    pdf_path = OUT_PDF / f"{basename}.pdf"
    png_path = OUT_PNG / f"{basename}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print("Guardado:", pdf_path)
    print("Guardado:", png_path)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {description}:\n{path}")


def load_oof_predictions(model_name: str, dataset: str, rep_key: str) -> pd.DataFrame:
    """
    model_name:
        'TabNet' o 'Multitask'
    """
    if model_name == "TabNet":
        path = RESULTS_TABNET / f"{dataset}_{rep_key}" / f"TabNet_{dataset}_{rep_key}_oof_predictions.csv"
    elif model_name == "Multitask":
        path = RESULTS_MULTITASK / rep_key / f"Multitask_{dataset}_{rep_key}_oof_predictions.csv"
    else:
        raise ValueError("model_name debe ser 'TabNet' o 'Multitask'.")

    require_path(path, f"predicciones OOF {model_name} {dataset} {rep_key}")
    return pd.read_csv(path)


def load_comparison_global() -> pd.DataFrame:
    path = RESULTS_COMPARISON / "comparison_global_oof_all_models.csv"
    require_path(path, "comparison_global_oof_all_models.csv")
    return pd.read_csv(path)


def load_delta_table() -> pd.DataFrame:
    path = RESULTS_COMPARISON / "comparison_single_vs_multitask_delta.csv"
    require_path(path, "comparison_single_vs_multitask_delta.csv")
    return pd.read_csv(path)


def save_source_tables_to_excel(source_tables: dict) -> None:
    excel_path = OUT_XLSX / "datos_fuente_figuras_resultados.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in source_tables.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    print("Excel datos fuente:", excel_path)


# =============================================================================
# 2. MATRICES DE CONFUSIÓN MULTITASK
# =============================================================================

def plot_confusion_matrix_ax(ax, cm, class_names, title):
    vmax = max(float(np.max(cm)), 1.0)

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=vmax * 1.20,
    )

    ax.set_title(title, color="black", pad=15)
    ax.set_xlabel("Predicted label", color="black", labelpad=10)
    ax.set_ylabel("True label", color="black", labelpad=10)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", color="black")
    ax.set_yticklabels(class_names, color="black")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="black",
                fontsize=13,
                fontweight="bold",
            )

    for spine in ax.spines.values():
        spine.set_color("black")

    return im


def generate_multitask_confusion_figures(source_tables: dict) -> None:
    print_header("FIGURAS 7-9: MATRICES DE CONFUSIÓN MULTITASK")

    for rep_key, info in REP_INFO.items():
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))

        cm_tables = []

        for ax, dataset in zip(axes, ["jacket", "rotor"]):
            df_pred = load_oof_predictions("Multitask", dataset, rep_key)

            y_true = df_pred["y_true"].astype(int).to_numpy()
            y_pred = df_pred["y_pred"].astype(int).to_numpy()

            class_names = DATASET_INFO[dataset]["class_names"]
            labels = np.arange(len(class_names))

            cm = confusion_matrix(y_true, y_pred, labels=labels)

            title = f"Global Confusion Matrix - Multitask {info['label']} - {DATASET_INFO[dataset]['title']}"
            im = plot_confusion_matrix_ax(ax, cm, class_names, title)

            cm_df = pd.DataFrame(
                cm,
                index=[f"True {c}" for c in class_names],
                columns=[f"Pred {c}" for c in class_names],
            )
            cm_df.insert(0, "dataset", dataset)
            cm_df.insert(1, "representation", info["long"])
            cm_tables.append(cm_df.reset_index().rename(columns={"index": "true_class"}))

        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.025)
        fig.suptitle(f"Multitask confusion matrices - {info['long']}", color="black", y=1.02)
        plt.tight_layout()

        basename = f"fig_confusion_multitask_{rep_key}_jacket_rotor"
        save_figure(fig, basename)
        plt.close(fig)

        source_tables[f"cm_multitask_{rep_key}"] = pd.concat(cm_tables, ignore_index=True)


# =============================================================================
# 3. COMPARACIÓN SINGLE-TASK VS MULTITASK
# =============================================================================

def generate_single_vs_multitask_lineplot(dataset: str, source_tables: dict) -> None:
    print_header(f"FIGURA COMPARACIÓN: {dataset.upper()}")

    df = load_comparison_global()
    df_dataset = df[df["dataset"].str.lower() == dataset.lower()].copy()

    if df_dataset.empty:
        raise ValueError(f"No hay datos para dataset={dataset} en comparison_global_oof_all_models.csv")

    x_labels = [METRIC_LABELS[m] for m in METRICS]
    x = np.arange(len(METRICS))

    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    for representation in ["FFT log-magnitude", "STFT log-power", "Welch PSD log-power"]:
        df_rep = df_dataset[df_dataset["representation"] == representation]

        if df_rep.empty:
            continue

        color = REP_COLORS.get(representation, None)

        for family, linestyle, marker, label_suffix in [
            ("Single-task", "--", "o", "Single-task"),
            ("Multitask", "-", "s", "Multitask"),
        ]:
            row = df_rep[df_rep["family"] == family]

            if row.empty:
                continue

            values = [float(row.iloc[0][m]) for m in METRICS]

            ax.plot(
                x,
                values,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.4,
                markersize=8,
                color=color,
                label=f"{representation.split()[0]} - {label_suffix}",
            )

    title_dataset = DATASET_INFO[dataset]["title"]
    ax.set_title(title_dataset, color="black")
    ax.set_xlabel("Metric", color="black")
    ax.set_ylabel("Metric value", color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, color="black")
    ax.grid(axis="y", alpha=0.25)

    if dataset == "jacket":
        ax.set_ylim(0.96, 1.01)
    else:
        ax.set_ylim(0.60, 0.95)

    ax.legend(
        frameon=True,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
    )

    plt.tight_layout()

    basename = f"fig_comparison_single_vs_multitask_{dataset}"
    save_figure(fig, basename)
    plt.close(fig)

    source_tables[f"comparison_{dataset}"] = df_dataset


def generate_delta_barplot(source_tables: dict) -> None:
    print_header("FIGURA: DELTAS MULTITASK - SINGLE-TASK")

    delta = load_delta_table().copy()

    metric_cols = [
        "delta_accuracy_Multitask_minus_Single",
        "delta_balanced_accuracy_Multitask_minus_Single",
        "delta_f1_macro_Multitask_minus_Single",
        "delta_f1_weighted_Multitask_minus_Single",
    ]

    labels = {
        "delta_accuracy_Multitask_minus_Single": "Accuracy",
        "delta_balanced_accuracy_Multitask_minus_Single": "Balanced\nAccuracy",
        "delta_f1_macro_Multitask_minus_Single": "Macro-F1",
        "delta_f1_weighted_Multitask_minus_Single": "Weighted-F1",
    }

    for dataset in ["jacket", "rotor"]:
        df_dataset = delta[delta["dataset"] == dataset].copy()

        fig, ax = plt.subplots(figsize=(12, 7.2))

        x = np.arange(len(metric_cols))
        width = 0.23

        for idx, representation in enumerate(["FFT log-magnitude", "STFT log-power", "Welch PSD log-power"]):
            row = df_dataset[df_dataset["representation"] == representation]
            if row.empty:
                continue

            values = [float(row.iloc[0][m]) for m in metric_cols]
            color = REP_COLORS.get(representation, None)

            ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                label=representation.split()[0],
                color=color,
                edgecolor="black",
                linewidth=0.8,
            )

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(f"Absolute improvement: Multitask - Single-task ({DATASET_INFO[dataset]['title']})", color="black")
        ax.set_xlabel("Metric", color="black")
        ax.set_ylabel("Absolute improvement", color="black")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in metric_cols], color="black")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=True, title="Representation")

        plt.tight_layout()

        basename = f"fig_delta_multitask_minus_single_{dataset}"
        save_figure(fig, basename)
        plt.close(fig)

    source_tables["delta_multitask_minus_single"] = delta


# =============================================================================
# 4. PCA DEL ESPACIO LATENTE MULTITASK
# =============================================================================

def load_latents_for_representation(rep_key: str, dataset: str):
    latents_dir = RESULTS_MULTITASK / rep_key / "latents"
    require_path(latents_dir, f"carpeta de latentes {rep_key}")

    Z_all = []
    y_all = []
    pred_all = []

    for fold_id in range(1, 6):
        path = latents_dir / f"multitask_{rep_key}_fold{fold_id}_latents_test.npz"
        require_path(path, f"latentes multitask {rep_key} fold {fold_id}")

        data = np.load(path)

        if dataset == "jacket":
            Z_all.append(data["Z_jacket"])
            y_all.append(data["y_jacket"])
            pred_all.append(data["pred_jacket"])
        elif dataset == "rotor":
            Z_all.append(data["Z_rotor"])
            y_all.append(data["y_rotor"])
            pred_all.append(data["pred_rotor"])
        else:
            raise ValueError("dataset debe ser 'jacket' o 'rotor'.")

    Z = np.vstack(Z_all).astype(np.float32)
    y = np.concatenate(y_all).astype(int)
    y_pred = np.concatenate(pred_all).astype(int)

    return Z, y, y_pred


def generate_pca_latent_plots(source_tables: dict, rep_keys=("stft",)) -> None:
    print_header("FIGURAS PCA DEL ESPACIO LATENTE MULTITASK")

    for rep_key in rep_keys:
        for dataset in ["jacket", "rotor"]:
            Z, y, y_pred = load_latents_for_representation(rep_key, dataset)

            pca = PCA(n_components=3, random_state=42)
            Zp = pca.fit_transform(Z)

            class_names = DATASET_INFO[dataset]["class_names"]

            df_pca = pd.DataFrame({
                "PC1": Zp[:, 0],
                "PC2": Zp[:, 1],
                "PC3": Zp[:, 2],
                "y_true": y,
                "y_pred": y_pred,
                "class_name": [class_names[i] for i in y],
                "predicted_class": [class_names[i] for i in y_pred],
                "dataset": dataset,
                "representation": REP_INFO[rep_key]["long"],
            })

            df_pca.to_csv(
                OUT_CSV / f"pca_latent_{dataset}_{rep_key}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            source_tables[f"pca_{dataset}_{rep_key}"] = df_pca

            # PCA 2D
            fig, ax = plt.subplots(figsize=(9.5, 7.2))

            for cls_idx, cls_name in enumerate(class_names):
                mask = y == cls_idx
                ax.scatter(
                    Zp[mask, 0],
                    Zp[mask, 1],
                    s=26,
                    alpha=0.78,
                    color=CLASS_COLORS[cls_idx % len(CLASS_COLORS)],
                    edgecolors="black",
                    linewidths=0.25,
                    label=cls_name,
                )

            evr = pca.explained_variance_ratio_ * 100

            ax.set_title(f"PCA 2D latent space - {DATASET_INFO[dataset]['title']} - {REP_INFO[rep_key]['label']}", color="black")
            ax.set_xlabel(f"PC1 ({evr[0]:.2f}%)", color="black")
            ax.set_ylabel(f"PC2 ({evr[1]:.2f}%)", color="black")
            ax.grid(alpha=0.25)
            ax.legend(frameon=True, loc="best")

            plt.tight_layout()

            basename = f"fig_pca2d_latent_{dataset}_{rep_key}"
            save_figure(fig, basename)
            plt.close(fig)

            # PCA 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            for cls_idx, cls_name in enumerate(class_names):
                mask = y == cls_idx
                ax.scatter(
                    Zp[mask, 0],
                    Zp[mask, 1],
                    Zp[mask, 2],
                    s=22,
                    alpha=0.78,
                    color=CLASS_COLORS[cls_idx % len(CLASS_COLORS)],
                    edgecolors="black",
                    linewidths=0.20,
                    label=cls_name,
                )

            ax.set_title(f"PCA 3D latent space - {DATASET_INFO[dataset]['title']} - {REP_INFO[rep_key]['label']}", color="black", pad=20)
            ax.set_xlabel(f"PC1 ({evr[0]:.2f}%)", color="black", labelpad=10)
            ax.set_ylabel(f"PC2 ({evr[1]:.2f}%)", color="black", labelpad=10)
            ax.set_zlabel(f"PC3 ({evr[2]:.2f}%)", color="black", labelpad=10)
            ax.legend(frameon=True, loc="best")

            plt.tight_layout()

            basename = f"fig_pca3d_latent_{dataset}_{rep_key}"
            save_figure(fig, basename)
            plt.close(fig)


# =============================================================================
# 5. TABLA DE RESULTADOS COMO IMAGEN OPCIONAL
# =============================================================================

def generate_results_table_images(source_tables: dict) -> None:
    print_header("FIGURAS: TABLAS DE RESULTADOS")

    df = load_comparison_global().copy()

    table_cols = [
        "family",
        "dataset",
        "representation",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
    ]

    df_table = df[table_cols].copy()

    for col in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        df_table[col] = df_table[col].map(lambda x: f"{float(x):.4f}")

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.axis("off")

    table = ax.table(
        cellText=df_table.values,
        colLabels=[
            "Model type",
            "Dataset",
            "Representation",
            "Accuracy",
            "Balanced\nAccuracy",
            "Macro-F1",
            "Weighted-F1",
        ],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#e8f1f2")
        else:
            cell.set_facecolor("white")

    ax.set_title("Global OOF performance summary", fontsize=16, color="black", pad=18)

    plt.tight_layout()
    save_figure(fig, "fig_table_global_oof_performance")
    plt.close(fig)

    source_tables["global_oof_table"] = df_table


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    print_header("GENERANDO FIGURAS DEL APARTADO DE RESULTADOS")

    require_path(REPRO_ROOT, "carpeta reproducible")
    require_path(RESULTS_COMPARISON / "comparison_global_oof_all_models.csv", "comparación global")
    require_path(RESULTS_COMPARISON / "comparison_single_vs_multitask_delta.csv", "deltas single vs multitask")
    require_path(RESULTS_MULTITASK, "resultados multitask")
    require_path(RESULTS_TABNET, "resultados TabNet")

    source_tables = {}

    generate_multitask_confusion_figures(source_tables)

    generate_single_vs_multitask_lineplot("jacket", source_tables)
    generate_single_vs_multitask_lineplot("rotor", source_tables)

    generate_delta_barplot(source_tables)

    # Por defecto genera PCA para STFT, como en el apartado visual del artículo.
    # Para generar PCA en todas las representaciones, cambia a:
    # generate_pca_latent_plots(source_tables, rep_keys=("fft", "stft", "welch"))
    generate_pca_latent_plots(source_tables, rep_keys=("stft",))

    generate_results_table_images(source_tables)

    save_source_tables_to_excel(source_tables)

    config = {
        "repro_root": str(REPRO_ROOT),
        "output_root": str(OUT_ROOT),
        "figures_png": str(OUT_PNG),
        "figures_pdf": str(OUT_PDF),
        "source_csv": str(OUT_CSV),
        "source_excel": str(OUT_XLSX),
        "generated_figures": [
            "fig_confusion_multitask_fft_jacket_rotor",
            "fig_confusion_multitask_stft_jacket_rotor",
            "fig_confusion_multitask_welch_jacket_rotor",
            "fig_comparison_single_vs_multitask_jacket",
            "fig_comparison_single_vs_multitask_rotor",
            "fig_delta_multitask_minus_single_jacket",
            "fig_delta_multitask_minus_single_rotor",
            "fig_pca2d_latent_jacket_stft",
            "fig_pca3d_latent_jacket_stft",
            "fig_pca2d_latent_rotor_stft",
            "fig_pca3d_latent_rotor_stft",
            "fig_table_global_oof_performance",
        ],
    }

    with open(OUT_ROOT / "config_figuras_resultados.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print_header("FIGURAS GENERADAS CORRECTAMENTE")
    print("Carpeta principal:")
    print(OUT_ROOT)
    print("\nPDF:")
    print(OUT_PDF)
    print("\nPNG:")
    print(OUT_PNG)
    print("\nDatos fuente:")
    print(OUT_XLSX / "datos_fuente_figuras_resultados.xlsx")


if __name__ == "__main__":
    main()
