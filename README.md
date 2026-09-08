# Multitask Learning for Vibration-Based Wind Turbine Classification

This repository contains the source code and reproducibility materials associated with the manuscript:

**Improving Vibration-Based Classification in a Wind Turbine Jacket Structure and Rotor through Multitask Learning and Spectral Representations**

The study evaluates supervised single-task and multitask learning approaches for two vibration-based classification problems related to wind turbines:

1. Structural damage classification in a laboratory-scale jacket-type structure.
2. Rotor imbalance classification in a laboratory-adapted Enair E30Pro wind turbine.

Three signal representations are evaluated:

- FFT log-magnitude
- STFT log-power
- Welch PSD log-power

The revised study also includes heterogeneous machine-learning baselines, Top-K sensitivity analysis, auxiliary-label randomization experiments, and a capacity-matched single-task neural control.

---

## Reproducibility Settings

Unless otherwise indicated, the principal experiments reported in the revised manuscript use the following settings:

- Random seed: `42`
- Cross-validation splitter: `StratifiedGroupKFold`
- Number of folds: `5`
- Shuffle: `True`
- Cross-validation random state: `42`
- Top-K selected features: `4096`
- Multitask training epochs: `120`
- Optimizer: `Adam`
- Initial learning rate: `1e-3`
- Weight decay: `1e-5`
- Learning-rate scheduler: `StepLR`
- Scheduler step size: `40` epochs
- Scheduler decay factor: `0.7`
- Jacket batch size: `256`
- Rotor batch size: `128`
- Loss: weighted cross-entropy

For the rotor dataset, the original experiment identifier is used as the group variable. All windows originating from the same experiment therefore remain in the same cross-validation fold.

For the jacket dataset, true experiment-level identifiers were unavailable in the source dataset. Each sample was therefore assigned a unique group identifier.

The same cross-validation partitions are reused across the corresponding representations, models, and robustness controls.

---

## Repository Structure and Execution Order

The principal scripts should be used in the following order.

### 1. Dataset preparation and spectral representations

`01 crea data set y representaciones cv5 .py`

Main functions:

- Dataset preparation
- FFT log-magnitude representation
- STFT log-power representation
- Welch PSD log-power representation
- Five-fold cross-validation preparation
- Fold-specific Top-K feature selection
- Fold-specific standardization

Feature selection and standardization are fitted exclusively using the training subset of each fold.

---

### 2. Main single-task and multitask experiments

`02 entrenar modelos y generar resultados.py`

Main functions:

- TabNet single-task baseline training
- Multitask neural network training
- Out-of-fold predictions
- Main classification metrics
- Accuracy
- Balanced accuracy
- Macro-F1
- Weighted-F1
- Confusion-matrix data

The multitask architecture uses the shared encoder:

`4096 -> 1024 -> 256 -> 64`

followed by two task-specific classification heads:

`64 -> 32 -> 5`

---

### 3. Manuscript figures

`03 generar graficas resultados articulo.py`

Main functions:

- Confusion matrices
- Normalized confusion matrices
- Single-task versus multitask comparison plots
- Principal manuscript result figures
- Latent-space visualization outputs where applicable

---

### 4. Additional machine-learning and neural baselines

`04 baselines adicionales revision.py`

This script evaluates the additional heterogeneous baselines introduced during manuscript revision:

- Random Forest
- XGBoost
- One-dimensional CNN
- Bidirectional RNN

These models provide broader predictive benchmarks but are not used as capacity-matched controls of multitask parameter sharing.

---

### 5. Top-K sensitivity analysis

`05 topk sensitivity revision.py`

This script evaluates sensitivity to the number and type of selected features.

Evaluated Top-K values include:

- 256
- 512
- 1024
- 2048
- 4096
- 8192
- Full feature set

The analysis compares:

- Variance-based Top-K feature selection
- Supervised ANOVA F-test feature selection

The sensitivity analysis uses a Random Forest classifier for computational tractability.

---

### 6. Robustness checks and auxiliary-label randomization

`06 robustness checks tabnet weighted and negative control.py`

Main functions:

- Class-weighted TabNet robustness experiment
- Auxiliary-label randomization experiment
- Negative-control analysis
- Target-task performance with real auxiliary labels
- Target-task performance with randomized auxiliary labels

The relevant negative-control comparison keeps the target-task labels unchanged and randomizes only the auxiliary-task training labels.

The results are interpreted as evidence that auxiliary supervision can influence target-task optimization within the shared network. They are not interpreted by themselves as proof of superior positive cross-task transfer.

---

### 7. Capacity-matched single-task control

`07_capacity_matched_single_task_control.py`

This script implements the capacity-matched single-task neural control introduced in response to the reviewer.

The single-task control uses:

- The same encoder architecture as the multitask model
- The same preprocessing
- The same Top-K features
- The same fold assignments
- The same weighted cross-entropy treatment
- The same Adam optimizer
- The same learning-rate scheduler
- The same 120-epoch training budget
- The same effective number of parameter-update steps per epoch

Single-task architecture:

`4096 -> 1024 -> 256 -> 64 -> 32 -> 5`

Multitask architecture:

Shared encoder:

`4096 -> 1024 -> 256 -> 64`

Two task-specific heads:

`64 -> 32 -> 5`

---

## Capacity-Matched Parameter Comparison

The capacity-matched single-task network contains:

**4,479,109 trainable parameters**

The multitask network contains:

**4,481,354 trainable parameters**

Difference:

**2,245 parameters**

which corresponds to approximately:

**0.0501%**

Two independent capacity-matched single-task networks require:

**8,958,218 trainable parameters**

One multitask model requires:

**4,481,354 trainable parameters**

Therefore, the shared multitask architecture reduces the total parameter count by approximately:

**49.97%**

relative to maintaining two independent capacity-matched single-task networks.

The capacity-matched experiments show that multitask learning does not provide a consistent predictive advantage over an equivalently structured single-task model. The principal demonstrated advantage of the shared architecture is therefore model consolidation and parameter-efficient joint modeling.

---

## Main Script-to-Result Mapping

| Manuscript analysis/result | Main reproducibility script |
|---|---|
| Dataset preparation | `01 crea data set y representaciones cv5 .py` |
| FFT representation | `01 crea data set y representaciones cv5 .py` |
| STFT representation | `01 crea data set y representaciones cv5 .py` |
| Welch PSD representation | `01 crea data set y representaciones cv5 .py` |
| Five-fold partitions | `01 crea data set y representaciones cv5 .py` |
| Top-K feature preparation | `01 crea data set y representaciones cv5 .py` |
| TabNet baselines | `02 entrenar modelos y generar resultados.py` |
| Multitask model | `02 entrenar modelos y generar resultados.py` |
| Main confusion matrices and figures | `03 generar graficas resultados articulo.py` |
| Random Forest / XGBoost / CNN / RNN | `04 baselines adicionales revision.py` |
| Top-K sensitivity | `05 topk sensitivity revision.py` |
| Weighted TabNet robustness check | `06 robustness checks tabnet weighted and negative control.py` |
| Auxiliary-label negative control | `06 robustness checks tabnet weighted and negative control.py` |
| Capacity-matched single-task control | `07_capacity_matched_single_task_control.py` |

---

## Statistical Reporting

The five cross-validation folds contain overlapping training subsets and are therefore not treated as independent observations for conventional paired inferential tests.

Accordingly, the revised manuscript does not use ordinary paired t-tests or Wilcoxon signed-rank tests over the five folds to claim statistical significance.

Fold-level comparisons are reported descriptively using:

- Mean
- Standard deviation
- Paired mean differences
- Better / tie / worse fold counts where applicable

---

## Spectral Representation Interpretation

The three representation pipelines do not use fully harmonized logarithmic transformations.

FFT uses:

`log10(1 + |X| + epsilon)`

whereas STFT and Welch PSD use:

`log10(power + epsilon)`

with:

`epsilon = 1e-12`

Therefore, differences among FFT, STFT, and Welch PSD should be interpreted as differences among the complete implemented representation-preprocessing pipelines and not as evidence of the intrinsic superiority of one spectral estimator.

---

## Dependencies

The Python dependencies required to reproduce the experiments are provided in:

`requirements.txt`

Users are encouraged to create an isolated Python environment before installing the dependencies.

Example:

```bash
python -m venv .venv
