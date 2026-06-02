# Multitask-Learning

REPRODUCIBILITY PACKAGE FOR THE ARTICLE

**Improving Vibration-Based Classification in a Wind Turbine Jacket Structure and Rotor through Multitask Learning and Spectral Representations**

## Script 1

**01_create_datasets_representations_cv5.py**

### Purpose of this script

This script generates:

### 1. Multiclass Jacket Dataset

* 5740 samples
* 58008 features
* 5 classes: Healthy + Crack_Level_1 to Crack_Level_4
* Groups: one sample per group

### 2. Multiclass Rotor Dataset

* 1120 windows
* 35 experiments
* 32 windows per experiment
* 5 classes: Healthy + Imbalance_Level_1 to Imbalance_Level_4
* Groups: experiment identifier

### 3. Signal Representations

* FFT log-magnitude
* STFT log-power
* Welch PSD log-power

### 4. Cross-Validation Pipeline

* StratifiedGroupKFold
* 5 folds
* Variance-based Top-K feature selection (K = 4096)
* StandardScaler fitted exclusively on the training subset of each fold

### Parameters

* Random seed = 42
* STFT:

  * Window = Hann
  * nperseg = 256
  * noverlap = 128
  * nfft = 256
* Welch PSD:

  * Window = Hann
  * nperseg = 512
  * noverlap = 256
  * nfft = 512
* Top-K = 4096

### Execution Order

Run this script first.

1) 01_crear_datasets_representaciones_cv5.
2) 02_entrenar_modelos_y_generar_resultados
3) 03_generar_graficas_resultados_articulo
