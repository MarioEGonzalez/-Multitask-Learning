# -Multitask-Learning
REPRODUCIBILIDAD DEL ARTÍCULO MULTITASK

Artículo:
Improving Vibration-Based Classification in a Wind Turbine Jacket Structure and Rotor
through Multitask Learning and Spectral Representations


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
- Seed = 42
- STFT: window=hann, nperseg=256, noverlap=128, nfft=256
- Welch: window=hann, nperseg=512, noverlap=256, nfft=512
- Top-K = 4096

Orden de ejecución:
1) 01_crear_datasets_representaciones_cv5.
2) 02_entrenar_modelos_y_generar_resultados
3) 03_generar_graficas_resultados_articulo
