# Predição de Crises Epilépticas via EEG

Pipeline completo de predição ictal com estimação individual do horizonte pré-ictal via **PELT + K-Means**, validado em 4 datasets públicos com LOSO por contexto de crise.

## Hipótese central

O algoritmo PELT detecta mudanças estruturais nas features de EEG e o K-Means separa os dois estados (inter/pré-ictal) sem supervisão? O horizonte pré-ictal estimado individualmente (H5) é comparado com horizontes fixos (H1–H4) no classificador.

## Datasets

| Dataset           | Fonte                      | Pacientes selecionados | Canais        |
| ----------------- | -------------------------- | ---------------------- | ------------- |
| CHB-MIT           | PhysioNet HTTP             | 7                      | 17 bipolares  |
| Siena Scalp EEG   | PhysioNet HTTP             | 7                      | 17 unipolares |
| SeizeIT2          | OpenNeuro S3 anônimo       | 7                      | 2 (R0 fixo)   |
| Mendeley Epilepsy | API pública (cloudscraper) | 6                      | 17 unipolares |

## Ordem de execução

```
NB1 → NB2 ─┐
NB1 → NB3 ──┴→ NB4 → NB4.2 (opcional)
                 └──→ NB5 → NB6 (interativo)
```

## Instalação

```bash
pip install numpy scipy pandas matplotlib scikit-learn xgboost \
            ruptures PyWavelets tqdm boto3 cloudscraper mne \
            openpyxl joblib ipywidgets
```

Python 3.10 ou 3.11 recomendado.

## Notebooks

| Notebook               | Função                                                 |
| ---------------------- | ------------------------------------------------------ |
| `NB1_final.ipynb`      | Aquisição, validação de contextos, pré-processamento   |
| `NB2_preictal.ipynb`   | Estimação do horizonte pré-ictal (PELT + K-Means)      |
| `NB3_features.ipynb`   | Extração de 19 features/canal em janelas de 30s        |
| `NB4_loso_v2.ipynb`    | LOSO em 3 etapas + coleta de predições e importâncias  |
| `NB4_2_pretrain.ipynb` | Experimento: pré-treino com contextos descartados      |
| `NB5_resultados.ipynb` | Análise completa de resultados (20 perguntas)          |
| `NB6_interativo.ipynb` | Análise interativa de métricas por evento (ipywidgets) |

## Arquivos gerados (`data/results/`)

`Result_stage1_windows.csv` · `Result_stage2_levels.csv` · `Result_stage3_models.csv` · `Result_predictions_per_window.csv` · `Result_feature_importances_rf.csv` · `Result_nb4_summary.json`

## Parâmetros principais

- `GUARD = 30 min` — zona proibida pós-crise
- `MAX_PRE = 30 min` — horizonte pré-ictal máximo
- `WIN_SEG = 30s / STEP = 15s` — segmentação de features
- `N_SEEDS = 5` — seeds de undersample
- `N_CONSEC = 5` — janelas consecutivas para definir evento (NB5/NB6)
