# Pipeline de Predição de Crises Epilépticas a partir de EEG

Pipeline intra-paciente para predição de crises epilépticas usando sinais de EEG. Cada modelo é treinado e avaliado nos dados do mesmo paciente — sem transferência entre pacientes ou datasets.

---

## Datasets

| Dataset  | Pacientes | Canais              | Interictal                    |
| -------- | --------- | ------------------- | ----------------------------- |
| CHB-MIT  | 10        | 23 (bipolar)        | Gravações sem crise dedicadas |
| Siena    | 10        | ~28 (referencial)   | Intra-arquivo                 |
| Mendeley | 3         | 17-23 (referencial) | Intra-arquivo                 |
| SeizeIT2 | 10        | 2 (behind-the-ear)  | Gravações sem crise dedicadas |

**Total: 33 pacientes · ~217 contextos de crise válidos**

Critério de inclusão: ≥ 2 crises registradas e pelo menos 1 par de crises consecutivas com gap ≥ 1,5h.

---

## Estrutura dos Notebooks

```
NB1 → NB2 → NB3 → NB4
```

### NB1 — Aquisição e Pré-processamento

- Download dos EDFs de cada dataset
- Filtragem passa-banda 0,5–40 Hz (Butterworth ordem 4, zero-phase)
- Notch 60 Hz (CHB-MIT) / 50 Hz (demais)
- Reamostragem para 256 Hz
- Seleção de canais por nível (R5→R0)
- Saída: `data/signals/*_signal.npz`

### NB2 — Estimação da Janela Pré-ictal

- Para cada crise: segmenta 60min antes do onset em janelas de 30s
- Extrai as 19 features por canal (mesmas do NB3)
- PELT para detecção de pontos de mudança
- K-Means (k=2) + índice Silhouette para identificar estados eletrofisiológicos
- Estima PRE_SEC individual por paciente (mediana das estimativas válidas)
- Pacientes sem Silhouette válido (< 0,30) recebem a mediana global como fallback
- Define as 4 janelas fixas a partir da distribuição PRE_SEC dos 33 pacientes
- Saída: `data/level1_signals/*_L1_w{N}.npz` · `data/logs/presec_estimates.json`

### NB3 — Extração de Features

- 19 features por canal em janelas de 30s (passo 15s)
- 6 níveis de canal: R5 (17 canais) → R0 (2 canais)
- Otimização: calcula R5 uma vez e deriva R3/R2/R1/R0 por fatiamento de colunas
- SeizeIT2 processado separadamente (R0, canais behind-the-ear)
- Retomada automática — arquivos já gerados são pulados
- Saída: `data/features/{nível}/*__w{N}.npz`

### NB4 — Treinamento e Avaliação (LOSO)

Três etapas sequenciais, uma variável livre por etapa:

**Etapa 1 — Qual janela é melhor?**
`1 modelo × 5 janelas × LOSO` → janela ótima por paciente

**Etapa 2 — Qual nível de canal é melhor?**
`1 modelo × 6 níveis × LOSO` (janela ótima da Etapa 1) → curva de degradação R5→R0

**Etapa 3 — Qual modelo performa melhor?**
`3 modelos × LOSO` (janela ótima + nível ótimo) → melhor modelo geral

- Saída: `data/results/nb4_results.csv`

---

## Contexto de Crise Válido

Cada contexto = par (interictal limpo + pré-ictal de uma crise específica).

```
0h ──── 2h30 ──────── 3h15 ──── 4h ──── 4h20 ────
        │             │          │        │
        INTERICTAL    UNKNOWN    PRÉ-ICTAL ICTAL
        (classe 0)    (descarta)  N min    (descarta)
```

- **Interictal** termina sempre 1,5h antes da crise
- **Unknown** = buffer entre fim do interictal e início do pré-ictal (descartado)
- **Pré-ictal** = N minutos antes do onset (N definido após NB2)
- **Ictal e pós-ictal** = descartados

Regra: interictal disponível ≥ N min. Se menor, contexto descartado para aquela janela. Se maior, undersampling 1:1.

---

## Validação Cruzada

**Leave-One-Seizure-Out (LOSO)** por contexto de crise. Para N contextos válidos:

- N folds — cada fold usa 1 contexto como teste e N-1 como treino
- Treino balanceado 1:1 por undersampling do interictal
- Teste preserva distribuição natural

---

## Hierarquia de Canais

| Nível | Canais | Features | Analogia                          |
| ----- | ------ | -------- | --------------------------------- |
| R5    | 17     | 323      | EEG hospitalar completo           |
| R4    | 15     | 285      | Sem occipital                     |
| R3    | 13     | 247      | Sem occipital e parietal          |
| R2    | 11     | 209      | Só frontal e temporal             |
| R1    | 4      | 76       | Só temporal                       |
| R0    | 2      | 38       | Dispositivo vestível (= SeizeIT2) |

R0 ⊆ R1 ⊆ R2 ⊆ R3 ⊆ R4 ⊆ R5 — verificado por assert automático.

---

## Estrutura de Diretórios

```
data/
  signals/          # NB1 — sinais filtrados por gravação
  level1_signals/   # NB2 — rótulos por gravação e janela
  features/         # NB3 — features por gravação, nível e janela
    R5/ R4/ R3/ R2/ R1/ R0/
  results/          # NB4 — métricas por fold
  logs/             # manifestos e estimativas PRE_SEC
```

---

## Dependências

```
numpy pandas scipy PyWavelets ruptures scikit-learn xgboost tqdm joblib
```

---

## Objetivos Científicos

**Objetivo 1:** Estimar automaticamente a duração do período pré-ictal de cada paciente a partir dos próprios dados de EEG e avaliar se essa janela personalizada proporciona melhor desempenho de predição em comparação com janelas fixas representativas da população estudada.

**Objetivo 2:** Avaliar o impacto da redução progressiva de canais EEG sobre a predição intra-paciente — de um EEG hospitalar completo (17 canais, R5) até um dispositivo vestível de 2 eletrodos (R0, equivalente ao SeizeIT2).
