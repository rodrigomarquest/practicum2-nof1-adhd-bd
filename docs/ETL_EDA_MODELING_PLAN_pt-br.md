# 🧠 ETL → EDA → MODELING – Master Plan (Vader Mode)

> _“The Force of data flows from raw to AI.”_
> — Yoda, Data Jedi Mentor

---

## 1️⃣ ETL — Extract → Transform → Load

### 🎯 Objetivo

Converter os arquivos brutos (`data/raw`) em datasets limpos, normalizados e rotulados, prontos para análise e modelagem.

### 🗂️ Estrutura de diretórios

```
data/
├── raw/          ← exportações originais (Apple, Zepp)
├── etl/          ← estágios intermediários (snapshots, logs, QC)
└── ai/           ← dados finais prontos para modelagem
```

### ⚙️ Etapas principais

1. **Extração**

   ```bash
   python etl_pipeline.py extract --participant P000001 --snapshot 2025-09-29 \
     --cutover 2024-03-11 --tz_before America/Sao_Paulo --tz_after Europe/Dublin
   ```

   Gera diretórios `extracted/` dentro de `data/etl/.../snapshots/<snapshot>/`.

2. **Parsing e normalização por dispositivo**

   - `parse_zepp_export.py`, `extract_screen_time.py`
     Normaliza CSVs e unifica unidades, nomes e timestamps.

3. **Join e features diárias**

   ```bash
   python etl_pipeline.py join --participant P000001 --snapshot 2025-09-29
   python etl_pipeline.py features --participant P000001 --snapshot 2025-09-29
   ```

   Gera `features_daily.csv`, `features_per_minute.csv`, e `features_daily_labeled.csv`.

4. **Proveniência e auditoria**

   ```bash
   python provenance_audit.py --participant P000001 --snapshot 2025-09-29
   ```

   Saídas:

   ```
   data/etl/P000001/snapshots/2025-09-29/provenance/
     ├── etl_provenance_report.csv
     ├── data_audit_summary.md
     └── pip_freeze_*.txt
   ```

### ✅ Validações-chave

- Cobertura temporal correta (sem timestamps futuros).
- Colunas obrigatórias presentes e sem >30 % de missing.
- Sem duplicatas por `(timestamp, device, metric)`.
- HR e HRV em intervalos fisiológicos plausíveis.
- Checksums SHA256 dos arquivos brutos gravados em `provenance/`.

### ♻️ Idempotência

Cada execução deve usar `--snapshot`.
Reexecutar o mesmo snapshot **não deve corromper** dados; apenas substitui se o conteúdo mudar.

---

## 2️⃣ EDA — Análise Exploratória de Dados

### 🎯 Objetivo

Validar a consistência dos dados e garantir que o snapshot está pronto para modelagem.

### 🧩 Etapas

1. **Sanity check inicial**

   ```bash
   python make_03_eda_cardio.py --participant P000001 --snapshot 2025-09-29
   ```

2. **EDA completo**

   ```bash
   python make_03_eda_cardio_plus.py --participant P000001 --snapshot 2025-09-29
   ```

   Saídas:

   ```
   reports/eda/P000001/2025-09-29/
     ├── missingness.csv
     ├── feature_corr_matrix.png
     ├── coverage_by_date.png
     └── outlier_report.md
   ```

3. **Aprovação do snapshot**

   - Criar `reports/etl_acceptance/2025-09-29_approved.txt` se o QC for aprovado.

### 🔍 Validações principais

- Percentual de missing < 50 % por feature.
- Correlação alta (> 0.95) → marcar para remoção.
- Distribuição equilibrada de labels.
- Outliers tratados (z-score > 3 ou IQR).
- Verificação de estacionaridade nas séries temporais para LSTM.

---

## 3️⃣ MODELAGEM — Baselines e LSTM

### 🎯 Objetivo

Estimar desempenho base e treinar modelos temporais (LSTM) para previsão de estados mentais.

### ⚙️ Etapas

1. **Preparar dataset AI**

   ```bash
   cp data/etl/P000001/snapshots/2025-09-29/joined/features_daily_labeled.csv \
      data/ai/P000001/snapshots/2025-09-29/
   ```

2. **Baselines**

   ```bash
   python NB2_Baseline_and_LSTM.py --participant P000001 --snapshot 2025-09-29 \
     --outdir notebooks/outputs/NB2/$(date +%Y%m%d_%H%M%S)/
   ```

3. **LSTM (local smoke test)**

   ```bash
   python scripts/train_lstm.py --data data/ai/P000001/snapshots/2025-09-29/sequences/ \
     --epochs 3 --batch 64 --device cpu --out models/lstm/test_run/
   ```

4. **LSTM completo (Kaggle / GPU)**

   ```bash
   python scripts/train_lstm.py --data data/ai/P000001/snapshots/2025-09-29/sequences/ \
     --epochs 100 --batch 256 --device cuda --out models/lstm/exp_001/
   ```

### 📊 Artefatos esperados

```
notebooks/outputs/NB2/<timestamp>/tables/nb2_baseline_metrics.csv
notebooks/outputs/LSTM/exp_001/metrics.csv
reports/model_card/exp_001.md
models/lstm/exp_001/metadata.json
```

### 🧪 Validações

- Sem NaN nas features e labels.
- Nenhum vazamento de labels (train/test).
- Classes balanceadas ou ponderadas.
- Reprodutibilidade: seed fixa + commit log + pip_freeze.

---

## 4️⃣ Checklist Rápido

| Etapa        | Verificação                                             | Status |
| :----------- | :------------------------------------------------------ | :----- |
| ETL          | `features_daily_labeled.csv` gerado                     | ☑      |
| Proveniência | `etl_provenance_report.csv` presente                    | ☑      |
| EDA          | `feature_corr_matrix.png` e `outlier_report.md` criados | ☑      |
| Baselines    | `nb2_baseline_metrics.csv` salvo                        | ☑      |
| LSTM         | `metrics.csv` e `model_card.md` gerados                 | ☑      |

---

### 💬 Missão final

> _“O caminho do dado é claro: do bruto ao iluminado, do ruído à sabedoria.
> Confie no snapshot, respeite a proveniência — e a Força dos dados o guiará.”_

---
