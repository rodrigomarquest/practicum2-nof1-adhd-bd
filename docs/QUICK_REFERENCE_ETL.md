# Quick Reference: ETL Pipeline Completo (Fases 1-3)

**Última Atualização**: 6 de Novembro de 2025  
**Status**: ✅ Pronto para Produção

---

## Ordem de Execução Recomendada

### 1️⃣ Extrair dados brutos → features (Fase 1)

```bash
# Activity (Apple Raw → Activity Metrics)
make activity DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# Cardio (Zepp CSV → HR Metrics)
make cardio DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# Sleep (Zepp CSV → Sleep Metrics)
make sleep DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
```

**Output**: `data/etl/P000001/2025-11-06/features/<domain>/<vendor>/<variant>/features_daily.csv`

---

### 2️⃣ Enriquecer por domínio (Fase 2: Prejoin)

```bash
make enrich-prejoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
```

**Output**:

- `data/etl/P000001/2025-11-06/enriched/prejoin/<domain>/<vendor>/<variant>/enriched_<domain>.csv`
- **Columns added**: `_7d` (rolling avg 7d), `_zscore` (standardized)

**Exemplos**:

- zepp_steps → zepp_steps_7d, zepp_steps_zscore
- apple_hr_mean → apple_hr_mean_7d, apple_hr_mean_zscore

---

### 3️⃣ Juntar domínios → joined (Fase 3: Global Join + Coalescência)

```bash
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06
```

**Output**:

- `data/etl/P000001/2025-11-06/joined/joined_features_daily.csv`
- `data/etl/P000001/2025-11-06/qc/join_qc.csv` (QC automatizado)

**Columns added** (coalesced):

- `act_steps = coalesce(apple_steps, zepp_steps)`
- `act_active_min = coalesce(apple_exercise_min, zepp_exercise_min)`
- `hr_mean = coalesce(apple_hr_mean, zepp_hr_mean)`
- `hr_std = coalesce(apple_hr_std, zepp_hr_std)`
- `n_hr = coalesce(apple_n_hr, zepp_n_hr)`

**Mantém** (não deleta):

- Todas as colunas originais por vendor (apple*\*, zepp*\*) para debugging

---

### 4️⃣ Enriquecer cross-domain (Fase 3: Postjoin)

```bash
make enrich-postjoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
```

**Output**: `data/etl/P000001/2025-11-06/enriched/postjoin/<domain>/enriched_<domain>.csv`

**Columns added**:

**Activity**:

- `act_steps_vs_hr_7d_corr` — Correlação 7d entre atividade e HR

**Cardio**:

- `hr_mean_vs_act_7d_corr` — Correlação 7d entre HR e atividade
- `hr_variability_ratio` — Razão HR_std / HR_mean

**Sleep**:

- `sleep_activity_ratio` — Razão sleep_hours / exercise_minutes

---

## Pipeline Completo em Uma Linha

```bash
# Assumindo que features já estão em place
make enrich-prejoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128 && \
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 && \
make enrich-postjoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06
```

---

## Variáveis Principais

| Variável      | Exemplo    | Descrição                                             |
| ------------- | ---------- | ----------------------------------------------------- |
| `PID`         | P000001    | Participant ID                                        |
| `SNAPSHOT`    | 2025-11-06 | Data da snapshot (YYYY-MM-DD)                         |
| `DRY_RUN`     | 0 \| 1     | 0 = escrever, 1 = simular                             |
| `MAX_RECORDS` | 128        | Limitar registros (testing). **Não afeta join final** |

---

## Validação & QC

### Verificar cobertura por domínio

```bash
cat data/etl/P000001/2025-11-06/qc/join_qc.csv
```

**Exemplo output**:

```
n_rows,date_min,date_max,coverage_activity,coverage_cardio,coverage_sleep,used_prejoin_activity,used_prejoin_cardio
201,2018-04-06,2025-07-28,65.67,34.33,NaN,True,True
```

### Verificar colunas no joined

```bash
python -c "import pandas as pd; df = pd.read_csv('data/etl/P000001/2025-11-06/joined/joined_features_daily.csv'); print(f'Rows: {len(df)}, Cols: {len(df.columns)}'); print('Coalesced:', [c for c in df.columns if c in ['act_steps', 'hr_mean', 'n_hr']])"
```

### Verificar enriquecimentos postjoin

```bash
python -c "import pandas as pd; df = pd.read_csv('data/etl/P000001/2025-11-06/enriched/postjoin/activity/enriched_activity.csv'); print([c for c in df.columns if '_corr' in c or '_ratio' in c])"
```

---

## Tipos de Dados (Normalizados)

| Coluna       | Tipo       | Exemplo    | Notas                                               |
| ------------ | ---------- | ---------- | --------------------------------------------------- |
| `date`       | YYYY-MM-DD | 2025-11-06 | String na CSV (datetime64 internamente até escrita) |
| `*_steps`    | float64    | 6923.0     | Apple + Zepp                                        |
| `*_hr_*`     | float64    | 73.6       | Apple + Zepp                                        |
| `*_*_7d`     | float64    | 7432.5     | Rolling avg (7d, min_periods=1)                     |
| `*_*_zscore` | float64    | -0.234     | Standardized (mean=0, std=1)                        |
| `*_corr`     | float64    | 0.567      | Pearson correlation [-1, 1]                         |
| `*_ratio`    | float64    | 1.234      | Ratio (numerator / denominator)                     |

---

## Invariantes (Respeitar Sempre)

✅ **MAX_RECORDS**: Afeta **apenas** seeds + prejoin. Join **nunca** trunca.

✅ **datetime64**: Mantém como datetime64 internamente até escrita CSV.

✅ **Vendor preservation**: Todas as colunas originais (apple*\*, zepp*\*) são mantidas.

✅ **Cascata**: Join prioriza `enriched/prejoin` → `features` → `legacy`.

✅ **Modularização**: Cada stage é módulo Python independente (`python -m domains...`).

---

## EDA Dataset

Arquivo de referência para análise exploratória:

```
data/etl/<PID>/<SNAPSHOT>/joined/joined_features_daily.csv
```

**Contém**:

- Data + Domínios (activity, cardio, sleep)
- Colunas brutos (vendor-specific)
- Colunas \_7d (rolling averages)
- Colunas \_zscore (standardized)
- Colunas coalesced (hr_mean, act_steps, etc.)
- Colunas \_corr/\_ratio (cross-domain enrichments quando postjoin executado)

---

## Troubleshooting

### "joined CSV not found"

```bash
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06
# Garante que enriched/prejoin ou features/ existem
```

### "MAX_RECORDS cutting off join final"

```bash
# ❌ ERRADO - MAX_RECORDS não afeta join
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# ✅ CORRETO - use MAX_RECORDS em prejoin, depois join normalmente
make enrich-prejoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06  # sem MAX_RECORDS
```

### "datetime mixed-type errors"

Já resolvido! O pipeline mantém datetime64 internamente até escrita.

---

## Próximas Etapas

1. **Fase 4: QC Comparativo** (TBD)

   - Comparar postjoin vs legacy
   - Detectar drift/anomalias

2. **Fase 5: Labels & Aggregation** (opcional)
   - Labeling baseado em critérios clínicos
   - Agregação por período (semanal, mensal)

---

## Documentação Completa

- 📖 `docs/PHASE3_ENRICHED_GLOBAL_ARCHITECTURE.md` (Fase 3 detalhada)
- 📖 `docs/PHASE2_ENRICHED_PREJOIN_ARCHITECTURE.md` (Fase 2 detalhada)
- 📖 `docs/ETL_ARCHITECTURE_COMPLETE.md` (Visão geral Fases 1-3)
- 📖 `docs/FASE3_STATUS.txt` (Status executivo)

---

**Status**: ✅ Pronto para Produção  
**Versão**: ETL v4.1.0  
**Data**: 6 de Novembro de 2025
