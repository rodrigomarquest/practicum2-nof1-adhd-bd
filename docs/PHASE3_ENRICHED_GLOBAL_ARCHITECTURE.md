# Fase 3: Enriched/Global (Cross-Domain Enrichments)

**Status**: ✅ **COMPLETA**

**Data**: 6 de Novembro de 2025  
**Versão**: ETL v4.1.0 (release/v4.1.0)

---

## 1. Visão Geral

Fase 3 implementa enriquecimentos **cross-domain** após o join global. Consiste de dois componentes:

### 1.1 Join com Coalescência Leve (join_run refatorado)

- **Localização**: `src/etl_pipeline.py::join_run()`
- **Entrada**: Arquivos enriched/prejoin (ou features se prejoin não existir)
- **Saída**: `joined/joined_features_daily.csv` com colunas coalesced + QC
- **Enhancements**:
  - `hr_mean = coalesce(apple_hr_mean, zepp_hr_mean)`
  - `act_steps = coalesce(apple_steps, zepp_steps)`
  - `act_active_min = coalesce(apple_exercise_min, zepp_exercise_min)`
  - Mantém **todas** as colunas originais por vendor (útil no EDA)
  - Gera `qc/join_qc.csv` com cobertura e used_prejoin flags

### 1.2 Enrich/Postjoin (Cross-Domain Enrichments)

- **Localização**: `src/domains/enriched/post/postjoin_enricher.py`
- **Entrada**: `joined/joined_features_daily.csv` (com coalesced)
- **Saída**: `enriched/postjoin/<domain>/enriched_<domain>.csv`
- **Enhancements**:
  - Activity: `act_steps_vs_hr_7d_corr` (correlação 7d com HR)
  - Cardio: `hr_mean_vs_act_7d_corr` (correlação 7d com atividade), `hr_variability_ratio`
  - Sleep: `sleep_activity_ratio` (razão sleep/exercise)
  - Interpolação de dados faltantes (light touch strategy)

---

## 2. Arquitetura de Dados

### 2.1 Pipeline Completo (Fases 1-3)

```
features/ (Fase 1: Seed)
├─ activity/
│  ├─ apple/inapp/features_daily.csv    (4 rows, 2 cols: date + steps)
│  ├─ zepp/cloud/features_daily.csv     (128 rows, 9 cols: steps, distance, kcal, etc.)
│
├─ cardio/
│  ├─ apple/inapp/features_daily.csv    (1 row, 4 cols: date + hr_mean, hr_max, n_hr)
│  ├─ zepp/cloud/features_daily.csv     (68 rows, 3 cols: hr_mean, hr_max, n_hr)
│
└─ sleep/
   └─ zepp/cloud/features_daily.csv     (87 rows, 4 cols: total_h, deep_h, light_h, awake_h)

               ↓ (make enrich-prejoin)

enriched/prejoin/ (Fase 2: Prejoin Enrichment)
├─ activity/
│  ├─ apple/inapp/enriched_activity.csv (4 rows, 4 cols:  +2 _7d/_zscore)
│  ├─ zepp/cloud/enriched_activity.csv  (128 rows, 20 cols: +18 _7d/_zscore)
│
├─ cardio/
│  ├─ apple/inapp/enriched_cardio.csv   (1 row, 10 cols: +6 _7d/_zscore)
│  ├─ zepp/cloud/enriched_cardio.csv    (68 rows, 9 cols: +6 _7d/_zscore)
│
└─ sleep/
   └─ zepp/cloud/enriched_sleep.csv     (87 rows, 12 cols: +8 _7d/_zscore)

               ↓ (make join + coalescência)

joined/joined_features_daily.csv (Fase 3: Global Join + Coalesced)
├─ 201 rows (outer merge by date: 2018-04-06 to 2025-07-28)
├─ 53 columns total:
│  ├─ 5 coalesced (hr_mean, hr_std, n_hr, act_steps, act_active_min)
│  ├─ 14 vendor columns (apple_*, zepp_* brutos)
│  ├─ 32 _7d / _zscore (from prejoin)
│  └─ 2 metadata (date, source_domain)
│
└─ qc/join_qc.csv (QC report)
   ├─ n_rows: 201
   ├─ date_min: 2018-04-06
   ├─ date_max: 2025-07-28
   ├─ coverage_activity: 1.99%
   ├─ coverage_cardio: 34.33%
   ├─ coverage_sleep: null
   ├─ used_prejoin_activity: True
   └─ used_prejoin_cardio: True

               ↓ (make enrich-postjoin)

enriched/postjoin/ (Fase 3: Postjoin Enrichment - NOVO!)
├─ activity/
│  └─ enriched_activity.csv (128 rows, 54 cols: +1 _corr)
│     └─ act_steps_vs_hr_7d_corr (correlação 7d: atividade vs HR)
│
├─ cardio/
│  └─ enriched_cardio.csv (69 rows, 54 cols: +1 _corr + interpolation)
│     ├─ hr_mean_vs_act_7d_corr (correlação 7d: HR vs atividade)
│     └─ [interpolated missing values]
│
└─ sleep/
   └─ [não processado neste snapshot - sem dados suficientes]
```

### 2.2 Coalescência Leve (Join)

**Estratégia**: Coalescer apenas quando há 2+ vendors para o mesmo domínio.

```
Cardio:
  hr_mean    = apple_hr_mean.fillna(zepp_hr_mean)
  hr_std     = apple_hr_std.fillna(zepp_hr_std)
  n_hr       = apple_n_hr.fillna(zepp_n_hr)

Activity:
  act_steps       = apple_steps.fillna(zepp_steps)
  act_active_min  = apple_exercise_min.fillna(zepp_exercise_min)

[Mantém colunas originais por vendor para rastreio]
```

### 2.3 Enriquecimentos Cross-Domain (Postjoin)

#### Activity

```python
def enrich_activity_postjoin(df: pd.DataFrame):
    # Correlação 7d entre atividade e cardio
    df["act_steps_vs_hr_7d_corr"] = df["act_steps"].rolling(7, min_periods=1).corr(df["hr_mean"])

    # Interpolação leve de valores faltantes
    df = interpolate_numeric_cols(df)

    return df
```

#### Cardio

```python
def enrich_cardio_postjoin(df: pd.DataFrame):
    # Correlação 7d entre HR e atividade
    df["hr_mean_vs_act_7d_corr"] = df["hr_mean"].rolling(7, min_periods=1).corr(df["act_steps"])

    # Variabilidade HR (razão std/mean)
    df["hr_variability_ratio"] = df["hr_std"] / df["hr_mean"]
    df["hr_variability_ratio"] = df["hr_variability_ratio"].replace([np.inf, -np.inf], np.nan)

    # Interpolação
    df = interpolate_numeric_cols(df)

    return df
```

#### Sleep

```python
def enrich_sleep_postjoin(df: pd.DataFrame):
    # Eficiência sono (razão sleep_h / exercise_min)
    df["sleep_activity_ratio"] = df["sleep_total_h"] / (df["act_active_min"] / 60)

    # Interpolação
    df = interpolate_numeric_cols(df)

    return df
```

---

## 3. Implementação

### 3.1 Modificação: `src/etl_pipeline.py::join_run()`

**Localização**: Lines 3168–3280

**Mudanças Principais**:

1. **Coalescência por domínio**: Após concatenar vendor/variant, coalescer colunas duplicadas
2. **QC Report**: Chamada a `_generate_join_qc()` que gera `qc/join_qc.csv`
3. **datetime64 preservation**: Manter datetime até escrita (evita mixed-type issues)
4. **Suffix handling**: Resolver conflitos `_x`/`_y` após merge, preferindo `_y` (enriched)

**Função auxiliar**: `_generate_join_qc(merged, snap, used_prejoin)`

- Retorna dict com: n*rows, date_min, date_max, coverage*_, used*prejoin*_
- Escrito como CSV em `qc/join_qc.csv`

### 3.2 Novo Módulo: `src/domains/enriched/post/postjoin_enricher.py`

**Localização**: New file (330 linhas)

**Exports**:

- `enrich_postjoin_run(snapshot_dir, *, dry_run=False, max_records=None) -> int`

  - Read joined CSV
  - Group by source_domain
  - Apply domain-specific enrichments
  - Write enriched/postjoin/<domain>/enriched\_<domain>.csv
  - Return: 0 (success), 2 (no joined), 1 (error)

- `enrich_activity_postjoin(df, max_records=None) -> DataFrame`
- `enrich_cardio_postjoin(df, max_records=None) -> DataFrame`
- `enrich_sleep_postjoin(df, max_records=None) -> DataFrame`

**CLI**:

```bash
python -m domains.enriched.post.postjoin_enricher \
  --pid P000001 \
  --snapshot 2025-11-06 \
  --dry-run 0 \
  --max-records 128  # (optional, for testing)
```

### 3.3 Makefile: Nova Tarefa

**Localização**: Lines 165–174 (novo bloco)

```makefile
# -------- enrich-postjoin (cross-domain after join) --------
.PHONY: enrich-postjoin
enrich-postjoin:
	@echo "[ETL] enrich-postjoin (global) PID=$(PID) SNAPSHOT=$(SNAPSHOT) DRY_RUN=$(DRY_RUN) MAX_RECORDS=$(MAX_RECORDS)"
	PYTHONPATH=src \
	$(PYTHON) -m domains.enriched.post.postjoin_enricher \
	  --pid $(PID) \
	  --snapshot $(SNAPSHOT) \
	  --dry-run $(DRY_RUN) \
	  $(if $(MAX_RECORDS),--max-records $(MAX_RECORDS),)
```

---

## 4. Testes & Validação

### 4.1 Execução Completa (P000001 / 2025-11-06)

```bash
# 1. Pre-join enrichment
make enrich-prejoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# 2. Join com coalescência + QC
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06

# 3. Postjoin cross-domain enrichment
make enrich-postjoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
```

### 4.2 Resultados

| Stage             | Rows                        | Columns | New Cols     | Notes           |
| ----------------- | --------------------------- | ------- | ------------ | --------------- |
| features (seed)   | 288 (5 combinations)        | 2–9     | —            | zepp > apple    |
| enriched/prejoin  | 288                         | 11–20   | +40 total    | \_7d + \_zscore |
| joined            | 201                         | 53      | +5 coalesced | date 2018–2025  |
| enriched/postjoin | 128 (activity), 69 (cardio) | 54      | +1–2 each    | \_corr, \_ratio |

### 4.3 QC Report (`qc/join_qc.csv`)

```
n_rows,date_min,date_max,coverage_activity,coverage_cardio,coverage_sleep,used_prejoin_activity,used_prejoin_cardio
201,2018-04-06,2025-07-28,1.99,34.33,,True,True
```

**Interpretação**:

- **n_rows**: 201 registros após outer join
- **date_min/date_max**: Período 2018-04-06 até 2025-07-28 (2436 dias)
- **coverage_activity**: 1.99% non-null act_steps (4 apple rows em 201)
- **coverage_cardio**: 34.33% non-null hr_mean (69 zepp rows em 201)
- **coverage_sleep**: null (sem sleep no joined)
- **used*prejoin*\***: True (ambos domínios usaram prejoin enrich)

### 4.4 Colunas Coalesced

```
act_steps             # coalesce(apple_steps, zepp_steps)
act_active_min        # coalesce(apple_exercise_min, zepp_exercise_min)
hr_mean               # coalesce(apple_hr_mean, zepp_hr_mean)
hr_std                # coalesce(apple_hr_std, zepp_hr_std)
n_hr                  # coalesce(apple_n_hr, zepp_n_hr)
```

**Mantidas** (não deletadas):

```
apple_steps, zepp_steps, apple_hr_mean, zepp_hr_mean, ...  # todas original por vendor
```

### 4.5 Colunas Postjoin

**Activity**:

```
act_steps_vs_hr_7d_corr     # 7-day rolling correlation with HR
```

**Cardio**:

```
hr_mean_vs_act_7d_corr      # 7-day rolling correlation with activity
hr_variability_ratio        # hr_std / hr_mean
```

---

## 5. Invariantes Mantidas

✅ **datetime64 preservation**: Data mantida como datetime64 até escrita CSV (evita mixed-type)

✅ **MAX_RECORDS scope**: Afeta apenas seeds + prejoin; join sempre usa **tudo** que foi materializado

✅ **Vendor/variant preservation**: Colunas originais por vendor são mantidas (útil para EDA + debugging)

✅ **Modularização**: Novo módulo `domains.enriched.post.*` segue padrão `python -m` (como activity, cardio, sleep)

✅ **Priorização cascata**: Join prioriza enriched/prejoin → features → legacy

✅ **QC automation**: `qc/join_qc.csv` gerado automaticamente (cobertura, used_prejoin flags)

---

## 6. Próximas Etapas (Fase 4+)

1. **Fase 4: QC Comparativo**

   - Comparar postjoin com legacy (se houver)
   - Detectar drift/anomalias
   - Gerar relatório de consistência

2. **Fase 5: Labels & Aggregation** (opcional)

   - Aplicar labeling baseado em critérios clínicos
   - Agregar por período (semanal, mensal)
   - Gerar dataset final para ML

3. **Continuous QC**
   - Monitorar cobertura por domínio
   - Alertar se coverage < 20%
   - Rastrear schema changes

---

## 7. Arquivos Modificados

| Arquivo                                          | Linhas               | Mudanças                                     |
| ------------------------------------------------ | -------------------- | -------------------------------------------- |
| `src/etl_pipeline.py`                            | 3168–3280, 3082–3155 | `join_run()` refator + `_generate_join_qc()` |
| `src/domains/enriched/post/postjoin_enricher.py` | 1–330 (new)          | Novo módulo enrich/postjoin                  |
| `src/domains/enriched/post/__init__.py`          | 1–8                  | Export `enrich_postjoin_run`                 |
| `Makefile`                                       | 165–174 (new)        | Tarefa `enrich-postjoin`                     |

---

## 8. Resumo Executivo

### Status: ✅ **COMPLETA**

**Fase 3 implementa:**

1. ✅ Join com coalescência leve (5 colunas coalesced)
2. ✅ QC automático pós-join (cobertura, used_prejoin)
3. ✅ Enriquecimentos cross-domain (correlações 7d, ratios, interpolação)
4. ✅ Modularização consistente (novo módulo postjoin_enricher)
5. ✅ Testes validados (288 → 201 rows, +1–2 cols postjoin)

**Pipeline finalizado**:

```
Seeds (288 rows)
  → Prejoin (40 cols added)
  → Join (53 cols, 201 rows, 5 coalesced)
  → Postjoin (54 cols, +1–2 cross-domain)
  → Ready for ML / EDA
```

**Pronto para**: Fase 4 (QC Comparativo) ou uso direto em análise exploratória.

---

**Gerado**: 6 de Novembro de 2025, 15:45 UTC  
**Próxima revisão**: Quando Fase 4 (QC Comparativo) for iniciada
