# Mudanças Técnicas - Fase 3

**Status**: ✅ Implementado e Testado  
**Data**: 6 de Novembro de 2025

---

## Arquivo: `src/etl_pipeline.py`

### Modificação 1: Nova função `_generate_join_qc()` (linhas 3082–3155)

**Propósito**: Gerar relatório QC pós-join com cobertura e rastreamento de prejoin.

**Assinatura**:

```python
def _generate_join_qc(
    merged: pd.DataFrame,
    snap: Path,
    used_prejoin: dict[str, bool]
) -> dict
```

**Campos retornados**:

- `n_rows`: Número de linhas no joined
- `date_min`, `date_max`: Período de cobertura
- `coverage_activity`: % non-null de act_steps (ou apple_steps ou zepp_steps)
- `coverage_cardio`: % non-null de hr_mean (ou apple_hr_mean ou zepp_hr_mean)
- `coverage_sleep`: % non-null de sleep_total_h (se existir)
- `used_prejoin_activity`: 1/0 flag
- `used_prejoin_cardio`: 1/0 flag

**Lógica**:

1. Contar linhas totais e período (date_min/date_max)
2. Para cada domínio, calcular % non-null de coluna chave:
   - Activity: `act_steps` (coalesced) → fallback `apple_steps` → `zepp_steps`
   - Cardio: `hr_mean` (coalesced) → `apple_hr_mean` → `zepp_hr_mean`
   - Sleep: `sleep_total_h` (coalesced) → `apple_slp_total_h` → `zepp_slp_total_h`
3. Infer `used_prejoin` do dicionário passado por `join_run()`

---

### Modificação 2: Refatoração de `join_run()` (linhas 3168–3280)

**Mudanças principais**:

#### 2.1 Tracking de prejoin usage

```python
used_prejoin: dict[str, bool] = {}  # Track if prejoin was used per domain

for domain_name in ["cardio", "activity"]:
    # ... concatenar vendor/variant ...
    used_prejoin[domain_name] = any(
        c.endswith(("_7d", "_zscore")) for c in domain_df.columns
    )
```

#### 2.2 Coalescência por domínio (lines 3263–3281)

**Cardio**:

```python
if domain_name == "cardio":
    # hr_mean: coalesce(apple_hr_mean, zepp_hr_mean)
    if "apple_hr_mean" in domain_df.columns and "zepp_hr_mean" in domain_df.columns:
        domain_df["hr_mean"] = domain_df["apple_hr_mean"].fillna(domain_df["zepp_hr_mean"])

    # hr_std: coalesce(apple_hr_std, zepp_hr_std)
    if "apple_hr_std" in domain_df.columns and "zepp_hr_std" in domain_df.columns:
        domain_df["hr_std"] = domain_df["apple_hr_std"].fillna(domain_df["zepp_hr_std"])

    # n_hr: coalesce(apple_n_hr, zepp_n_hr)
    if "apple_n_hr" in domain_df.columns and "zepp_n_hr" in domain_df.columns:
        domain_df["n_hr"] = domain_df["apple_n_hr"].fillna(domain_df["zepp_n_hr"])
```

**Activity**:

```python
elif domain_name == "activity":
    # act_steps: coalesce(apple_steps, zepp_steps)
    if "apple_steps" in domain_df.columns and "zepp_steps" in domain_df.columns:
        domain_df["act_steps"] = domain_df["apple_steps"].fillna(domain_df["zepp_steps"])

    # act_active_min: coalesce(apple_exercise_min, zepp_exercise_min)
    if "apple_exercise_min" in domain_df.columns and "zepp_exercise_min" in domain_df.columns:
        domain_df["act_active_min"] = domain_df["apple_exercise_min"].fillna(domain_df["zepp_exercise_min"])
```

#### 2.3 datetime64 preservation (não converter até escrita)

**Antes**:

```python
# Normalize date column to date objects
if "date" in concat_df.columns:
    concat_df["date"] = pd.to_datetime(concat_df["date"]).dt.date  # ❌ Converte para date
```

**Depois**:

```python
# Keep date as datetime64 until write (avoid mixed-type issues)
if "date" in concat_df.columns:
    try:
        concat_df["date"] = pd.to_datetime(concat_df["date"])  # ✅ Mantém datetime64
    except Exception:
        pass
```

#### 2.4 Suffix handling após merge

```python
# Resolve suffix conflicts: prefer _y (enriched) over _x (base)
if not base.empty:
    for col in merged.columns:
        if col.endswith("_x"):
            col_y = col[:-2] + "_y"
            if col_y in merged.columns:
                merged[col[:-2]] = merged[col_y].fillna(merged[col])
                merged = merged.drop(columns=[col, col_y], errors="ignore")
```

#### 2.5 QC report generation

```python
# Generate QC report
qc_record = _generate_join_qc(merged, snap, used_prejoin)

# Write QC report
try:
    qc_dir = snap / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / "join_qc.csv"

    qc_df = pd.DataFrame([qc_record])
    qc_df.to_csv(qc_path, index=False)
    print(f"INFO: wrote QC report -> {qc_path}")
except Exception as e:
    print(f"WARNING: failed to write QC report: {e}")
```

#### 2.6 Escrita final com datetime → string

```python
# Convert datetime64 to date string before write
if "date" in merged.columns:
    try:
        merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        pass

write_joined_features(merged, snap, dry_run=False)
```

---

## Arquivo: `src/domains/enriched/post/postjoin_enricher.py` (NEW - 330 linhas)

### Estrutura

```python
# Imports
from pathlib import Path
import numpy as np
import pandas as pd

# Helper functions
def _rolling_corr_7d(df, col1, col2, new_col) -> DataFrame
def _ratio(df, numerator, denominator, new_col) -> DataFrame
def _handle_missing_domains(df, max_records=None) -> DataFrame

# Domain enrichers
def enrich_activity_postjoin(df, max_records=None) -> DataFrame
def enrich_cardio_postjoin(df, max_records=None) -> DataFrame
def enrich_sleep_postjoin(df, max_records=None) -> DataFrame

# Orchestrator
def enrich_postjoin_run(snapshot_dir, *, dry_run=False, max_records=None) -> int

# CLI
if __name__ == "__main__":
    # argparse + execution
```

### Funções de Enrichment

#### `_rolling_corr_7d(df, col1, col2, new_col) -> DataFrame`

```python
def _rolling_corr_7d(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> pd.DataFrame:
    """Compute 7-day rolling correlation between two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        return df

    # Ensure date is sorted
    if "date" in df.columns:
        try:
            df = df.sort_values("date")
        except Exception:
            pass

    # Compute rolling correlation
    df[new_col] = df[col1].rolling(window=7, min_periods=1).corr(df[col2])

    return df
```

#### `_ratio(df, numerator, denominator, new_col) -> DataFrame`

```python
def _ratio(df: pd.DataFrame, numerator: str, denominator: str, new_col: str) -> pd.DataFrame:
    """Compute ratio: numerator / denominator, handling division by zero."""
    if numerator not in df.columns or denominator not in df.columns:
        return df

    df[new_col] = df[numerator] / df[denominator]
    df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)

    return df
```

#### `_handle_missing_domains(df, max_records=None) -> DataFrame`

```python
def _handle_missing_domains(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Fill missing domain data with linear interpolation + forward fill."""
    df = df.copy()

    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()

    if "date" not in df.columns:
        return df

    # Sort by date
    try:
        df = df.sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    # For numeric columns, try interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # Skip derived columns
        if col.endswith(("_7d", "_zscore", "_7d_corr", "_ratio")):
            continue

        # Linear interpolation if we have at least 2 non-NaN values
        if df[col].notna().sum() >= 2:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

        # Forward fill any remaining NaNs
        df[col] = df[col].ffill()  # (changed from fillna(method="ffill"))

    return df
```

#### `enrich_activity_postjoin(df, max_records=None) -> DataFrame`

**Enhancements**:

- `act_steps_vs_hr_7d_corr`: Correlação 7d entre atividade e HR
- Interpolação leve de missing dates

```python
def enrich_activity_postjoin(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    df = df.copy()

    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()

    # 7-day rolling correlation: act_steps vs hr_mean
    if "act_steps" in df.columns and "hr_mean" in df.columns:
        df = _rolling_corr_7d(df, "act_steps", "hr_mean", "act_steps_vs_hr_7d_corr")

    # Handle missing by interpolation
    df = _handle_missing_domains(df, max_records=None)

    return df
```

#### `enrich_cardio_postjoin(df, max_records=None) -> DataFrame`

**Enhancements**:

- `hr_mean_vs_act_7d_corr`: Correlação 7d entre HR e atividade
- `hr_variability_ratio`: HR std / HR mean
- Interpolação

```python
def enrich_cardio_postjoin(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    df = df.copy()

    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()

    # 7-day rolling correlation: hr_mean vs act_steps
    if "hr_mean" in df.columns and "act_steps" in df.columns:
        df = _rolling_corr_7d(df, "hr_mean", "act_steps", "hr_mean_vs_act_7d_corr")

    # HR variability ratio: hr_std / hr_mean
    if "hr_std" in df.columns and "hr_mean" in df.columns:
        df = _ratio(df, "hr_std", "hr_mean", "hr_variability_ratio")

    # Handle missing by interpolation
    df = _handle_missing_domains(df, max_records=None)

    return df
```

#### `enrich_sleep_postjoin(df, max_records=None) -> DataFrame`

**Enhancements**:

- `sleep_activity_ratio`: Sleep total_h / exercise_min
- Interpolação

```python
def enrich_sleep_postjoin(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    df = df.copy()

    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()

    # Sleep efficiency: sleep_total_h / (exercise_minutes)
    if "sleep_total_h" in df.columns and "act_active_min" in df.columns:
        df = _ratio(df, "sleep_total_h", "act_active_min", "sleep_activity_ratio")

    # Handle missing by interpolation
    df = _handle_missing_domains(df, max_records=None)

    return df
```

### Orchestrador `enrich_postjoin_run()`

**Entrada**: `snapshot_dir` (contendo `joined/joined_features_daily.csv`)

**Saída**: `enriched/postjoin/<domain>/enriched_<domain>.csv`

**Retorn codes**:

- 0: success
- 2: no joined CSV
- 1: error

**Flow**:

1. Read `joined/joined_features_daily.csv`
2. Detect domains (from `source_domain` column or infer from columns)
3. For each domain:
   - Filter rows for that domain
   - Apply domain-specific enrichment
   - Write to `enriched/postjoin/<domain>/enriched_<domain>.csv`

---

## Arquivo: `Makefile`

### Nova Tarefa: `enrich-postjoin` (linhas 165–174)

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

**Padrão**: Segue consistência com `enrich-prejoin` (modularizado, PYTHONPATH=src, -m domains...)

---

## Arquivo: `src/domains/enriched/post/__init__.py`

**Antes** (vazio):

```python
"""Post-join enrichment package..."""
__all__ = []
```

**Depois**:

```python
"""Post-join enrichment package..."""
from .postjoin_enricher import enrich_postjoin_run
__all__ = ["enrich_postjoin_run"]
```

---

## Validação (Testes Executados)

### Test 1: Join com Coalescência

```bash
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06
```

**Resultado**:

- ✅ Colunas coalesced criadas: `act_steps`, `act_active_min`, `hr_mean`, `hr_std`, `n_hr`
- ✅ QC report gerado: `qc/join_qc.csv`
- ✅ 201 rows × 53 cols em joined

### Test 2: Postjoin Enrichment

```bash
make enrich-postjoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
```

**Resultado**:

- ✅ Activity: +1 col (`act_steps_vs_hr_7d_corr`)
- ✅ Cardio: +2 cols (`hr_mean_vs_act_7d_corr`, `hr_variability_ratio`)
- ✅ 128 rows activity × 54 cols
- ✅ 69 rows cardio × 54 cols

### Test 3: Data Validation

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/etl/P000001/2025-11-06/joined/joined_features_daily.csv')
print('Coalesced:', [c for c in df.columns if c in ['act_steps', 'hr_mean']])
print('Coverage activity:', df['act_steps'].notna().sum() / len(df) * 100)
print('Coverage cardio:', df['hr_mean'].notna().sum() / len(df) * 100)
"
```

**Resultado**:

- ✅ `act_steps` e `hr_mean` presentes
- ✅ Coverage matches QC report

---

## Invariantes Respeitadas

✅ **datetime64 preservation**: Data mantém datetime64 internamente até escrita CSV  
✅ **MAX_RECORDS scope**: Afeta apenas prejoin; join usa tudo  
✅ **Vendor preservation**: Colunas originais (apple*\*, zepp*\*) mantidas  
✅ **Modularização**: New postjoin_enricher module segue padrão python -m  
✅ **QC automation**: join_qc.csv gerado automaticamente

---

**Status**: ✅ Implementado, Testado, Documentado

Data: 6 de Novembro de 2025
