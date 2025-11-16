# Canonical Python Entrypoints

**Generated:** 2025-11-16  
**Purpose:** Lista definitiva de todos os entrypoints Python referenciados pelo Makefile, testes e pipeline principal. **NÃO DELETAR OU MOVER** nenhum módulo importado por estes entrypoints.

---

## 1. Makefile Entrypoints

### 1.1 Main Pipeline Orchestrator

- **`scripts/run_full_pipeline.py`**
  - Executa stages 0-9 (Ingest → Report)
  - Usado por: `ingest`, `aggregate`, `unify`, `segment`, `label`, `nb2`, `nb3`, `report`, `pipeline`, `quick`, `nb2-only`, `nb3-only`

### 1.2 Data Preparation Scripts

- **`scripts/prepare_nb2_dataset.py`**
  - Remove label leakage, prepara dataset limpo para NB2
  - Usado por: `prep-nb2` target

---

## 2. scripts/run_full_pipeline.py Imports

### 2.1 Direct Imports (Stages)

```python
from src.etl.stage_csv_aggregation import run_csv_aggregation
from src.etl.stage_unify_daily import run_unify_daily
from src.etl.stage_apply_labels import run_apply_labels
```

### 2.2 Lazy Imports (NB3 Analysis)

```python
# Used in stage_6_nb2():
from src.etl.nb3_analysis import create_calendar_folds

# Used in stage_7_nb3():
from src.etl.nb3_analysis import (
    create_calendar_folds,
    compute_shap_values,
    detect_drift_adwin,
    detect_drift_ks_segments,
    create_lstm_sequences,
    train_lstm_model
)

# Used in stage_8_tflite():
from src.etl.nb3_analysis import convert_to_tflite, measure_latency
```

---

## 3. Test Suite Imports

### 3.1 Zepp Data Loaders

```python
# tests/test_zepp_sleep_loader.py
from src.domains.parse_zepp_export import discover_zepp_tables
from src.domains.sleep.sleep_from_extracted import load_zepp_sleep_daily

# tests/test_zepp_cardio_loader.py
from src.domains.parse_zepp_export import discover_zepp_tables
from src.domains.cardiovascular.cardio_from_extracted import load_zepp_cardio_daily

# tests/test_zepp_activity_seed.py
from src.domains.parse_zepp_export import discover_zepp_tables
from src.domains.activity.zepp_activity import load_zepp_activity_daily
```

### 3.2 Legacy Imports (etl_modules compatibility)

```python
# tests/test_io_utils.py
from etl_modules.io_utils import read_csv_sniff
```

### 3.3 ETL Pipeline Tests

```python
# tests/test_cli_extract_logging.py
import src.etl_pipeline as etl_mod
```

---

## 4. Script Entrypoints (scripts/)

### 4.1 Biomarker Extraction

- **`scripts/extract_biomarkers.py`**
  ```python
  from src.biomarkers import aggregate
  ```

### 4.2 Zepp Data Preparation

- **`scripts/prepare_zepp_data.py`**
  - Prepara dados Zepp raw para biomarkers
  - Standalone script (no src imports beyond pathlib)

### 4.3 NB2 Dataset Preparation

- **`scripts/prepare_nb2_dataset.py`**
  - Remove label leakage
  - Standalone script (pandas/numpy only)

---

## 5. Core Module Dependencies (src/)

### 5.1 ETL Stage Modules (CANONICAL)

**Location:** `src/etl/`

#### Stage Executors

- **`stage_csv_aggregation.py`**

  - Função: `run_csv_aggregation(participant, extracted_dir, output_dir)`
  - Parse export.xml + Zepp CSVs → daily\_\*.csv

- **`stage_unify_daily.py`**

  - Função: `run_unify_daily(participant, snapshot, extracted_dir, output_dir)`
  - Merge Apple + Zepp → features_daily_unified.csv

- **`stage_apply_labels.py`**
  - Função: `run_apply_labels(participant, snapshot, etl_dir)`
  - Apply PBSI labels → features_daily_labeled.csv

#### NB3 Analysis Module

- **`nb3_analysis.py`**
  - Funções:
    - `create_calendar_folds(df, n_folds, train_months, val_months)`
    - `compute_shap_values(model, X)`
    - `detect_drift_adwin(feature_series, delta)`
    - `detect_drift_ks_segments(feature_series, segments)`
    - `create_lstm_sequences(X, y, sequence_length)`
    - `train_lstm_model(X_train, y_train, X_val, y_val, ...)`
    - `convert_to_tflite(keras_model, output_path)`
    - `measure_latency(tflite_path, sample_input, n_runs)`

### 5.2 Domain Modules (CANONICAL)

**Location:** `src/domains/`

#### Zepp Parsing

- **`parse_zepp_export.py`**
  - Função: `discover_zepp_tables(zepp_dir)`
  - Descobre tabelas CSV em export Zepp

#### Sleep Domain

- **`domains/sleep/sleep_from_extracted.py`**
  - Função: `load_zepp_sleep_daily(zepp_dir)`

#### Cardiovascular Domain

- **`domains/cardiovascular/cardio_from_extracted.py`**
  - Função: `load_zepp_cardio_daily(zepp_dir)`

#### Activity Domain

- **`domains/activity/zepp_activity.py`**
  - Função: `load_zepp_activity_daily(zepp_dir)`

#### Biomarkers

- **`biomarkers/aggregate.py`** (usado por extract_biomarkers.py)
  - Função: `aggregate(data_files, output_path)`

### 5.3 ETL Infrastructure (CANONICAL)

**Location:** `src/etl/`

#### Configuration & I/O

- **`config.py`** - ETL configuration
- **`io_utils.py`** - File I/O utilities (também em `etl_modules/io_utils.py` - legacy)

#### Parsing Modules

- **`apple_raw_to_per_metric.py`** - Apple export.xml parsing
- **`zepp_join.py`** - Zepp data joining

---

## 6. Legacy Module Compatibility

### 6.1 etl_modules/ (ARCHIVED)

**Status:** Movido para `src/etl/` em v4.1.1  
**Compatibility Layer:** Alguns testes ainda importam `etl_modules.io_utils`

**Migração:**

```python
# OLD (deprecated):
from etl_modules.io_utils import read_csv_sniff

# NEW (canonical):
from src.etl.io_utils import read_csv_sniff
```

---

## 7. Import Graph (Dependency Tree)

### Level 0: CLI Entrypoints (NO src/ imports)

- `scripts/prepare_zepp_data.py`
- `scripts/prepare_nb2_dataset.py`

### Level 1: Stage Executors

- `scripts/run_full_pipeline.py`
  - → `src.etl.stage_csv_aggregation`
  - → `src.etl.stage_unify_daily`
  - → `src.etl.stage_apply_labels`
  - → `src.etl.nb3_analysis` (lazy import)

### Level 2: Domain Loaders

- `scripts/extract_biomarkers.py`

  - → `src.biomarkers.aggregate`

- Test suite:
  - → `src.domains.parse_zepp_export`
  - → `src.domains.sleep.sleep_from_extracted`
  - → `src.domains.cardiovascular.cardio_from_extracted`
  - → `src.domains.activity.zepp_activity`

### Level 3: Core Infrastructure

All stage executors depend on:

- `src/etl/config.py`
- `src/etl/io_utils.py`
- `src/etl/parse_zepp_export.py`

---

## 8. Deletion Safety Rules

### ✅ SAFE TO DELETE

- Arquivos em `archive/`
- Notebooks em `notebooks/` (exceto se referenciados no Makefile)
- Documentação em `docs/` (exceto release notes)
- Scripts em `tools/` não referenciados

### ⚠️ REQUIRES VALIDATION

- Qualquer arquivo em `src/domains/` não listado acima
- Arquivos em `src/tools/` (verificar se usado por pipeline)
- Arquivos em `src/features/`, `src/modeling/`, `src/models/`

### ❌ NEVER DELETE

Todos os módulos listados nas seções 5.1, 5.2, 5.3:

- `src/etl/stage_*.py` (3 arquivos)
- `src/etl/nb3_analysis.py`
- `src/domains/parse_zepp_export.py`
- `src/domains/*/loader.py` ou `*_from_extracted.py`
- `src/biomarkers/aggregate.py`
- `src/etl/config.py`, `src/etl/io_utils.py`

---

## 9. Verification Commands

### Check Makefile references:

```bash
grep -E "\.py|python|PYTHON" Makefile
```

### Check test imports:

```bash
grep -rE "^from src\.|^import src\." tests/
```

### Check pipeline imports:

```bash
grep -E "^from src\.|^import src\." scripts/run_full_pipeline.py
```

### Find all src module imports:

```bash
grep -rE "^from src\.|^import src\." src/ scripts/ tests/
```

---

## 10. Module Inventory Summary

### Canonical Entrypoints (8 total)

1. `scripts/run_full_pipeline.py` ⭐ **MAIN ORCHESTRATOR**
2. `scripts/prepare_nb2_dataset.py`
3. `scripts/extract_biomarkers.py`
4. `scripts/prepare_zepp_data.py`

### Core Stage Modules (4 total)

1. `src/etl/stage_csv_aggregation.py`
2. `src/etl/stage_unify_daily.py`
3. `src/etl/stage_apply_labels.py`
4. `src/etl/nb3_analysis.py`

### Domain Loaders (4 total)

1. `src/domains/parse_zepp_export.py`
2. `src/domains/sleep/sleep_from_extracted.py`
3. `src/domains/cardiovascular/cardio_from_extracted.py`
4. `src/domains/activity/zepp_activity.py`

### Infrastructure (3 total)

1. `src/etl/config.py`
2. `src/etl/io_utils.py`
3. `src/biomarkers/aggregate.py`

### Supporting Modules (variable count)

- `src/etl/apple_raw_to_per_metric.py`
- `src/etl/zepp_join.py`
- `src/etl/iphone_backup/` (pasta completa)
- `src/etl/cardiovascular/` (pasta completa)
- `src/etl/common/` (pasta completa)

---

## 11. CI/CD Integration Points

### GitHub Actions (if exists):

- Should run: `make pipeline PID=P000001 SNAPSHOT=auto`
- Depends on: `scripts/run_full_pipeline.py` + all stage modules

### Pre-commit Hooks (if exists):

- Lint checks: `pylint src/etl/stage_*.py`
- Type checks: `mypy scripts/run_full_pipeline.py`

---

**Última Atualização:** 2025-11-16  
**Versão Pipeline:** v4.1.2  
**Mantenedor:** Rodrigo Marques Teixeira

---

## Apêndice A: Full Import List (Alphabetical)

```python
# Stage executors (run_full_pipeline.py)
from src.etl.stage_csv_aggregation import run_csv_aggregation
from src.etl.stage_unify_daily import run_unify_daily
from src.etl.stage_apply_labels import run_apply_labels

# NB3 analysis (lazy imports in run_full_pipeline.py)
from src.etl.nb3_analysis import create_calendar_folds
from src.etl.nb3_analysis import compute_shap_values
from src.etl.nb3_analysis import detect_drift_adwin
from src.etl.nb3_analysis import detect_drift_ks_segments
from src.etl.nb3_analysis import create_lstm_sequences
from src.etl.nb3_analysis import train_lstm_model
from src.etl.nb3_analysis import convert_to_tflite
from src.etl.nb3_analysis import measure_latency

# Biomarkers (extract_biomarkers.py)
from src.biomarkers import aggregate

# Domain loaders (test suite)
from src.domains.parse_zepp_export import discover_zepp_tables
from src.domains.sleep.sleep_from_extracted import load_zepp_sleep_daily
from src.domains.cardiovascular.cardio_from_extracted import load_zepp_cardio_daily
from src.domains.activity.zepp_activity import load_zepp_activity_daily

# Legacy compatibility (tests)
from etl_modules.io_utils import read_csv_sniff

# ETL pipeline module (tests)
import src.etl_pipeline as etl_mod
```

---

**🔒 PROTECTION LEVEL: CANONICAL** - Não modificar sem approval do maintainer.
