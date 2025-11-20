# Refactoring: NB2 → ML6, NB3 → ML7

**Data**: $(date +%Y-%m-%d)
**Objetivo**: Eliminar confusão entre naming de estágios de modelagem (NB2/NB3) e notebooks Jupyter (NB0, NB1, NB2.ipynb, NB3.ipynb)

## Convenção Nova

- **ML6** = Stage 6 (Modeling/Machine Learning) - Static daily classifier (LogisticRegression, tabular models)
- **ML7** = Stage 7 (Modeling/Machine Learning) - LSTM sequence classifier (seq_len=14, SHAP, drift)

## Arquivos/Diretórios Renomeados

### Diretórios de Dados
- `data/ai/P000001/2025-11-07/nb2/` → `ml6/`
- `data/ai/P000001/2025-11-07/nb3/` → `ml7/`

### Arquivos CSV
- `features_daily_nb2.csv` → `features_daily_ml6.csv`
- `features_nb2_clean.csv` → `features_ml6_clean.csv`

### Módulos Python
- `src/models/run_nb2.py` → `run_ml6.py`
- `src/etl/nb3_analysis.py` → `ml7_analysis.py`
- `scripts/prepare_nb2_dataset.py` → `prepare_ml6_dataset.py`

### Funções Renomeadas
- `stage_5_prep_nb2()` → `stage_5_prep_ml6()`
- `stage_6_nb2()` → `stage_6_ml6()`
- `stage_7_nb3()` → `stage_7_ml7()`
- `prepare_nb3_features()` → `prepare_ml7_features()`

### Constantes
- `NB3_FEATURE_COLS` → `ML7_FEATURE_COLS`
- `NB3_ANTI_LEAK_COLS` → `ML7_ANTI_LEAK_COLS`

### Makefile Targets
- `prep-nb2` → `prep-ml6`
- `nb2` → `ml6`
- `nb3` → `ml7`
- `nb2-only` → `ml6-only`
- `nb3-only` → `ml7-only`

## Arquivos Atualizados

### Código Python (Principais)
- ✅ `scripts/run_full_pipeline.py` (1,163 linhas)
- ✅ `src/models/run_ml6.py`
- ✅ `src/etl/ml7_analysis.py`
- ✅ `scripts/prepare_ml6_dataset.py`
- ✅ `src/models/__init__.py`
- ✅ `src/nb_common/{env,portable,__init__}.py`

### Documentação
- ✅ `README.md`
- ✅ `pipeline_overview.md`
- ✅ `RUN_REPORT.md`
- ✅ `Makefile`
- ✅ `docs/**/*.md` (53 arquivos atualizados)

### Build/Config
- ✅ `Makefile`
- ✅ Paths de CV/SHAP/Drift em configs

## NOT Modificado (Conforme Solicitado)

- ❌ `notebooks/NB2_Baseline.ipynb` (usuário atualizará manualmente)
- ❌ `notebooks/NB3_DeepLearning.ipynb` (usuário atualizará manualmente)
- ❌ `notebooks/NB0*.ipynb`, `NB1*.ipynb`
- ❌ `archive/` (backups históricos preservados)

## Verificações de Sanidade

### Lint
```bash
# No errors found in main files
python -m pylint scripts/run_full_pipeline.py  # OK
python -m pylint src/etl/ml7_analysis.py       # OK
python -m pylint src/models/run_ml6.py         # OK
```

### Testes Recomendados
```bash
# 1. Pipeline completo (Stages 5-7)
make pipeline PARTICIPANT=P000001 SNAPSHOT=2025-11-07

# 2. QC end-to-end
make qc-all SNAPSHOT=2025-11-07

# 3. Verificar outputs
ls -la data/ai/P000001/2025-11-07/ml6/
ls -la data/ai/P000001/2025-11-07/ml7/

# 4. Verificar logs (Stage 5: ML6, Stage 6: ML6, Stage 7: ML7)
tail -100 scripts/logs/pipeline_*.log
```

## Status

- ✅ Discovery completo (200+ ocorrências mapeadas)
- ✅ Diretórios/arquivos renomeados
- ✅ Módulos Python renomeados
- ✅ Código refatorado (run_full_pipeline.py + módulos)
- ✅ Documentação atualizada (README + docs/)
- ✅ Makefile atualizado
- ✅ Lint checks passando
- ⏳ **PENDING**: Test run completo do pipeline
- ⏳ **PENDING**: Commit + push (aguardando validação final)

## Commits Sugeridos

```bash
git add -A
git commit -m "refactor: rename NB2→ML6, NB3→ML7 to avoid notebook numbering confusion

- Rename directories: nb2/ → ml6/, nb3/ → ml7/
- Rename modules: run_nb2.py → run_ml6.py, nb3_analysis.py → ml7_analysis.py
- Update all Python code, docs, Makefile (200+ occurrences)
- Preserve .ipynb notebooks (user will update manually)
- Closes #<issue-number>"
```

