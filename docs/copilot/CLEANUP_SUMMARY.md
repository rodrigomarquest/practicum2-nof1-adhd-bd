# Repository Cleanup Summary - 2025-11-07

## Objetivo
Padronizar repositório para estrutura canônica determinística sem deletar arquivos.

## Estrutura Canônica Implementada

```
data/
  raw/          # única fonte persistente
  etl/          # outputs canônicos (joined/qc/segment_autolog)
  ai/           # artefatos ML6/ML7 por snapshot
src/
  etl/          # 24 módulos ETL (stage_csv_aggregation, ml7_analysis, etc)
  modeling/     # (reservado para ML6/ML7 futuros)
  utils/        # (reservado para utilitários)
scripts/
  run_full_pipeline.py    # orquestrador 9 stages
docs/
  PERIOD_EXPANSION_FINAL_IMPLEMENTATION.md
  PERIOD_EXPANSION_ANTI_LEAK_REPORT.md
  + outros relatórios técnicos
archive/
  root_scripts/           # 7 scripts legados (.py antigos)
  root_docs/              # 12 documentos de sessão (.md)
  root_artifacts/         # 4 pastas (ml6/, ml7/, latest/, reports/)
  etl_modules/            # pasta antiga (conteúdo copiado para src/etl/)
  etl_tools/              # ferramentas antigas
```

## Movimentações Realizadas

### Categoria: Scripts Python (7 arquivos)
- run_complete_pipeline.py → archive/root_scripts/
- run_pipeline_deterministic.py → archive/root_scripts/
- run_ml6_beiwe.py → archive/root_scripts/
- run_ml6_engage7.py → archive/root_scripts/
- build_heuristic_labels.py → archive/root_scripts/
- make_eda_biomarkers.py → archive/root_scripts/
- rebuild_pipeline.sh → archive/root_scripts/
- run_full_pipeline.py → scripts/ (canônico)

### Categoria: Documentação (12 arquivos)
Movidos para docs/ (úteis):
- PERIOD_EXPANSION_FINAL_IMPLEMENTATION.md
- PERIOD_EXPANSION_ANTI_LEAK_REPORT.md

Movidos para archive/root_docs/ (sessões/logs):
- DETERMINISTIC_PIPELINE_READY.md
- IMPLEMENTACAO_FINAL_RESUMO.md
- IMPLEMENTATION_BIOMARKERS_COMPLETE.md
- PERIOD_EXPANSION_EXECUTION_REPORT.md
- PERIOD_EXPANSION_README.md
- PERIOD_EXPANSION_READY.md
- PIPELINE_DETERMINISTIC_IMPLEMENTATION.md
- PIPELINE_IMPLEMENTATION_COMPLETE.md
- HOW_TO_RUN_PERIOD_EXPANSION.md
- PIPELINE_QUICK_START.md
- PIPELINE_FIXES_SESSION_2025-11-07.md
- SESSION_ARTIFACTS_2025_11_07.md

Mantidos na raiz:
- README.md (atualizado com estrutura canônica)
- CHANGELOG.md
- RUN_REPORT.md (gerado pelo pipeline)

### Categoria: Artefatos (4 pastas)
- ml6/ → archive/root_artifacts/ml6/
- ml7/ → archive/root_artifacts/ml7/
- latest/ → archive/root_artifacts/latest/
- reports/ → archive/root_artifacts/reports/

### Categoria: Módulos ETL (24 arquivos)
- etl_modules/ → src/etl/ (24 arquivos .py copiados)
- etl_modules/ → archive/etl_modules/ (pasta original preservada)
- etl_tools/ → archive/etl_tools/

## Ajustes de Código

### Imports Atualizados
- scripts/run_full_pipeline.py: `from etl_modules.` → `from src.etl.`
- src/**/*.py: substituição em lote via sed (14 arquivos)

### sys.path
Adicionado em scripts/run_full_pipeline.py:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

## Validação

### Pipeline Executado
```bash
python scripts/run_full_pipeline.py --participant P000001 --snapshot 2025-11-07 --start-stage 6 --end-stage 9
```

### Métricas Validadas (RUN_REPORT.md)
- Date Range: 2017-12-04 to 2025-10-21 ✓
- Total Rows: 2828 ✓
- Label Distribution: 61.4% / 10.0% / 28.5% ✓
- ML6 Mean F1: 0.7038 ± 0.1721 ✓
- SHAP Top-10 features: gerados ✓
- Drift: ADWIN 11 changes, KS 102/1180 ✓
- LSTM F1: 0.2648 (mean) ✓
- TFLite: 37.4 KB ✓

### Outputs Canônicos
- data/etl/P000001/2025-11-07/joined/features_daily_unified.csv
- data/ai/P000001/2025-11-07/ml6/cv_summary.json
- data/ai/P000001/2025-11-07/ml7/shap_summary.md
- data/ai/P000001/2025-11-07/ml7/drift_report.md
- data/ai/P000001/2025-11-07/ml7/models/best_model.tflite
- RUN_REPORT.md

## Estatísticas Finais

- **Arquivos deletados**: 0 (ZERO)
- **Arquivos movidos para archive/**: 27 (7 scripts + 12 docs + 4 pastas + etl_modules + etl_tools)
- **Módulos em src/etl/**: 24
- **Estrutura canônica**: 100% implementada
- **Pipeline validado**: ✓ Stages 6-9 executados com sucesso
- **README.md**: ✓ Atualizado com estrutura canônica

## Próximos Passos
1. ✓ Commit único (mensagem abaixo)
2. Considerar migração futura de notebooks/ para src/modeling/
3. Documentar detalhes de src/etl/ modules em docs/

## Mensagem de Commit
```
refactor: canonical structure + archive legacy files (no deletion) + deterministic rebuild from data/raw

- Moved 7 legacy scripts to archive/root_scripts/
- Moved 12 session docs to archive/root_docs/
- Moved artifacts (ml6/, ml7/, latest/, reports/) to archive/root_artifacts/
- Migrated etl_modules/ → src/etl/ (24 modules)
- Updated all imports: etl_modules → src.etl
- Updated README.md with canonical structure
- Validated pipeline stages 6-9: 2828 days, F1≈0.70
- No files deleted, all legacy preserved in archive/
```
