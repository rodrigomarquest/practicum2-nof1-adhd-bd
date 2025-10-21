# Project Memory — Practicum2 N-of-1 ADHD+BD

## Snapshot atual de referência
- PID: P000001
- SNAP: 2025-09-29
- Cutover: 2024-03-11
- TZ: before=America/Sao_Paulo, after=Europe/Dublin

## Arquitetura (resumo)
- `etl_pipeline.py` com subcomandos: `extract`, `cardio`, `full`.
- Saídas:
  - `data_ai/<PID>/snapshots/<SNAP>/per-metric/*` (apple_*).
  - `features_cardiovascular.csv`, `features_daily_updated.csv`.
- Manifests: `extract_manifest.json`, `cardio_manifest.json`.

## Decisões fixas
- **Sem fallback** de dados: corrigir ETLs na fonte.
- Escrita **atômica** para CSV/JSON.
- Idempotência por arquivos de saída e manifests.

## Rotas comuns
- Run extract → cardio (ou full).
- Geração notebook 03 EDA (plus opcional).
- QC: validar existence + size dos CSVs chave antes de EDA.

## Pendências
- Consolidar Zepp loader definitivo.
- Ajustes de notebook plus (gráficos já ok no simples).
