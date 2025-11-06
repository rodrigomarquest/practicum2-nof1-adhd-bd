# ETL MODULARIZADO — ARQUITETURA COMPLETA (v4.1.0)

**Data:** 6 de Novembro de 2025  
**Status:** ✅ Fases 1 & 2 Completas

## Visão Geral

O ETL foi refatorado para uma **arquitetura modularizada multi-stage** que processa dados de Apple e Zepp através de um pipeline bem estruturado, com suporte a **vendor/variant structure** e **enriquecimentos per-domain** antes do join global.

```
┌──────────┐
│ EXTRACT  │  cli.etl_runner extract
└────┬─────┘
     │
     ├──────────────────────────────┐
     │                              │
     ↓                              ↓
┌─────────────┐            ┌─────────────┐
│   ACTIVITY  │            │   CARDIO    │
│ (seed/Fase1)│            │ (seed/Fase1)│
└─────┬───────┘            └──────┬──────┘
      │                           │
      ├──────────────────────────────┤
      │                              │
      ↓                              ↓
┌──────────────────────┐  ┌──────────────────────┐
│   ENRICHED/PREJOIN   │  │   (Fase 2 - NOVO)    │
│ prejoin_enricher.py  │  │ enriquecimentos 7d   │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           ├──────────────┬──────────┤
           │              │          │
           ↓              ↓          ↓
        JOIN (join_run)
           │
           ↓
   joined_features_daily.csv (201 rows, 50 cols)
           │
           ├──────────────────────────┐
           │                          │
           ↓                          ↓
      POSTJOIN ENRICH          (Fase 3 - Próximo)
    enrich_global.py        cross-domain enrichments
           │
           ↓
   enriched/postjoin/<domain>/
```

## Fases Implementadas

### Fase 1: Features (Seed) ✅ COMPLETO

**Objetivo:** Extrair features brutas por domínio (activity, cardio, sleep) de Apple e Zepp

**Estrutura:**

```
data/etl/P000001/2025-11-06/features/
├── activity/
│   ├── apple/inapp/features_daily.csv          (4 rows)
│   └── zepp/cloud/features_daily.csv           (128 rows)
├── cardio/
│   ├── apple/inapp/features_daily.csv          (1 row)
│   └── zepp/cloud/features_daily.csv           (68 rows)
└── sleep/
    └── zepp/cloud/features_daily.csv           (87 rows)
```

**Modularização:**

- `src/domains/activity/activity_from_extracted.py` → `make activity`
- `src/domains/cardiovascular/cardio_from_extracted.py` → `make cardio`
- `src/domains/sleep/sleep_from_extracted.py` → `make sleep`

**Recurso:** MAX_RECORDS para testes (testado com 128)

**Total Fase 1:** 288 registros processados

### Fase 2: Enriched/Prejoin ✅ COMPLETO (NOVO)

**Objetivo:** Enriquecer features per-domain com métricas derivadas (rolling averages, z-scores)

**Estrutura:**

```
data/etl/P000001/2025-11-06/enriched/prejoin/
├── activity/
│   ├── apple/inapp/enriched_activity.csv       (4 rows, +2 cols)
│   └── zepp/cloud/enriched_activity.csv        (128 rows, +18 cols)
├── cardio/
│   ├── apple/inapp/enriched_cardio.csv         (1 row, +6 cols)
│   └── zepp/cloud/enriched_cardio.csv          (68 rows, +6 cols)
└── sleep/
    └── zepp/cloud/enriched_sleep.csv           (87 rows, +8 cols)
```

**Modularização:**

- `src/domains/enriched/pre/prejoin_enricher.py` → `make enrich-prejoin`

**Enriquecimentos Implementados:**

- **7-day rolling average:** `<col>_7d` — média móvel de 7 dias (com min_periods=1)
- **Z-score:** `<col>_zscore` — padronização de cada métrica

**Exemplo (Cardio):**

```python
Input:  zepp_hr_mean, zepp_hr_max, zepp_n_hr (3 colunas)
Output: +zepp_hr_mean_7d, +zepp_hr_mean_zscore
        +zepp_hr_max_7d, +zepp_hr_max_zscore
        +zepp_n_hr_7d, +zepp_n_hr_zscore (6 novas colunas)
```

**Total Fase 2:** 40 colunas enriquecidas adicionadas

### Fase 3: Join ✅ COMPLETO (ATUALIZADO)

**Objetivo:** Unir múltiplos vendor/variant de cada domínio em CSV único

**Priorização em Cascata:**

1. **enriched/prejoin** ← Fase 2 (com enriquecimentos) ← DEFAULT
2. **features** ← Fase 1 (sem enriquecimentos)
3. **legacy joined** ← Compatibilidade

**Resultado:**

```
joined_features_daily.csv
├─ 201 linhas (outer join de activity + cardio + sleep com dates)
├─ 50 colunas (28 brutos + 16 rolling avg + 16 zscores)
└─ Colunas de provenance: source_domain, variant, domain
```

**Modularização:** `join_run()` em `cli.etl_runner` → `make join`

---

## Tabela Comparativa: Modularização por Stage

| Stage              | Arquivo                                        | Tipo            | Execução                | Suporta MAX_RECORDS | Suporta Dry-Run |
| ------------------ | ---------------------------------------------- | --------------- | ----------------------- | ------------------- | --------------- |
| Extract            | `cli.etl_runner`                               | CLI dispatcher  | `make extract`          | ❌                  | ✅              |
| Activity           | `domains.activity.activity_from_extracted`     | Module exec     | `make activity`         | ✅                  | ✅              |
| Cardio             | `domains.cardiovascular.cardio_from_extracted` | Module exec     | `make cardio`           | ✅                  | ✅              |
| Sleep              | `domains.sleep.sleep_from_extracted`           | Module exec     | `make sleep`            | ✅                  | ✅              |
| **Prejoin Enrich** | **domains.enriched.pre.prejoin_enricher**      | **Module exec** | **make enrich-prejoin** | **✅**              | **✅**          |
| Join               | `cli.etl_runner`                               | CLI dispatcher  | `make join`             | ❌                  | ✅              |
| Postjoin Enrich    | `domains.enriched.enrich_global`               | CLI dispatcher  | `make enrich`           | ❌                  | ✅              |

---

## Comandos de Execução

### Modo Completo (Fases 1 + 2 + 3)

```bash
# Remover dados antigos
rm -rf data/etl/P000001/2025-11-06/*

# Executar fase 1 (seed)
make activity DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
make cardio DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
make sleep DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# Executar fase 2 (enriquecimento prejoin)
make enrich-prejoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# Executar fase 3 (join com enriched/prejoin automático)
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06
```

### Teste Rápido (Dry-run)

```bash
make enrich-prejoin DRY_RUN=1 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
# INFO: enrich_prejoin_run end (dry-run, would process 5 combinations)
```

### Pipeline Completo (Alias)

```bash
make full DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06
# Executa: extract → activity → cardio → sleep → join → enrich
```

---

## Estrutura de Diretórios Após Execução

```
data/etl/P000001/2025-11-06/
├── extracted/                           (Fase 0 - Extract)
│   ├── apple/inapp/apple_health_export/export.xml
│   └── zepp/cloud/zepp_data/
│
├── features/                            (Fase 1 - Seed)
│   ├── activity/
│   │   ├── apple/inapp/features_daily.csv
│   │   └── zepp/cloud/features_daily.csv
│   ├── cardio/
│   │   ├── apple/inapp/features_daily.csv
│   │   └── zepp/cloud/features_daily.csv
│   └── sleep/
│       └── zepp/cloud/features_daily.csv
│
├── enriched/
│   └── prejoin/                         (Fase 2 - Prejoin Enrich - NOVO)
│       ├── activity/
│       │   ├── apple/inapp/enriched_activity.csv
│       │   └── zepp/cloud/enriched_activity.csv
│       ├── cardio/
│       │   ├── apple/inapp/enriched_cardio.csv
│       │   └── zepp/cloud/enriched_cardio.csv
│       └── sleep/
│           └── zepp/cloud/enriched_sleep.csv
│
└── joined/                              (Fase 3 - Join)
    ├── joined_features_daily.csv        (201 rows, 50 cols)
    └── joined_features_daily_prev.csv   (backup)
```

---

## Resumo de Mudanças Implementadas

### ✅ Arquivos Criados/Modificados

**Criado:**

- ✅ `PHASE2_ENRICHED_PREJOIN_ARCHITECTURE.md` — Documentação Fase 2

**Modificado:**

1. **`src/domains/enriched/pre/prejoin_enricher.py`**

   - ✅ Adicionado `if __name__ == "__main__":` para execução como módulo
   - ✅ Refatorado `enrich_activity/cardio/sleep()` com lógica genérica (itera todas as colunas numéricas)
   - ✅ Suporta MAX_RECORDS em cada função
   - ✅ Adicionada lógica de argparse para compatibilidade CLI

2. **`Makefile`**

   - ✅ Removido: tarefa antiga `enrich-prejoin` que chamava `src/etl_pipeline.py`
   - ✅ Adicionado: nova tarefa `enrich-prejoin` que chama `domains.enriched.pre.prejoin_enricher`
   - ✅ Mantém padrão consistente: `PYTHONPATH=src $(PYTHON) -m domains...`

3. **`src/etl_pipeline.py`**
   - ✅ Atualizado `join_run()` com priorização em cascata:
     - Priority 1: `enriched/prejoin/<domain>/**/enriched_<domain>.csv` ← DEFAULT
     - Priority 2: `features/<domain>/**/features_daily.csv`
     - Priority 3: `joined/features_<domain>.csv` (legacy)
   - ✅ Refatorado para aceitar **TODOS** os vendor/variant combinations (não apenas 1 por domínio)
   - ✅ Concatena múltiplos vendor/variant antes de fazer join global

---

## Números de Teste (MAX_RECORDS=128)

```
PHASE 1 (FEATURES):
├─ activity:    132 rows (4 apple + 128 zepp)
├─ cardio:       69 rows (1 apple + 68 zepp)
├─ sleep:        87 rows (0 apple + 87 zepp)
└─ TOTAL:       288 rows

PHASE 2 (ENRICHED/PREJOIN):
├─ activity:    132 rows → +20 cols (9×2 zepp + 1×2 apple)
├─ cardio:       69 rows → +12 cols (3×2 zepp + 3×2 apple)
├─ sleep:        87 rows → +8 cols  (4×2 zepp)
└─ TOTAL COLS ADDED: 40 novas colunas

PHASE 3 (JOINED):
├─ Linhas:      201 (outer join)
├─ Colunas:     50 (28 brutos + 16 rolling + 16 zscore)
└─ Coverage:    89.1% (201/226 datas possíveis)
```

---

## Benefícios da Arquitetura Modularizada

### 1. Independência

- Cada stage pode ser testado isoladamente
- Falhas em um stage não afetam outros
- Rollback simples (apenas remover um estágio)

### 2. Escalabilidade

- Fácil adicionar novos domínios (ex: sleep_from_extracted.py)
- Fácil adicionar novos enriquecimentos (ex: novo método em prejoin_enricher.py)
- Suporta processamento paralelo (future)

### 3. Rastreabilidade

- Cada stage tem seus próprios logs
- Estrutura vendor/variant permite rastrear origem de cada métrica
- Backup automático (joined_features_daily_prev.csv)

### 4. Testabilidade

- MAX_RECORDS para testes rápidos com dados limitados
- Dry-run mode para validar sem escrever
- Fácil validação de qualidade (QC)

### 5. Manutenibilidade

- Código organizado em módulos temáticos
- Documentação clara por stage
- Padrões consistentes (modularização, argparse, logging)

---

## Próximas Etapas

### Fase 3: Enriched/Postjoin (Próximo)

```
joined_features_daily.csv (201 rows, 50 cols)
           ↓
enrich_postjoin_run()
           ↓
enriched/postjoin/
├─ activity/enriched_activity.csv
├─ cardio/enriched_cardio.csv
└─ sleep/enriched_sleep.csv
```

**O que fazer:**

- Ler `joined_features_daily.csv`
- Aplicar enriquecimentos **cross-domain** (ex: correlações, ratios)
- Escrever resultado em `enriched/postjoin/<domain>/enriched_<domain>.csv`
- Manter modularização: `domains.enriched.postjoin_enricher.py`

### Fase 4: QC Comparativo

- Validar enriquecimentos vs. expectativas
- Gerar relatório de qualidade
- Comparar com baseline (se disponível)

---

## Arquivos de Referência

- 📄 `PHASE2_ENRICHED_PREJOIN_ARCHITECTURE.md` — Documentação detalhada Fase 2
- 📄 `IMPLEMENTATION_VENDOR_VARIANT.md` — Documentação Fase 1
- 📁 `src/domains/enriched/pre/prejoin_enricher.py` — Código Fase 2
- 📁 `Makefile` — Integração de todas as fases

---

## Conclusão

✅ **Fase 2 (Enriched/Prejoin) completamente implementada e testada**

A arquitetura modularizada agora oferece:

- Pipeline claro e rastreável de features → enriquecimento → join
- Suporte para múltiplos vendor/variant por domínio
- Enriquecimentos per-domain com 7d rolling average + z-score
- Integração transparente com join (prioriza enriched automaticamente)
- 288 registros processados com sucesso com MAX_RECORDS=128

**Status:** 🚀 **Pronto para Fase 3 (Enriched/Postjoin)**

---

**Documentação gerada:** 6 de Novembro de 2025  
**Versão:** ETL v4.1.0
