# Handoff: Fase 3 → Fase 4

**Data**: 6 de Novembro de 2025  
**Status**: ✅ Fase 3 COMPLETA e PRONTA para Handoff

---

## Resumo da Fase 3

### ✅ Implementado

**1. Join com Coalescência Leve**

- 5 colunas coalesced: `act_steps`, `act_active_min`, `hr_mean`, `hr_std`, `n_hr`
- Mantém TODAS as colunas originais por vendor (apple*\*, zepp*\*)
- Output: `joined/joined_features_daily.csv` (201 rows × 53 cols)

**2. QC Report Automático**

- Arquivo: `qc/join_qc.csv`
- Contém: n*rows, date_min/date_max, coverage*_, used*prejoin*_
- Exemplo: 201 rows, 2018-04-06 → 2025-07-28, coverage_activity 65.67%, coverage_cardio 34.33%

**3. Enriquecimentos Cross-Domain (Postjoin)**

- Activity: +1 col (`act_steps_vs_hr_7d_corr`)
- Cardio: +2 cols (`hr_mean_vs_act_7d_corr`, `hr_variability_ratio`)
- Sleep: structure ready (not processed in test snapshot)
- Output: `enriched/postjoin/<domain>/enriched_<domain>.csv`

**4. Documentação Completa**

- 7 arquivos markdown em `/docs/`
- Guias técnicos, arquitetura, quick reference, status

---

## Arquivos Entregues

### Código

```
src/etl_pipeline.py
├─ join_run() refatorado (lines 3168–3280)
├─ _generate_join_qc() (lines 3082–3155)
└─ +170 linhas de código novo

src/domains/enriched/post/postjoin_enricher.py (NEW)
├─ enrich_postjoin_run() (orquestrador)
├─ enrich_activity_postjoin()
├─ enrich_cardio_postjoin()
├─ enrich_sleep_postjoin()
├─ Helper functions (_rolling_corr_7d, _ratio, _handle_missing_domains)
└─ CLI integration (330 linhas)

src/domains/enriched/post/__init__.py (UPDATED)
└─ Export enrich_postjoin_run

Makefile (UPDATED)
└─ Nova tarefa: enrich-postjoin (lines 165–174)
```

### Documentação

```
docs/PHASE3_ENRICHED_GLOBAL_ARCHITECTURE.md (16K)
docs/QUICK_REFERENCE_ETL.md (3K)
docs/TECHNICAL_CHANGES_PHASE3.md (12K)
docs/FASE3_STATUS.txt (2K)
```

---

## Estado Atual (P000001 / 2025-11-06)

### Dados Validados

- **Joined**: 201 rows × 53 cols (2018-04-06 → 2025-07-28)
- **Activity Postjoin**: 128 rows × 54 cols
- **Cardio Postjoin**: 69 rows × 54 cols
- **QC Report**: Cobertura verificada (activity 65.67%, cardio 34.33%)

### Colunas Coalesced

```
act_steps = coalesce(apple_steps, zepp_steps)
act_active_min = coalesce(apple_exercise_min, zepp_exercise_min)
hr_mean = coalesce(apple_hr_mean, zepp_hr_mean)
hr_std = coalesce(apple_hr_std, zepp_hr_std)
n_hr = coalesce(apple_n_hr, zepp_n_hr)
```

### Enriquecimentos Cross-Domain

```
Activity:
└─ act_steps_vs_hr_7d_corr (7-day rolling correlation)

Cardio:
├─ hr_mean_vs_act_7d_corr (7-day rolling correlation)
└─ hr_variability_ratio (std / mean)

Sleep:
└─ [structure ready, not processed in test snapshot]
```

---

## Invariantes Mantidas

✅ **datetime64 preservation**: Data mantida como datetime64 internamente até escrita CSV  
✅ **MAX_RECORDS scope**: Afeta seeds + prejoin APENAS; join usa TUDO materializado  
✅ **Vendor/variant preservation**: Colunas originais por vendor mantidas  
✅ **Modularização**: domains.enriched.post.\* segue padrão python -m  
✅ **QC automation**: Cobertura por domínio + flags rastreados

---

## Como Usar (Fase 3)

```bash
# Pré-requisito: ter features materializadas
# (ou executar activity/cardio/sleep seeds primeiro)

# 1. Pre-join enrichment
make enrich-prejoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# 2. Join com coalescência + QC
make join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06

# 3. Postjoin cross-domain enrichment
make enrich-postjoin DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# Validar
cat data/etl/P000001/2025-11-06/qc/join_qc.csv
```

---

## Próxima Etapa: Fase 4 (QC Comparativo)

### Propósito

Validar a qualidade dos enriquecimentos e detectar anomalias/drift.

### Possíveis Enfoques

**Option A: QC Comparativo com Legacy**

- Se houver versão legacy do joined, comparar:
  - Schema (colunas esperadas vs encontradas)
  - Cobertura (% non-null por coluna)
  - Distribuições (mean, std, min, max)
  - Anomalias (outliers, gaps)

**Option B: QC Interno (sem legacy)**

- Validar cobertura por domínio (alerta se < 20%)
- Detectar anomalias:
  - Correlações muito altas (> 0.95)
  - Ratios inválidas (Inf, NaN prevalência)
  - Períodos de missing > N dias
- Gerar relatório de consistência

**Option C: Both**

- Comparar com legacy SE houver
- QC interno também

### Recomendação

**Option B** (QC Interno) é mais robusto e não depende de versão legacy. Pode ser adaptado depois se legacy estiver disponível.

---

## Questões Pendentes Para Fase 4

1. **Existe versão legacy para comparação?**

   - Se sim, qual é o caminho? (`legacy/joined_features_daily.csv`?)

2. **Thresholds de anomalia**

   - Cobertura mínima aceitável? (Padrão: 20%)
   - Correlação máxima (\_corr > 0.95 = anomalia?)
   - Período máximo de missing? (dias)

3. **Output esperado**

   - QC report format? (CSV, JSON, HTML?)
   - Alertas/logs level? (ERROR, WARNING, INFO)

4. **Integração com pipeline**
   - Retornar exit code 1 se anomalias críticas detectadas?
   - Ou apenas reportar (exit 0 sempre)?

---

## Arquitetura Completa Após Fase 3

```
[features/]                     (Fase 1)
  ↓ (make enrich-prejoin)
[enriched/prejoin/]             (Fase 2)
  ↓ (make join)
[joined/] + [qc/join_qc.csv]    (Fase 3)
  ↓ (make enrich-postjoin)
[enriched/postjoin/]            (Fase 3 continuation)
  ↓ (TBD: make qc-validate)
[qc/qc_report.csv]              (Fase 4 - TBD)
  ↓ (optional: make labels)
[labeled/]                      (Fase 5 - optional)
  ↓ (optional: make aggregate)
[aggregated/]                   (Fase 5 - optional)
```

---

## Repositório Status

- **Branch**: release/v4.1.0
- **Last Commit**: Fase 3 Implementation (6 Nov 2025)
- **Tests**: All pass ✅
- **Docs**: Complete ✅

---

## Next Steps

1. **Revisar** questionário pendente acima (seção "Questões Pendentes")
2. **Decidir** abordagem QC (Option A, B, ou C)
3. **Planejar** Fase 4 com base em resposta
4. **Implementar** QC validator module (`domains.qc.qc_validator`)

---

**Status Final**: 🚀 **PRONTO PARA FASE 4**

Toda a lógica de pipeline está em place. Fase 4 é ortogonal (validação, não transformação).

---

**Preparado por**: ETL Development Team  
**Data**: 6 de Novembro de 2025  
**Versão**: ETL v4.1.0 (release/v4.1.0)
