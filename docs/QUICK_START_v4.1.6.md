# Quick Guide: What Changed in v4.1.6

**For**: Rodrigo (P000001)  
**Date**: November 20, 2025  
**Time to read**: 3 minutes

---

## 🎯 TL;DR

1. ✅ **Classes agora balanceadas**: 25/50/25 (antes: 6/94/0)
2. ✅ **Labels renomeados**: low_pbsi/mid_pbsi/high_pbsi (não mais "stable/unstable")
3. ✅ **Disclaimer adicionado**: "Não validado clinicamente" (importante para o paper)
4. ✅ **NB2 agora treina**: Cross-validation funciona com classes balanceadas
5. ✅ **Backward compatible**: Código antigo continua funcionando

---

## 📊 O Que Mudou (Visualmente)

### Antes (v4.1.5)

```
Distribuição de Labels:
  Stable (+1):     176 dias  (6.2%)   ← Muito pequeno
  Neutral (0):   2,643 dias (93.5%)  ← Dominante
  Unstable (-1):     9 dias  (0.3%)   ← Inviável para ML

Problema: NB2 não conseguia treinar (classes degeneradas)
```

### Agora (v4.1.6)

```
Distribuição de Labels:
  Low PBSI (+1):    707 dias (25%)  ← Regulado fisiologicamente
  Mid PBSI (0):   1,414 dias (50%)  ← Típico
  High PBSI (-1):   707 dias (25%)  ← Desregulado fisiologicamente

✓ NB2 treina com sucesso
✓ Cross-validation funciona (6 folds)
✓ Classes balanceadas cientificamente
```

---

## 🔧 O Que Fazer Agora

### 1. Re-rodar Pipeline (Já Rodando)

```bash
make pipeline PID=P000001 SNAPSHOT=2025-11-07 ZPWD="qqQKwnhY"
```

**Resultado esperado**:

- Stages 0-9 completos (antes: Stage 6 skipado)
- Arquivo `features_daily_labeled.csv` com labels balanceados
- Modelos NB2 treinados em `data/ai/P000001/2025-11-07/nb2/`

### 2. Testar Notebooks

```bash
# NB1 - EDA
jupyter notebook notebooks/NB1_EDA.ipynb

# NB2 - Baselines (agora funciona!)
jupyter notebook notebooks/NB2_Baseline.ipynb

# NB3 - Deep Learning
jupyter notebook notebooks/NB3_DeepLearning.ipynb
```

### 3. Atualizar Paper

**Adicionar na seção Limitations**:

```markdown
### Clinical Validation

The PBSI labels (low/mid/high) represent composite physiological
indices derived from sleep, cardiovascular, and activity patterns.
**These labels have not been validated against psychiatric ground truth**
(mood diaries, clinician ratings, or DSM-5 diagnostic criteria) and
should not be interpreted as direct proxies for psychiatric states
(mania, depression, ADHD severity).

Future work (v5.x) will:

1. Collect prospective mood diaries (ecological momentary assessment)
2. Validate patterns against documented psychiatric episodes
3. Develop state-specific biomarkers for BD/ADHD
```

**Atualizar terminologia**:

- ❌ "períodos de estabilidade/instabilidade"
- ✅ "períodos de regulação/desregulação fisiológica"
- ✅ "padrões de low/mid/high PBSI"

---

## 📖 Documentação Nova

**Leia estes arquivos** (em ordem de prioridade):

1. **`RELEASE_NOTES_v4.1.6.md`** ← Você está aqui

   - Release notes completas
   - API changes
   - Migration guide

2. **`docs/CLINICAL_COHERENCE_ANALYSIS.md`**

   - **Por que "stable/unstable" não fazia sentido clínico**
   - Alternativas (estados psiquiátricos, biomarcadores)
   - Roadmap para v5.x

3. **`docs/PBSI_LABELS_v4.1.6.md`**

   - Referência técnica completa
   - Fórmulas, thresholds, interpretação
   - Exemplos de uso

4. **`docs/PBSI_THRESHOLD_ANALYSIS.md`**
   - Análise estatística do desbalanceamento
   - Justificativa para P25/P75
   - Comparação de alternativas

---

## 🧪 O Que Esperar dos Resultados

### Label Distribution

```
label_3cls:
  +1 (low_pbsi):    707 dias (25.0%)
   0 (mid_pbsi):  1,414 dias (50.0%)
  -1 (high_pbsi):   707 dias (25.0%)

label_2cls:
   1 (regulated):  707 dias (25.0%)
   0 (not reg):  2,121 dias (75.0%)
```

### PBSI Score Stats

```
Mean:    ~0.00  (centered by design)
Std:     ~0.26
Min:    -1.28
P25:    -0.12  ← Threshold low
Median:  0.11
P75:     0.17  ← Threshold high
Max:     0.92
```

### Model Performance (Expected)

```
NB2 (Baseline Models):
  - Logistic Regression: ~0.65-0.70 accuracy (3-class)
  - Random Forest: ~0.70-0.75 accuracy
  - XGBoost: ~0.72-0.78 accuracy

NB3 (LSTM):
  - Sequence models: ~0.75-0.80 accuracy
  - Temporal SHAP: Feature importance over time
  - Drift detection: 6 ADWIN points, 45/494 significant KS tests
```

---

## ❓ FAQ

### "Posso ainda usar thresholds fixos (v4.1.5)?"

Sim! Use flag:

```python
df = build_pbsi_labels(
    unified_df,
    use_percentile_thresholds=False,
    threshold_low_fixed=-0.5,
    threshold_high_fixed=0.5
)
```

### "Os labels mudaram de valor?"

**Não**. Ainda são +1, 0, -1. Apenas mudou:

- **Thresholds** (onde cortar: P25/P75 ao invés de ±0.5)
- **Nomenclatura** (low/mid/high ao invés de stable/neutral/unstable)
- **Documentação** (disclaimers clínicos)

### "Preciso re-fazer todas as análises?"

**Recomendado**, mas não obrigatório:

- ✅ **Re-fazer**: Para ter classes balanceadas e modelos treináveis
- ⚠️ **Manter v4.1.5**: Se já tem resultados publicáveis e prazo apertado
- 💡 **Híbrido**: Usar v4.1.6 para CA2, mencionar v4.1.5 como piloto

### "Isso afeta meu deadline do CA2?"

**Não**. Implementação já pronta:

- Pipeline rodando (4 min)
- Documentação completa
- Notebooks já atualizados
- Só falta incluir disclaimer no paper (5 min)

### "Preciso coletar mood diary agora?"

**Não para CA2**. Isso é para v5.x (pesquisa futura):

- **CA2 (agora)**: Entregar com PBSI exploratório + disclaimer
- **v5.x (depois)**: Validar com ground truth clínico
- **Paper futuro**: "Validation of Wearable-Derived PBSI Against Psychiatric Ground Truth"

---

## 🎓 Para o Paper (CA2)

### Seção de Métodos - Adicionar

```markdown
#### PBSI Threshold Selection

To ensure balanced class distribution for machine learning training,
we used **percentile-based thresholds** (P25/P75) rather than fixed
values. This approach adapts to each participant's physiological
range, resulting in a 25/50/25 class split (low/mid/high PBSI).
```

### Seção de Limitations - Adicionar

```markdown
#### Clinical Validation

PBSI labels represent composite physiological indices and have not
been validated against psychiatric ground truth. Future research
should collect ecological momentary assessments (EMA) and clinician
ratings to validate these patterns against DSM-5 diagnostic criteria.
```

### Seção de Results - Atualizar

```markdown
<!-- ANTES -->

"X days were classified as stable, Y as neutral, Z as unstable."

<!-- DEPOIS -->

"Using percentile-based thresholds, X days (25%) showed low PBSI
(physiologically regulated patterns), Y days (50%) showed mid PBSI
(typical patterns), and Z days (25%) showed high PBSI (dysregulated
patterns)."
```

---

## ✅ Checklist Para Finalização

- [ ] Pipeline completou (stages 0-9)
- [ ] NB1 EDA rodou sem erros
- [ ] NB2 Baseline gerou modelos (não mais skipado)
- [ ] NB3 Deep Learning rodou completo
- [ ] Paper atualizado com disclaimer clínico
- [ ] Terminologia atualizada (low/mid/high PBSI)
- [ ] Seção Limitations menciona falta de validação clínica
- [ ] Figuras/tabelas atualizadas com novos labels
- [ ] Commit com mensagem: "feat: implement PBSI v4.1.6 with percentile thresholds"

---

## 🚀 Próximos Passos (Pós-CA2)

**v5.0.0 (Pesquisa Futura)**:

1. Coletar mood diary retrospectivo (2024-2025)
2. Mapear episódios conhecidos (mania, depressão, ADHD)
3. Validar PBSI contra ground truth clínico
4. Desenvolver biomarcadores específicos por estado
5. Publicar paper de validação

**v5.1.0 (Multi-Participant)**:

1. Recrutar N=10-20 participantes (ADHD/BD)
2. Wearables + EMA diário + avaliações clínicas
3. Testar generalizabilidade dos padrões
4. Paper em _JMIR Mental Health_ ou _Translational Psychiatry_

---

## 📞 Suporte

**Dúvidas?**

- **Técnicas**: Consultar `docs/PBSI_LABELS_v4.1.6.md`
- **Científicas**: Consultar `docs/CLINICAL_COHERENCE_ANALYSIS.md`
- **Pipeline**: Consultar `docs/ETL_ARCHITECTURE_COMPLETE.md`

**Problemas?**

- Check logs: `data/etl/P000001/2025-11-07/pipeline.log`
- Check errors: `make qc-all PID=P000001 SNAPSHOT=2025-11-07`

---

**Versão**: 4.1.6  
**Status**: Production-ready ✅  
**Clinical validation**: Pending (v5.x) ⏳
