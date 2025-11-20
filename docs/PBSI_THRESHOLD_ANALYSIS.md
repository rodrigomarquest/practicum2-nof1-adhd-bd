# PBSI Threshold Analysis & Class Balancing

**Date**: November 20, 2025  
**Pipeline**: v4.1.5  
**Participant**: P000001  
**Snapshot**: 2025-11-07

## 1. Problema Identificado

### Distribuição Atual (Thresholds Fixos)
```
Stable   (+1): pbsi ≤ -0.5  →  176 dias (6.2%)
Neutral  (0):  -0.5 < pbsi < 0.5  → 2,643 dias (93.5%)
Unstable (-1): pbsi ≥ 0.5  →    9 dias (0.3%)
```

**Problema**: Desbalanceamento extremo (93.5% neutral) torna o dataset inadequado para modelagem:
- Cross-validation falha (folds com apenas 1 classe)
- Modelos não conseguem aprender padrões minoritários
- Não há variabilidade suficiente para treinar classificadores

## 2. Análise Estatística do PBSI Score

### Estatísticas Descritivas
```
Mean:    0.005  (praticamente zero)
Std:     0.260
Min:    -1.277
Q25:    -0.117
Median:  0.109
Q75:     0.172
Max:     0.917
```

### Observações Científicas

1. **Distribuição Normal**: O PBSI score tem média ~0 e desvio padrão 0.26, sugerindo distribuição aproximadamente normal.

2. **Thresholds Atuais São Extremos**: 
   - Threshold de -0.5 está abaixo do 5º percentil (P05 = -0.542)
   - Threshold de +0.5 está acima do 95º percentil (P95 = 0.250)
   - Resultado: apenas os casos mais extremos são classificados como stable/unstable

3. **Problema Metodológico**: Os thresholds fixos de ±0.5 não consideram a distribuição real dos dados do participante.

## 3. Avaliação Científica da Lógica Atual

### ✅ Pontos Positivos

1. **Z-score Segmentado**: Normalização por segmento previne data leakage ✓
2. **Pesos Justificados**: 40% sleep + 35% cardio + 25% activity tem lógica clínica ✓
3. **Convenção de Sinais**: Lower PBSI = more stable é consistente ✓
4. **Quality Score**: Penalização por dados faltantes é adequada ✓

### ❌ Problemas Científicos

1. **Thresholds Fixos Sem Justificativa Empírica**:
   - Por que ±0.5? Não há referência a estudos ou validação clínica
   - Não considera variabilidade individual do participante
   - Não é adaptativo à distribuição real dos dados

2. **Falta de Validação Clínica**:
   - Não há correspondência com eventos clínicos documentados
   - Não há validação com mood diaries ou avaliações clínicas
   - Labels são puramente algorítmicos

3. **Desbalanceamento Não Foi Previsto**:
   - Pipeline assume que teremos classes balanceadas
   - Não há mecanismo para ajustar thresholds automaticamente

## 4. Recomendações de Melhorias

### Opção A: Thresholds Baseados em Percentis (RECOMENDADA)

**Racional**: Garantir distribuição controlada de classes, adaptada aos dados do participante.

#### A1. Balanceamento 25/50/25 (Conservador)
```python
p25 = pbsi.quantile(0.25)  # -0.117
p75 = pbsi.quantile(0.75)  # +0.172

Stable:   pbsi ≤ -0.117  (25% dos dias)
Neutral:  -0.117 < pbsi < 0.172  (50% dos dias)
Unstable: pbsi ≥ 0.172  (25% dos dias)
```

**Vantagens**:
- Classes minoritárias têm 25% dos dados cada (n=707 dias) → suficiente para CV
- Classe majoritária (neutral) ainda domina com 50%
- Reflete distribuição natural dos dados
- Interpretação clara: "25% mais estáveis" vs "25% menos estáveis"

**Desvantagens**:
- Pode incluir dias "normais" nas classes extremas
- Perde interpretação absoluta dos thresholds

#### A2. Balanceamento 20/60/20 (Moderado)
```python
p20 = pbsi.quantile(0.20)  # -0.197
p80 = pbsi.quantile(0.80)  # +0.186

Stable:   pbsi ≤ -0.197  (20% dos dias)
Neutral:  -0.197 < pbsi < 0.186  (60% dos dias)
Unstable: pbsi ≥ 0.186  (20% dos dias)
```

**Vantagens**:
- Classes minoritárias ainda robustas (n=566 dias cada)
- Classe neutral mais ampla (mais conservador)
- Menos risco de "false positives" nas classes extremas

#### A3. Balanceamento 33/33/33 (Mais Balanceado)
```python
p33 = pbsi.quantile(0.33)  # +0.008
p67 = pbsi.quantile(0.67)  # +0.149

Stable:   pbsi ≤ 0.008  (33% dos dias)
Neutral:  0.008 < pbsi < 0.149  (33% dos dias)
Unstable: pbsi ≥ 0.149  (33% dos dias)
```

**Vantagens**:
- Máximo balanceamento (ideal para ML)
- Cada classe tem n=943 dias → excelente para CV
- Simplifica interpretação: "terço inferior/médio/superior"

**Desvantagens**:
- Perde noção de "extremos" vs "normal"
- Pode ser muito agressivo clinicamente

### Opção B: Thresholds Adaptativos com Validação Clínica

**Racional**: Combinar análise estatística com eventos clínicos documentados.

1. **Identificar Períodos Conhecidos**:
   - Documentar períodos de estabilidade clínica conhecida
   - Documentar períodos de instabilidade (episódios, crises)
   - Calcular PBSI médio para cada período

2. **Ajustar Thresholds Empiricamente**:
   ```python
   # Exemplo hipotético
   stable_periods_mean = -0.15
   unstable_periods_mean = +0.20
   
   threshold_stable = stable_periods_mean + 0.5 * std
   threshold_unstable = unstable_periods_mean - 0.5 * std
   ```

3. **Validar com Mood Diaries**:
   - Correlacionar PBSI com auto-relatos de humor
   - Ajustar thresholds para maximizar concordância

**Vantagens**:
- Cientificamente mais robusto (validação clínica)
- Interpretação clínica clara
- Generalizável para outros participantes

**Desvantagens**:
- Requer dados clínicos adicionais
- Mais trabalhoso de implementar

### Opção C: Classe Binária Simplificada

**Racional**: Para alguns estudos, predição binária é suficiente.

```python
# Usar apenas label_2cls (já implementado)
p50 = pbsi.median()  # 0.109

Stable (1):     pbsi ≤ 0.109  (50% dos dias)
Not Stable (0): pbsi > 0.109  (50% dos dias)
```

**Vantagens**:
- Balanceamento perfeito 50/50
- Simplifica modelagem e interpretação
- Reduz complexidade computacional

**Desvantagens**:
- Perde granularidade (não distingue neutral vs unstable)
- Menos informativo clinicamente

## 5. Implementação Recomendada

### Prioridade 1: Adicionar Parâmetro Configurável (CURTO PRAZO)

Modificar `build_pbsi.py` para aceitar thresholds como parâmetros:

```python
def build_pbsi_labels(
    unified_df: pd.DataFrame,
    threshold_stable: float = -0.117,   # P25 por padrão
    threshold_unstable: float = 0.172,  # P75 por padrão
    use_percentile: bool = True,        # Auto-ajustar por percentil
    ...
) -> pd.DataFrame:
    
    if use_percentile:
        pbsi_scores = df['pbsi_score'].dropna()
        threshold_stable = pbsi_scores.quantile(0.25)
        threshold_unstable = pbsi_scores.quantile(0.75)
        logger.info(f"Auto-adjusted thresholds: stable≤{threshold_stable:.3f}, unstable≥{threshold_unstable:.3f}")
    
    result['label_3cls'] = 1 if pbsi_score <= threshold_stable else (
        -1 if pbsi_score >= threshold_unstable else 0
    )
```

### Prioridade 2: Adicionar ao `config/label_rules.yaml` (MÉDIO PRAZO)

```yaml
pbsi_thresholds:
  method: "percentile"  # ou "fixed"
  stable_percentile: 0.25
  unstable_percentile: 0.75
  # Fallback para método fixed
  stable_fixed: -0.5
  unstable_fixed: 0.5
```

### Prioridade 3: Validação Clínica (LONGO PRAZO)

1. Coletar mood diaries retrospectivos
2. Documentar eventos clínicos conhecidos
3. Correlacionar com PBSI scores
4. Ajustar thresholds com base em ROC/AUC
5. Publicar validação em paper

## 6. Comparação de Métodos

| Método | Stable | Neutral | Unstable | CV Viável? | Interpretação Clínica | Complexidade |
|--------|--------|---------|----------|------------|----------------------|--------------|
| **Atual (±0.5)** | 6.2% | 93.5% | 0.3% | ❌ | ⚠️ Extremos | Baixa |
| **P25/P75** | 25% | 50% | 25% | ✅ | ✅ Quartis | Baixa |
| **P20/P80** | 20% | 60% | 20% | ✅ | ✅ Quintis | Baixa |
| **P33/P67** | 33% | 33% | 33% | ✅✅ | ⚠️ Terços | Baixa |
| **Validado** | ? | ? | ? | ✅ | ✅✅ Eventos | Alta |
| **Binário (P50)** | 50% | - | 50% | ✅✅ | ⚠️ Simples | Baixa |

## 7. Recomendação Final

**Para produção imediata (v4.1.6)**:
1. Implementar **Opção A1 (P25/P75)** como padrão
2. Manter thresholds fixos como opção alternativa
3. Documentar escolha no paper

**Para pesquisa futura**:
1. Coletar dados clínicos prospectivos
2. Validar thresholds com **Opção B**
3. Publicar estudo de validação separado

**Justificativa**:
- P25/P75 é cientificamente defensável (quartis)
- Resolve problema imediato de desbalanceamento
- Permite modelagem e publicação do estudo atual
- Não requer dados adicionais
- É reversível se validação futura sugerir outros valores

## 8. Impacto na Pipeline

### Arquivos a Modificar
1. `src/labels/build_pbsi.py` - lógica de thresholds
2. `config/label_rules.yaml` - configuração
3. `scripts/run_full_pipeline.py` - passar parâmetros
4. `docs/ETL_ARCHITECTURE_COMPLETE.md` - documentação

### Compatibilidade
- Snapshots antigos: manter thresholds fixos
- Novos snapshots: usar thresholds adaptativos
- Flag `--use-fixed-thresholds` para reproduzibilidade

### Testes Necessários
1. Verificar que P25/P75 gera classes balanceadas
2. Testar CV com novos thresholds
3. Verificar que NB2/NB3 treinam com sucesso
4. Comparar métricas com thresholds antigos

---

**Status**: Requer decisão e implementação  
**Autor**: GitHub Copilot (análise automatizada)  
**Revisão**: Pendente aprovação do pesquisador principal
