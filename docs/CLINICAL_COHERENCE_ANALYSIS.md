# Análise de Coerência Clínica: PBSI vs ADHD/BD

**Data**: 20 de Novembro de 2025  
**Crítica**: Os rótulos "stable/neutral/unstable" não fazem sentido clínico para ADHD/BD

---

## ❌ O Problema: Desconexão Clínica Fundamental

### 1. PBSI Atual: Conceito de "Estabilidade Fisiológica"

O **PBSI (Physio-Behavioral Stability Index)** foi projetado como:

```python
pbsi_score = 0.40 * sleep_sub + 0.35 * cardio_sub + 0.25 * activity_sub

# Labels:
# +1 (stable):   pbsi ≤ -0.5  → "Fisiologicamente estável"
# 0 (neutral):   -0.5 < pbsi < 0.5  → "Normal"
# -1 (unstable): pbsi ≥ 0.5  → "Fisiologicamente instável"
```

**Interpretação implícita**:

- "Stable" = Muito sono + HRV alta + Muita atividade
- "Unstable" = Pouco sono + HRV baixa + Pouca atividade

### 2. Realidade Clínica de ADHD/BD: Estados Psiquiátricos Distintos

ADHD e Transtorno Bipolar **não são um espectro de "estabilidade"** - são **estados clínicos qualitativamente diferentes**:

| Estado Clínico           | Características Clínicas                         | Biomarcadores Esperados                                  |
| ------------------------ | ------------------------------------------------ | -------------------------------------------------------- |
| **Eutimia (baseline)**   | Humor estável, funcionalidade preservada         | Sono regular, HRV normal, atividade moderada             |
| **Mania/Hipomania (BD)** | ↑ Energia, ↓ Necessidade de sono, hiperatividade | **Sono reduzido (~4-6h)**, atividade noturna ↑, HR ↑     |
| **Depressão (BD/MDD)**   | ↓ Energia, ↓ Motivação, fadiga                   | **Sono excessivo ou insônia**, HRV ↓, sedentarismo ↑     |
| **ADHD desregulado**     | Hiperatividade, desatenção, impulsividade        | **Variabilidade alta** em sono/atividade, fragmentação ↑ |
| **ADHD compensado**      | Sintomas controlados (medicação/estratégias)     | Padrões mais regulares                                   |

### 3. Por Que PBSI "Stable/Unstable" Falha Clinicamente?

#### Problema 1: "Estabilidade" ≠ Saúde Mental

**Exemplo contraditório**:

- **Mania aguda**: Pessoa dorme 4h, está hiperativa, FC alta → PBSI marca como "unstable" ✓
- **Depressão severa**: Pessoa dorme 12h, sedentária, FC baixa → PBSI marca como "stable" ❌

**↑ Depressão pode parecer "estável" porque tem sono longo e pouca atividade!**

#### Problema 2: ADHD Não É Um Pólo de Instabilidade

ADHD **não é o oposto de "estabilidade"** - é um transtorno neurodevelopmental com características específicas:

- **ADHD sintomático**: Alta variabilidade (não necessariamente "instável")
- **ADHD tratado**: Pode ter métricas "normais" mas ainda ter ADHD

#### Problema 3: Perda de Informação Clínica

O PBSI **homogeneíza estados qualitativamente diferentes**:

```
Mania (sono ↓, atividade ↑) → pbsi = +0.6 → "unstable"
Depressão (sono ↑, atividade ↓) → pbsi = +0.4 → "neutral"
ADHD (variabilidade ↑) → pbsi = ??? → ???
```

**Não é possível distinguir mania de depressão de ADHD!**

---

## ✅ O Que Deveria Ser Modelado?

### Opção A: Classificação de Estados Psiquiátricos (RECOMENDADO)

**Objetivo**: Predizer qual estado clínico o participante está vivenciando.

#### Labels Clinicamente Coerentes

```python
# 5 classes (multi-class)
labels = {
    "EUTHYMIC": 0,        # Baseline / eutimia
    "MANIC": 1,           # Mania/hipomania (BD)
    "DEPRESSIVE": 2,      # Depressão (BD ou MDD)
    "ADHD_SYMPTOMATIC": 3,  # ADHD descompensado
    "MIXED": 4,           # Estado misto (raro)
}
```

**Vantagens**:

- ✅ Alinhado com diagnósticos psiquiátricos (DSM-5/ICD-11)
- ✅ Interpretação clínica clara
- ✅ Permite validação com mood diaries / registros clínicos
- ✅ Generalizável para outros participantes

**Desafios**:

- ❌ Requer **ground truth** (mood diaries, registros médicos)
- ❌ Mais complexo (5 classes vs 3)
- ❌ Pode ter classe "UNKNOWN" para períodos sem documentação

#### Labels Binários Simplificados

```python
# 2 classes (mais viável para N-of-1)
labels = {
    "BASELINE": 0,     # Eutimia / funcionamento normal
    "SYMPTOMATIC": 1,  # Qualquer estado sintomático (mania/depressão/ADHD)
}
```

**Vantagens**:

- ✅ Mais simples de validar (pergunta: "estava bem ou mal?")
- ✅ Balanceamento mais fácil
- ✅ Ainda tem utilidade clínica (detecção de piora)

### Opção B: Biomarcadores Específicos por Condição

Ao invés de um índice único, modelar **sintomas específicos**:

```python
predictions = {
    "sleep_disturbance": 0-1,      # Sono perturbado? (comum em mania/depressão)
    "activity_irregularity": 0-1,  # Atividade irregular? (marcador ADHD)
    "autonomic_dysreg": 0-1,       # HRV alterada? (estresse/ansiedade)
    "circadian_misalignment": 0-1, # Ritmo circadiano desalinhado? (mania)
}
```

**Vantagens**:

- ✅ Não assume relação linear entre condições
- ✅ Captura nuances (ex: ADHD + depressão comórbida)
- ✅ Interpretação granular

**Desafios**:

- ❌ Requer múltiplos modelos
- ❌ Mais complexo de integrar

### Opção C: Predição de Severidade por Dimensão

Usar escalas clínicas como target:

```python
targets = {
    "manic_symptoms_severity": 0-10,   # YMRS (Young Mania Rating Scale)
    "depressive_symptoms_severity": 0-10,  # MADRS (Montgomery-Åsberg)
    "adhd_symptoms_severity": 0-10,    # ASRS (Adult ADHD Self-Report)
}
```

**Vantagens**:

- ✅ Alinhado com instrumentos clínicos validados
- ✅ Permite análise dimensional (não categórica)
- ✅ Útil para monitoramento longitudinal

**Desafios**:

- ❌ Requer coleta prospectiva de escalas
- ❌ Mais trabalhoso (múltiplas escalas)

---

## 🔬 Como Obter Ground Truth Clinicamente Válido?

### Retrospectivo (Viável Agora)

1. **Mood Diaries Retrospectivos**:

   ```
   Data: 2024-03-15
   Humor: 3/10 (muito deprimido)
   Sono: 12h (hipersonia)
   Atividade: Mal saí da cama
   → Label: DEPRESSIVE
   ```

2. **Registros Médicos**:

   - Consultas psiquiátricas com documentação de estado
   - Prescrições (ajustes de medicação = mudança de estado?)
   - Internações (episódios agudos documentados)

3. **Auto-relatos Estruturados**:
   - "Em março de 2024 eu estava em depressão severa"
   - "Em junho de 2024 tive um episódio hipomaníaco"
   - "Em setembro voltei ao normal"

### Prospectivo (Para Estudos Futuros)

1. **Daily Mood Tracking**:

   - App com questionário diário (2-3 min)
   - Escalas validadas (PHQ-2 para depressão, MDQ para mania, ASRS-6 para ADHD)

2. **Ecological Momentary Assessment (EMA)**:

   - 3-5 prompts/dia perguntando humor/energia/concentração
   - Captura variabilidade intra-dia

3. **Clinician Ratings**:
   - Avaliações semanais/mensais com psiquiatra
   - Uso de escalas padronizadas (YMRS, MADRS, ASRS)

---

## 📊 Comparação: PBSI Atual vs Alternativas Clínicas

| Aspecto                           | PBSI "Stable/Unstable"   | Estados Psiquiátricos   | Biomarcadores Específicos |
| --------------------------------- | ------------------------ | ----------------------- | ------------------------- |
| **Validade clínica**              | ❌ Baixa (conceito vago) | ✅ Alta (DSM-5 aligned) | ✅ Média-alta             |
| **Interpretabilidade**            | ⚠️ Ambígua               | ✅ Clara                | ✅ Granular               |
| **Requer ground truth**           | ❌ Não                   | ✅ Sim                  | ✅ Sim                    |
| **Balanceamento de classes**      | ❌ Extremo (93% neutral) | ⚠️ Depende de dados     | ⚠️ Variável               |
| **Generalizável para outros N=1** | ⚠️ Limitado              | ✅ Sim                  | ✅ Sim                    |
| **Publicável cientificamente**    | ❌ Difícil de defender   | ✅ Robusto              | ✅ Robusto                |

---

## 🎯 Recomendação Final

### Curto Prazo (CA2 Deliverable)

**OPÇÃO 1: Manter PBSI mas renomear para refletir significado real**

Trocar:

- ~~"Stable/Neutral/Unstable"~~ (termos vagos)
- **"Low PBSI / Medium PBSI / High PBSI"** (descritivo, neutro)

Ou melhor ainda:

- **"Physiologically Regulated / Typical / Dysregulated"** (mais preciso)

Justificativa no paper:

> "We computed a composite Physio-Behavioral Stability Index (PBSI) as an exploratory proxy for physiological regulation. **We acknowledge this index does not map directly to psychiatric diagnostic categories** (mania, depression, ADHD states), but rather captures variance in sleep, cardiovascular, and activity patterns. Future work should validate these patterns against clinical ground truth (mood diaries, clinician ratings)."

**+ Análise de Balanceamento com Thresholds Percentis (P25/P75)**

Como já analisado, ajustar thresholds para permitir modelagem.

### Médio Prazo (Pesquisa Contínua)

**OPÇÃO 2: Coletar Ground Truth e Re-rotular Dados**

1. Criar mood diary retrospectivo guiado:

   - "Em quais meses de 2024 você estava deprimido?"
   - "Houve períodos de hipomania? Quando?"
   - "ADHD estava mais difícil de controlar em algum período?"

2. Mapear para períodos:

   ```python
   labels = {
       "2024-01-01:2024-02-28": "DEPRESSIVE",
       "2024-03-01:2024-05-15": "EUTHYMIC",
       "2024-05-16:2024-06-30": "MANIC",
       # ...
   }
   ```

3. Re-treinar modelos com labels clínicos verdadeiros

4. **Publicar validação**: "From Wearable Data to Psychiatric States: A Ground-Truth Validated N-of-1 Study"

### Longo Prazo (Ciência de Alto Impacto)

**OPÇÃO 3: Estudo Prospectivo Multi-Participante**

1. Recrutar N=10-20 participantes com ADHD/BD
2. Wearables + EMA diário + avaliações clínicas mensais
3. Validar biomarcadores específicos (HRV, sleep variability, circadian misalignment)
4. Publicar em _JMIR Mental Health_ ou _Translational Psychiatry_

---

## 🔍 Análise do Seu Caso Específico (P000001)

### Informações do README_research_plan.md

```
Participant: N-of-1 study, ADHD + BD diagnosis
Data: 8 years of wearable data (2017-2025)
Devices: Apple Watch + Zepp (GTR 2/4) + Oura Ring
```

### O Que Sabemos Clinicamente?

Você tem diagnóstico de **ADHD + Transtorno Bipolar**. Portanto:

1. **ADHD é condição de base** (não varia, mas sintomas podem variar)
2. **BD produz episódios** (mania, depressão, eutimia)
3. **Objetivo realista**: Detectar quando está em:
   - Eutimia (baseline funcional)
   - Episódio depressivo
   - Episódio (hipo)maníaco
   - Estado misto (raro)

### Pergunta Científica Correta

❌ **Pergunta errada**: "Quando estou 'estável' vs 'instável'?"  
✅ **Pergunta certa**: "Quando estou em eutimia vs episódio (depressão/mania)?"

Ou ainda melhor:
✅ **"Posso predizer um episódio X dias antes baseado em biomarcadores?"** (early warning system)

---

## 💡 Ação Imediata Recomendada

### Opção A: Rename + Reframe (Mínimo Viável)

1. Mudar labels de `stable/neutral/unstable` para `low_pbsi/mid_pbsi/high_pbsi`
2. Ajustar thresholds para P25/P75 (balanceamento)
3. No paper: deixar claro que PBSI é **exploratório**, não validado clinicamente
4. **Sugerir como limitação**: "Future work should validate against psychiatric ground truth"

### Opção B: Coletar Ground Truth Retrospectivo (1 Semana Extra)

1. Criar spreadsheet com períodos conhecidos:

   ```
   Data Início | Data Fim | Estado
   2024-01-01  | 2024-02-15 | Depressão moderada
   2024-02-16  | 2024-05-30 | Eutimia
   2024-06-01  | 2024-07-15 | Hipomania
   ```

2. Re-rotular dataset com labels clínicos verdadeiros
3. Treinar modelo com **predição de estado psiquiátrico**
4. **Paper muito mais forte cientificamente**

### Opção C: Ambas (Recomendado)

1. **Curto prazo**: Rename + P25/P75 (entrega CA2 no prazo)
2. **Médio prazo**: Coletar ground truth + re-análise (paper futuro)

---

## 📝 Mudanças Necessárias no Código

### 1. Renomear Labels em `build_pbsi.py`

```python
# ANTES (clinicamente vago):
result['label_3cls'] = 1 if pbsi_score <= -0.5 else (-1 if pbsi_score >= 0.5 else 0)
# Interpretação: 1="stable", 0="neutral", -1="unstable"

# DEPOIS (descritivo):
result['label_3cls'] = 1 if pbsi_score <= threshold_low else (
    -1 if pbsi_score >= threshold_high else 0
)
# Interpretação: 1="low_pbsi" (regulado), 0="mid_pbsi", -1="high_pbsi" (desregulado)
```

### 2. Adicionar Documentação Clínica

```python
"""
PBSI Labels (Exploratory - Not Clinically Validated):
    +1 (low_pbsi):  Physiologically regulated (good sleep, high HRV, active)
    0 (mid_pbsi):   Typical physiological patterns
    -1 (high_pbsi): Physiologically dysregulated (poor sleep, low HRV, sedentary)

⚠️ IMPORTANT:
These labels do NOT map directly to psychiatric states (mania, depression, ADHD).
They are composite physiological indices requiring clinical validation.

For clinical interpretation, consult with psychiatrist and cross-reference with:
- Mood diaries
- Medication changes
- Life events
- Clinical assessments
"""
```

### 3. Adicionar Flag "Clinically Validated"

```python
df['has_clinical_ground_truth'] = False  # Default: sem validação
# Quando tiver mood diary:
df.loc[df['date'].between('2024-01-01', '2024-02-15'), 'clinical_state'] = 'DEPRESSIVE'
df.loc[df['clinical_state'].notna(), 'has_clinical_ground_truth'] = True
```

---

## 🎓 Impacto no Paper (CA2)

### Seção de Limitations (Adicionar)

> **Clinical Validation**: The PBSI labels used in this study are composite physiological indices and have not been validated against psychiatric ground truth (clinician ratings, mood diaries, or diagnostic interviews). While they capture variance in sleep, cardiovascular, and activity patterns, **they should not be interpreted as direct proxies for psychiatric states** (e.g., mania, depression, or ADHD symptom severity). Future research should:
>
> 1. Collect prospective mood diaries and clinical assessments
> 2. Validate physiological patterns against DSM-5 diagnostic criteria
> 3. Explore state-specific biomarkers (e.g., nocturnal activity in mania, sleep irregularity in ADHD)

### Seção de Future Work

> **Ground-Truth Validation**: A critical next step is collecting ecological momentary assessments (EMA) and clinician ratings to map wearable-derived patterns to psychiatric states. This would enable:
>
> - Early warning systems for mood episodes
> - Personalized symptom tracking
> - Medication response monitoring

---

## ✅ Resumo Executivo

| Questão                                              | Resposta                                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------- |
| **PBSI "stable/unstable" faz sentido para ADHD/BD?** | ❌ **NÃO**. São conceitos vagos que não mapeiam para estados psiquiátricos.      |
| **O que deveria ser modelado?**                      | ✅ Estados clínicos (eutimia, mania, depressão) ou biomarcadores específicos.    |
| **Precisa descartar todo o trabalho?**               | ❌ Não! Pode renomear labels e deixar claro que é exploratório.                  |
| **Como melhorar cientificamente?**                   | ✅ Coletar ground truth (mood diaries retrospectivos) e re-rotular.              |
| **O que fazer AGORA para CA2?**                      | ✅ **Opção A**: Rename + P25/P75 + disclaimers no paper.                         |
| **Isso é um problema grave?**                        | ⚠️ Médio. Não invalida o trabalho técnico, mas **limita interpretação clínica**. |

---

**Status**: Requer decisão sobre estratégia (Opção A, B ou C)  
**Próximo passo**: Escolher abordagem e implementar mudanças
