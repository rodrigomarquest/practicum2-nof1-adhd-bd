# 📊 ANÁLISE PROFUNDA: Readequação ETL para ADHD/BD com Validade Clínica

**Data:** 2025-11-07  
**Participante:** P000001  
**Objetivo:** Avaliar dados disponíveis e alinhá-los com literatura acadêmica sobre biomarcadores para ADHD e Transtorno Bipolar

---

## 1. DADOS ATUALMENTE DISPONÍVEIS

### 1.1 Apple Health (via inapp export.xml)

#### Tipos de dados encontrados:

- **Heart Rate:** 204,930 registros ✅ (com timestamps intra-diários)
- **Heart Rate Variability (SDNN):** 19 registros ✅ (CRÍTICO - sub-utilizado!)
- **Body Mass:** 749 registros
- **BMI:** 746 registros
- **Dietary Water:** 8 registros
- **Height:** 8 registros

#### Features atualmente extraídas (daily aggregation):

```
apple_cardio/inapp/features_daily.csv:
  - date
  - apple_hr_mean ✅ (média diária de BPM)
  - apple_hr_max  ✅ (máx diária)
  - apple_n_hr    ✅ (contagem de amostras)

apple_activity/inapp/features_daily.csv:
  - apple_steps ✅
  - apple_distance_m ✅
  - apple_active_kcal ✅
  - apple_exercise_min ✅
  - apple_stand_hours ✅
  - apple_move_goal_kcal (meta, não resultado)
  - apple_exercise_goal_min (meta)
  - apple_stand_goal_hours (meta)
```

**PROBLEMA:** HRV (Heart Rate Variability) NÃO é agregada nos features diários!

- Existe no XML (19 registros = SDNN em ms)
- Está no código como `apple_hrv_sdnn.csv` per-metric
- MAS não é incluída nas features_daily.csv finais

### 1.2 Zepp Cloud (via smartwatch/ring data)

#### Features atualmente extraídas (daily aggregation):

```
zepp_cardio/cloud/features_daily.csv (157 dias):
  - zepp_hr_mean ✅
  - zepp_hr_max ✅
  - zepp_n_hr ✅

zepp_activity/cloud/features_daily.csv (500 dias):
  - zepp_steps ✅
  - zepp_distance_m ✅
  - zepp_active_kcal ✅
  - zepp_act_cal_total ✅
  - zepp_exercise_min ✅
  - zepp_act_sedentary_min ✅ (IMPORTANTE!)
  - zepp_stand_hours ✅
  - zepp_act_sport_sessions ✅
  - zepp_act_score_daily ✅ (score do device)

zepp_sleep/cloud/features_daily.csv (252 dias):
  - zepp_slp_total_h ✅
  - zepp_slp_deep_h ✅
  - zepp_slp_light_h ✅
  - zepp_slp_rem_h ✅

zepp_health/cloud/features_daily.csv (SE DISPONÍVEL):
  - zepp_spo2_mean (SpO2 = oxigenação sanguínea)
  - zepp_temp_mean (Temp corporal)
  - zepp_stress_mean (score de stress do Zepp)
```

**OBSERVAÇÃO:** Zepp health data pode estar disponível!

---

## 2. LITERATURA ACADÊMICA: Biomarcadores para ADHD & BD

### 2.1 Heart Rate Variability (HRV) - **CRÍTICO**

#### Para ADHD:

- **Redução de HRV em repouso** é biomarcador bem-estabelecido
- Indica **disfunção autonômica** (predominância simpática)
- SDNN (Standard Deviation of NN intervals) < 50ms é anormal
- HRV baixa correlaciona com sintomas de impulsividade

**Referências:**

- Börger et al. (2021): "Heart rate variability in adults with ADHD"
- Thome et al. (2012): Reduced HRV in ADHD - systematic review

#### Para Transtorno Bipolar:

- **HRV alterada em episódios depressivos/maníacos**
- Redução de HRV prediz transição de humor
- Pode servir como biomarcador de estabilidade
- Aumento de LF/HF ratio em episódios maníacos

**Referências:**

- Lown et al. (2015): HRV alterada em BD durante episódios de humor
- Quintana et al. (2012): Reduced HRV in depression

### 2.2 Sleep Architecture - **MUY IMPORTANTE**

#### Para ADHD:

- Latência de sono aumentada (demora para pegar no sono)
- Fragmentação do sono aumentada
- Redução de REM latency
- **Duração total reduzida vs. objetivo**

#### Para Transtorno Bipolar:

- **Necessidade de sono REDUZIDA** durante maníacos (dorme 3h, sente-se descansado)
- Insônia no início/meio da noite em depressão
- REM latency curta (< 60 min) é biomarcador de depressão bipolar
- Duração de REM anormalmente longa

**Disponível:** ✅ Zepp sleep stages (deep, light, REM)

### 2.3 Activity & Sedentariness - **IMPORTANTE**

#### Para ADHD:

- **Hiperatividade:** Passos/movimento aumentados vs. controles
- Variabilidade ALTA de atividade (picos e vales)
- Dificuldade de manter ritmo consistente

#### Para Transtorno Bipolar:

- **Redução de atividade em depressão** (sedentário aumentado)
- **Aumento de atividade em episódios maníacos** (passos 2-3x maiores)
- Ritmo circadiano alterado (picos de atividade nos horários "errados")

**Disponível:** ✅ Steps, active minutes, sedentary minutes

### 2.4 Heart Rate (HR) Baseline - **MODERADO**

#### Para ADHD:

- Pode haver taquicardia baseline (HR > 85-90 bpm repouso)
- Menos específico que HRV, mas complementar

#### Para Transtorno Bipolar:

- Taquicardia em maníaco
- Bradicardia relativa em depressivo

**Disponível:** ✅ HR mean, HR max (mas falta HR em repouso específico)

### 2.5 SpO2 (Blood Oxygen) - **MODERADO**

- Pode indicar padrão respiratório irregular (stress/ansiedade)
- Mais relevante para comorbidades respiratórias
- Zepp pode ter dados

### 2.6 Stress Score (Zepp) - **POTENCIAL**

- Score proprietário do Zepp (baseado em HRV + HR)
- Útil como proxy agregado se confiável

---

## 3. GAPS CRÍTICOS NO ETL ATUAL

### 🔴 CRÍTICO - HRV não agregado

- **Impacto:** Perde o biomarcador mais importante para ADHD/BD
- **Solução:** Calcular SDNN diário agregado do arquivo `apple_hrv_sdnn.csv`
- **Métricas recomendadas:**
  - `apple_hrv_sdnn_mean` (média de SDNN do dia)
  - `apple_hrv_sdnn_std` (variância da variabilidade - meta-variabilidade!)
  - `apple_hrv_sdnn_min` (valor mínimo - indica piora)
  - `apple_hrv_sdnn_max` (valor máximo)

### 🟡 IMPORTANTE - Sleep stage durations não estão normalizadas

- Atual: duração absoluta (horas)
- **Recomendado:** Adicionar percentuais (deep%, light%, rem%)
- Cálculo: `deep_h / total_h * 100` → sleep_deep_pct

### 🟡 IMPORTANTE - Falta dados de repouso específicos

- HR média global vs. HR repouso noturno específico
- Recomendação: extrair HR durante sono (proxy de repouso)
- Nome: `apple_hr_nocturnal_mean`, `apple_hr_nocturnal_min`

### 🟡 IMPORTANTE - Falta variação intra-dia

- Atual: apenas média/max
- **Recomendado:** Adicionar coefficient of variation (CV) de HR
  - `apple_hr_cv = std(HR) / mean(HR)` → indica estabilidade autonômica
  - Menor CV = mais estável (melhor)

### 🟡 IMPORTANTE - Activity variability não capturada

- Atual: apenas steps totais/exercício
- **Recomendado:** Variância de passos (dentro do dia)
  - Dividir dia em blocos de 2-4h, calcular std de passos
  - Nome: `apple_activity_var` ou `zepp_activity_var`

### 🟢 BÔNUS - Zepp stress score

- **Se disponível:** Incluir `zepp_stress_mean` (proxy HRV agregado)

### 🟢 BÔNUS - SpO2 (oxigenação)

- **Se disponível:** Incluir `zepp_spo2_mean`

---

## 4. BIOMARCADORES RECOMENDADOS (POR ORDEM DE PRIORIDADE)

### Tier 1 - CRÍTICO para ADHD/BD:

1. ✅ **HRV SDNN (diário)** - Agregar

   - `apple_hrv_sdnn_mean`
   - `apple_hrv_sdnn_std`
   - `apple_hrv_sdnn_min`

2. ✅ **Sleep stages (percentuais)** - Normalizar

   - `sleep_deep_pct` (deep / total)
   - `sleep_rem_pct` (rem / total)
   - `sleep_light_pct` (light / total)

3. ✅ **HR variabilidade intra-dia** - Calcular

   - `apple_hr_cv` (std / mean)
   - `apple_hr_nocturnal_mean` (HR durante sono)

4. ✅ **Activity variabilidade** - Calcular
   - `apple_activity_variance` ou `zepp_activity_variance`

### Tier 2 - COMPLEMENTAR:

5. 🟡 **Activity ritmo circadiano** - Calcular

   - `activity_peak_hour` (hora de pico de passos)
   - `activity_peak_value` (passos nesse pico)
   - Detectar se é noturno (anormal para BD maníaco)

6. 🟡 **Sedentariness ratio**

   - `sedentary_ratio = sedentary_min / (sedentary_min + active_min)`
   - Especialmente importante para depressão

7. 🟡 **Zepp stress score** (se disponível)
   - Proxy agregado de stress autonômico

### Tier 3 - OPCIONAL:

8. 🟢 **SpO2** (se disponível)
9. 🟢 **Body temperature trends** (Zepp)

---

## 5. PLANO DE AÇÃO RECOMENDADO

### Fase 1: Verificação de dados (imediato)

- [ ] Confirmar se Zepp health data (stress, temp, spo2) existe
- [ ] Contar registros de HRV disponíveis (você viu 19 - confirmar)
- [ ] Verificar cobertura temporal de cada sensor

### Fase 2: Novas agregações ETL (requer mudanças)

- [ ] Extrair HRV diário (SDNN mean/std/min/max)
- [ ] Calcular HR CV intra-dia
- [ ] Calcular sleep stage percentuais
- [ ] Calcular activity variância
- [ ] Incluir HR noturno (durante sleep)

### Fase 3: Label heuristics

- Reformular com conhecimento de ADHD/BD:
  - Baixa HRV + alta atividade + reduzida duração REM → ADHD signature
  - Reduzida HR noturno + reduzido sono total + alta atividade → maníaco
  - Reduzida atividade + alta fragmentação sono → depressivo

### Fase 4: Validação clínica

- Comparar com self-reports (Zepp mood, Apple State of Mind)
- Validar contra diário de sintomas se disponível

---

## 6. QUESTÕES PARA CONFIRMAÇÃO

Antes de implementar, confirme:

1. **Zepp health data:** Os dados de stress/temp/spo2 do Zepp estão sendo extraídos?

   - Localização esperada: `zepp_health/cloud/features_daily.csv` ou similar
   - Se SIM: Quantas dias têm dados?

2. **HRV coverage:** Você quer agregar os 19 registros de HRV?

   - Isso vai gerar ~1-2 linhas por dia (muito sparse)
   - Alternativa: usar Zepp stress score como proxy se melhor coberto

3. **Timestamp intrablocado:** Você tem dados de HR com timestamp intra-diário?

   - Necessário para calcular CV e HR noturno
   - Localização: `per-metric/apple_heart_rate.csv`

4. **Prioridade temporal:** Qual é o horizon temporal que mais importa?
   - Últimos 30 dias? (atual)
   - Últimos 6 meses?
   - Todo o histórico (7 anos)?
   - Impacta decisão de agregar sparse data

---

## 7. REFERÊNCIAS ACADÊMICAS RECOMENDADAS

1. **HRV em ADHD:**

   - Börger et al. (2021). "Heart rate variability in adults with ADHD" - Eur Arch Psychiatry
   - Thome et al. (2012). Reduced HRV in ADHD - systematic review

2. **Sleep em ADHD:**

   - Cortese et al. (2016). ADHD sleep comorbidity - systematic review & meta-analysis
   - Owens et al. (2009). Sleep in children with ADHD

3. **HRV em Bipolar:**

   - Lown et al. (2015). Heart rate variability and depressive symptoms in bipolar disorder
   - Quintana et al. (2012). Reduced HRV in depression

4. **Activity em BD:**

   - Faurholt-Jepsen et al. (2015). Electronic objective monitoring in bipolar disorder

5. **N-of-1 methodology:**
   - Crawford & Howell (1998). Single-case research in clinical psychology
   - Smith et al. (2012). Best practices for single-case designs

---

## PRÓXIMAS AÇÕES

**Aguardando sua confirmação sobre:**

1. Prioridade das Tier 1 features
2. Disponibilidade de Zepp health data
3. Tolerance para data sparsity em HRV
4. Timeline de implementação

**Não vou alterar código até ter clareza nestes pontos.**

---

**Status:** ⏸️ ANÁLISE CONCLUÍDA - AGUARDANDO CONFIRMAÇÃO DO USUÁRIO
