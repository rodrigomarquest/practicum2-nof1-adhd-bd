# 🔬 ANÁLISE ESTRUTURADA: Dados Reais Disponíveis para ADHD/BD

**Data:** 2025-11-07  
**Participante:** P000001  
**Horizonte:** 30 meses (filtrado depois)  
**Dispositivos:** Apple Health (iPhone + Watch), Zepp/Amazfit (GTR2/GTR4), Ring (recente)

---

## 1. ESTRUTURA DE SEGMENTAÇÃO (S1-S6)

Baseado no firmware/OS versions:

| Segmento | Período             | iOS     | Watch | Ring FW              | Validação                     |
| -------- | ------------------- | ------- | ----- | -------------------- | ----------------------------- |
| **S1**   | 01-01-23 → 12-04-23 | iOS16.3 | GTR2  | -                    | Apple + GTR2                  |
| **S2**   | 13-04-23 → 29-08-24 | iOS17.0 | GTR4  | -                    | Apple + GTR4 (upgrade)        |
| **S3**   | 30-08-24 → 02-11-24 | iOS17.0 | GTR4  | ZeppOS3.5            | Apple + GTR4 + Ring (estreia) |
| **S4**   | 03-11-24 → 10-04-25 | iOS17.5 | GTR4  | ZeppOS3.5 (pre2.3.1) | Apple + GTR4 + Ring           |
| **S5**   | 11-04-25 → 25-07-25 | iOS17.5 | GTR4  | ZeppOS3.5 2.3.1      | Apple + GTR4 + Ring           |
| **S6**   | 26-07-25 → 01-08-25 | iOS17.5 | GTR4  | ZeppOS3.5 3.1.2.3    | Apple + GTR4 + Ring           |

**Insights:**

- Transition S1→S2: GTR2 → GTR4 (firmware incomparable)
- Transition S2→S3: Ring introduced (nova fonte de dados)
- Transition S3→S4: iOS17.0 → iOS17.5 + Ring firmware update
- Transition S4→S5: Ring firmware 2.3.1 (stable)
- Transition S5→S6: Ring firmware 3.1.2.3 (latest)

**Use case para validação:** Comparar HR/HRV/Sleep entre Apple e Zepp para confirmar sincronização real

---

## 2. DADOS ZEPP DISPONÍVEIS (Raw Extract)

### 2.1 HEARTRATE_AUTO (430,926 registros!)

```
Columns: date, time, heartRate
Range: 2022-12-09 → 2024-06-30
Frequency: ~100 pontos por dia (intra-diário)
Clinical value: ⭐⭐⭐⭐⭐ (Alta resolução para HRV calculado)
```

**O QUE PODEMOS FAZER:**

- Calcular SDNN intra-diário (da Zepp, já que Apple tem apenas 19)
- Calcular HR variability (CV)
- Calcular HR mínimo/máximo/quartis
- Detectar taquicardia patterns (HR > 100 consistently)

### 2.2 SLEEP (252+ dias com estrutura JSON!)

```
Columns: date, deepSleepTime (min), shallowSleepTime (min), wakeTime (min), start, stop, REMTime (min), naps (JSON)
Range: 2022-12-09 → [verificar]
Frequency: Daily
Clinical value: ⭐⭐⭐⭐⭐ (Completo!)
```

**Estrutura JSON em naps:**

```json
[
  { "start": "2022-01-08 00:33:10+0000", "end": "2022-01-08 00:33:28+0000" },
  { "start": "2022-01-08 00:33:29+0000", "end": "2022-01-08 00:33:43+0000" }
]
```

**O QUE PODEMOS FAZER:**

- Extrair sleep latency (tempo até dormir)
- Contar fragmentação (número de "naps"/despertares)
- Calcular REM latency (tempo até primeiro REM)
- Detectar inversão circadiana (dormir em horários errados)
- Calcular sleep efficiency = (deep+shallow+REM) / total_time

**CRITICAL PARA BD:** REM latency curta (<60min) = biomarcador de depressão bipolar!

### 2.3 ACTIVITY_STAGE (4,366 registros intra-diários!)

```
Columns: date, start, stop, distance, calories, steps
Range: 2022-12-09 → 2024-06-30
Frequency: ~8-10 "eventos" de atividade por dia
Clinical value: ⭐⭐⭐⭐⭐ (Permite análise de variância intra-dia!)
```

**O QUE PODEMOS FAZER:**

- Variança de passos entre eventos (= irregularidade)
- Número de "sessões" de atividade (fragmentação)
- Duração média das sessões
- Pico máximo de atividade (max de um evento)
- Detectar ritmo circadiano (picos noturnos = potencial maníaco)
- Entropia da distribuição temporal

**CRITICAL PARA ADHD:** Alta variância intra-dia = hiperatividade fragmentada

### 2.4 ACTIVITY_MINUTE (86,051 registros!)

```
Columns: date, time, steps
Range: 2022-12-09 → 2024-06-30
Frequency: Minuto-a-minuto ou hora-a-hora
Clinical value: ⭐⭐⭐⭐⭐ (Máxima granularidade!)
```

**O QUE PODEMOS FAZER:**

- Calcular sedentariness blocks (períodos > 2h sem movimento)
- Horários de "picos de atividade" (quando mais ativo)
- Variabilidade temporal (padrão consistente vs. erratic)
- Aceleração/desaceleração (mudanças abruptas)

### 2.5 ACTIVITY (500 dias)

```
Columns: date, steps, distance, runDistance, calories
Range: 2022-12-09 → 2024-06-30
Frequency: Daily summary
Clinical value: ⭐⭐⭐ (Resumo do acima)
```

### 2.6 BODY (735 registros)

```
Columns: time, weight, height, bmi, fatRate, bodyWaterRate, boneMass, metabolism, muscleRate, visceralFat
Range: 2022-10-02 → 2025-10-22
Frequency: ~1-2x por semana (irregular)
Clinical value: ⭐⭐ (Menos crítico, mas trend importante)
```

**Pode indicar:**

- Mudanças rápidas de peso (stress/depressão)
- Visceral fat = markers de stress/inflamação

### 2.7 HEALTH_DATA (VAZIO)

```
Resultado: 0 registros (sem stress score, sem SPO2, sem temperatura)
```

---

## 3. DADOS APPLE DISPONÍVEIS (Raw Extract)

### 3.1 Apple Health export.xml (1.5GB)

**Types encontrados:**

- **HKQuantityTypeIdentifierHeartRate:** 204,930 registros ✅
- **HKQuantityTypeIdentifierHeartRateVariabilitySDNN:** 19 registros ✅✅✅
- **HKQuantityTypeIdentifierBodyMass:** 749 registros
- **HKQuantityTypeIdentifierBodyMassIndex:** 746 registros
- **HKQuantityTypeIdentifierHeight:** 8 registros
- **HKQuantityTypeIdentifierDietaryWater:** 8 registros

**Clinical value:**

- HR + HRV: ⭐⭐⭐⭐⭐
- Body metrics: ⭐⭐

### 3.2 per-metric (NÃO ENCONTRADO - provavelmente não gerado)

- Esperado: `per-metric/apple_heart_rate.csv`
- Esperado: `per-metric/apple_hrv_sdnn.csv`
- **Status:** Precisa ser criado durante ETL!

---

## 4. DADOS RING HELIO (inferência)

**Status:** Não encontrado em extracted/
**Possibilidade:** Ring data pode estar:

1. Sincronizado via Apple Health (como HKSamples)
2. Em arquivo separado não explorado
3. Ainda não exportado pelo usuário

**Verificar com usuário:** Onde está data do Ring?

---

## 5. BIOMARCADORES TIER 1 RECOMENDADOS (Com dados reais)

### ✅ IMPLEMENTÁVEL - Tier 1

#### 1. **HRV SDNN Diário** (Apple + Zepp)

```
Fonte: Apple HRV (19 registros) + Zepp HEARTRATE_AUTO (430k)
Métrica: Calcular SDNN do Zepp HR intra-diário (já que Apple tem poucos)
Output:
  - zepp_hrv_sdnn_daily_mean     [ms]
  - zepp_hrv_sdnn_daily_std      [ms]
  - zepp_hrv_sdnn_daily_min      [ms]
  - zepp_hrv_sdnn_daily_max      [ms]
```

#### 2. **Sleep Architecture Diário** (Zepp)

```
Fonte: Zepp SLEEP (252+ dias)
Métrica: Normalizar tempos para percentuais + latency
Output:
  - sleep_deep_pct      = deepSleepTime / total_sleep_time * 100
  - sleep_rem_pct       = REMTime / total_sleep_time * 100
  - sleep_light_pct     = shallowSleepTime / total_sleep_time * 100
  - sleep_latency_min   = ?  (inferir de start vs primeira mudança REM)
  - sleep_efficiency    = (deep + shallow + REM) / (start→stop duration) * 100
  - sleep_fragmentation = COUNT(naps JSON entries)
  - wake_time_pct       = wakeTime / total_time * 100
```

**CRITICAL:** REM latency curta em Zepp = depressão bipolar predictor

#### 3. **HR Variability Intra-dia** (Apple + Zepp)

```
Fonte: Zepp HEARTRATE_AUTO (min-by-min), Apple HR
Métrica: Coeficiente de variação
Output:
  - zepp_hr_cv         = std(HR) / mean(HR)      [unitless]
  - zepp_hr_rmssd      = root mean square successive differences
  - apple_hr_cv        = std(HR) / mean(HR)
```

**Clinical:** Baixo CV = autonômico estável; Alto CV = instável (stress/ADHD)

#### 4. **Activity Variação Intra-dia** (Zepp)

```
Fonte: Zepp ACTIVITY_STAGE (4,366 eventos)
Métrica: Variância entre eventos
Output:
  - activity_stage_var          = std(steps per event)
  - activity_stage_num_events   = COUNT(events per day)
  - activity_stage_mean_duration = mean(stop - start per event)
  - activity_max_single_event   = max(steps in one event)
  - activity_min_single_event   = min(steps in one event)
```

**Clinical ADHD:** Alta variância = hiperatividade fragmentada

#### 5. **Sedentariness Fragmentation** (Zepp)

```
Fonte: Zepp ACTIVITY_MINUTE (86k records)
Métrica: Detectar blocos sedentários
Output:
  - sedentary_blocks_gt_120min  = COUNT(consecutive 0-steps > 120 min)
  - sedentary_ratio             = (total sedentary mins) / (24h)
  - active_ratio                = (active mins) / (24h)
```

**Clinical Depression:** Sedentary > 80% = significant depression indicator

#### 6. **Circadian Rhythm Disruption** (Zepp)

```
Fonte: Zepp ACTIVITY_MINUTE + HEARTRATE_AUTO
Métrica: Detectar picos de atividade/HR em horários anormais
Output:
  - activity_peak_hour          = argmax(sum steps per hour)
  - activity_peak_value         = max(sum steps per hour)
  - hr_peak_hour                = argmax(mean HR per hour)
  - nocturnal_activity_ratio    = (22:00-06:00 steps) / (total steps)
```

**Clinical BD Maníaco:** Picos noturnos 22h-06h = potencial maníaco

### 🟡 IMPLEMENTÁVEL - Tier 2

#### 7. **Sleep Timing Abnormality** (Zepp)

```
Output:
  - sleep_start_hour_of_day     = HOUR(start timestamp)
  - sleep_end_hour_of_day       = HOUR(stop timestamp)
  - sleep_duration_hours        = (stop - start) / 60
  - sleep_consistency_std       = std(sleep_start_hour across week)
```

**Clinical BD:** Necessidade de sono reduzida (dorme 3h) em maníaco

#### 8. **HR Baseline & Trends** (Apple + Zepp)

```
Output:
  - apple_hr_mean_daily         [bpm]
  - apple_hr_max_daily          [bpm]
  - zepp_hr_mean_daily          [bpm]
  - zepp_hr_max_daily           [bpm]
  - cross_device_hr_correlation [r value]
```

**Validation Purpose:** Correlação Apple-Zepp = confiabilidade de dados

#### 9. **Body Composition Trends** (Zepp)

```
Output:
  - weight_7day_delta           = weight_today - weight_7d_ago
  - visceral_fat_7day_delta
  - fat_rate_7day_delta
```

**Clinical:** Mudança rápida = stress marker

### 🟢 EXPLORÁVEL - Tier X

#### 10. **Device Sync Validation (Cross-device)**

```
Para cada S1-S6:
  - Dias com dados Apple AND Zepp HR
  - Correlação HR Apple vs Zepp (deve ser > 0.8)
  - Latência de sincronização (se aplicável)
  - Flagging inconsistencies
```

**Research Goal:** Validar replicação de dados entre devices!

---

## 6. MAPA DE DADOS x SEGMENTOS (S1-S6)

Precisa verificar em cada S1-S6:

| Métrica       | S1  | S2  | S3  | S4  | S5  | S6  | Validação        |
| ------------- | --- | --- | --- | --- | --- | --- | ---------------- |
| Apple HR      | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | Device ID em XML |
| Zepp HR       | ?   | ?   | ✅  | ✅  | ✅  | ✅  | Por firmware     |
| Zepp Sleep    | ?   | ?   | ✅  | ✅  | ✅  | ✅  | Por firmware     |
| Zepp Activity | ?   | ?   | ✅  | ✅  | ✅  | ✅  | Por firmware     |
| Ring Data     | ❌  | ❌  | ?   | ?   | ?   | ?   | **TBD**          |

---

## 7. DESAFIOS E SOLUÇÕES

### 🔴 Problema 1: HRV Apple tem apenas 19 registros

**Solução:** Usar SDNN calculado do Zepp HR_AUTO (430k pontos → calculate SDNN em sliding windows)

### 🔴 Problema 2: Zepp data ends em 2024-06-30 (gap 6 meses!)

**Solução:**

- Use Apple HR para últimos 6 meses como fallback
- Investigar se Ring data preenche gap

### 🔴 Problema 3: Segmentação S1-S2 sem Zepp

**Solução:**

- S1-S2 usa apenas Apple
- Marcar segment_device_flag = "apple_only" para S1-S2

### 🔴 Problema 4: HEALTH_DATA Zepp vazio (sem stress/spo2)

**Solução:** Usar HR-derived metrics (RMSSD, HRV) como proxy de stress

### 🟡 Problema 5: Ring data não encontrado

**Solução:** Verificar com usuário localização

- Pode estar em Apple Health?
- Arquivo separado?
- Ainda não exportado?

---

## 8. PLANO DE AÇÃO RECOMENDADO

### Fase 0: Validação (30 minutos)

- [ ] Confirmar localização dos dados de Ring (se existem)
- [ ] Confirmar gap Zepp 2024-06-30 vs esperado 2025-11-07
- [ ] Verificar disponibilidade de data/versão metadata em cada record Zepp

### Fase 1: Engenharia de Features (ETL Modification)

- [ ] Extrair metadata de device/firmware de cada record
- [ ] Criar arquivo segmentation (S1-S6 com datas/device mapping)
- [ ] Implementar SDNN calculation do Zepp HR_AUTO
- [ ] Implementar 10 features Tier 1 + Tier 2

### Fase 2: Feature Quality

- [ ] Cross-validation Apple ↔ Zepp (correlação)
- [ ] Flag missing data by segment
- [ ] Generate completeness matrix (S1-S6 vs metrics)

### Fase 3: Clinical Heuristics v2

- [ ] Integrar features Tier 1-2 em `build_heuristic_labels_v2.py`
- [ ] Usar clinical knowledge (REM latency, activity variance, etc)
- [ ] Output: mood labels + confidence scores

---

## 9. QUESTÕES PARA CONFIRMAÇÃO

Antes de começar implementação:

1. **Ring data:** Onde está? Apple Health? Arquivo separado? Zepp cloud?
2. **Zepp gap:** É normal que termine em jun/2024? Você tem acesso a dados mais recentes?
3. **S1-S2 sem Zepp:** Ok focar esses segmentos apenas em Apple?
4. **Prioridade:** Tier 1 first, depois Tier 2-X? Ou paralelizar?
5. **Validação:** Você tem any ground-truth (self-reports, clínical diagnóstico) para validar correlações?

---

**Status:** ✅ ANÁLISE COMPLETA - PRONTO PARA IMPLEMENTAÇÃO APÓS CONFIRMAÇÕES
