# Clinical Biomarkers Extraction (Tier 1 + 2 + X)

## Overview

Este módulo implementa a extração de biomarcadores clínicos com validade científica para diagnóstico de ADHD/BD via wearables.

**Arquitetura:**

- **Tier 1 (Core):** HRV, Sleep %, HR Variação, Activity Variância, Sedentariness, Circadian
- **Tier 2 (Clinical):** Sleep timing, HR trends, Body composition
- **Tier X (Exploratory):** Cross-device validation, Device reliability

## File Structure

```
src/biomarkers/
├── __init__.py              # Module entry point
├── segmentation.py          # Device mapping (S1-S6)
├── hrv.py                   # HRV SDNN/RMSSD/pNN50 (Tier 1)
├── sleep.py                 # Sleep architecture + JSON naps (Tier 1)
├── activity.py              # Activity variance + sedentariness (Tier 1)
├── circadian.py             # Nocturnal activity, sleep timing (Tier 2)
├── validators.py            # Cross-device validation (Tier X)
└── aggregate.py             # Main orchestrator

scripts/
└── extract_biomarkers.py    # CLI entry point for extraction
```

## Usage

### 1. Full Pipeline (After ETL Extract + Join)

```bash
make biomarkers PID=P000001 SNAPSHOT=2025-11-07
```

### 2. Programmatic Usage

```python
from src.biomarkers import aggregate

# Load raw data and extract all biomarkers
df_biomarkers = aggregate.aggregate_daily_biomarkers(
    zepp_hr_auto_path="data/etl/P000001/2025-11-07/extracted/zepp/cloud/HEARTRATE_AUTO/*.csv",
    zepp_sleep_path="data/etl/P000001/2025-11-07/extracted/zepp/cloud/SLEEP/*.csv",
    zepp_activity_stage_path="data/etl/P000001/2025-11-07/extracted/zepp/cloud/ACTIVITY_STAGE/*.csv",
    zepp_activity_minute_path="data/etl/P000001/2025-11-07/extracted/zepp/cloud/ACTIVITY_MINUTE/*.csv",
    apple_hr_path="data/etl/P000001/2025-11-07/extracted/apple/inapp/HKQuantityTypeIdentifierHeartRate.csv",
    apple_hrv_path="data/etl/P000001/2025-11-07/extracted/apple/inapp/HKQuantityTypeIdentifierHeartRateVariabilitySDNN.csv",
    apple_activity_path="data/etl/P000001/2025-11-07/extracted/apple/inapp/HKQuantityTypeIdentifierActiveEnergyBurned.csv",
    cutoff_date=pd.Timestamp("2025-11-07") - pd.DateOffset(months=30),
    participant="P000001",
    snapshot="2025-11-07",
)

print(df_biomarkers.shape)  # (N_days, N_features)
print(df_biomarkers.columns)
```

### 3. Individual Biomarker Extraction

```python
from src.biomarkers import hrv, sleep, activity, circadian

# HRV from Zepp
df_hrv = hrv.compute_hrv_daily(zepp_hr_auto_df)

# Sleep metrics
df_sleep = sleep.compute_sleep_metrics(zepp_sleep_df)

# Activity variance
df_activity_var = activity.compute_activity_stage_variance(zepp_activity_stage_df)

# Circadian rhythm
df_circadian = circadian.compute_circadian_metrics(zepp_activity_minute_df, zepp_sleep_df)
```

## Tier 1 Features (Core ADHD/BD Markers)

| Feature                        | Source                 | Clinical Significance              | Thresholds             |
| ------------------------------ | ---------------------- | ---------------------------------- | ---------------------- |
| `zepp_hrv_sdnn_ms`             | Zepp HR_AUTO           | HRV (↓ in ADHD/anxiety)            | < 50ms = abnormal      |
| `zepp_hrv_rmssd_ms`            | Zepp HR_AUTO           | Parasympathetic tone               | < 20ms = abnormal      |
| `zepp_hr_cv_pct`               | Zepp HR_AUTO           | HR variability proxy               | High = dysautonomia    |
| `sleep_deep_pct`               | Zepp SLEEP             | Deep sleep quality                 | < 20% = poor           |
| `sleep_rem_pct`                | Zepp SLEEP             | REM sleep quality                  | < 15% = poor           |
| `sleep_fragmentation_count`    | Zepp SLEEP (naps JSON) | Sleep stability                    | > 4 = fragmented       |
| `activity_variance_std`        | Zepp ACTIVITY_STAGE    | Activity irregularity (↑ in ADHD)  | > 50 = high variance   |
| `activity_fragmentation_ratio` | Zepp ACTIVITY_STAGE    | Activity fragmentation             | > 1.5 = fragmented     |
| `sedentary_time_pct`           | Zepp ACTIVITY_MINUTE   | Sedentariness (↑ in depression)    | > 80% = high           |
| `sedentary_blocks_count`       | Zepp ACTIVITY_MINUTE   | Immobility periods                 | > 8 blocks = excessive |
| `nocturnal_activity_pct`       | Zepp ACTIVITY_MINUTE   | Nocturnal hyperactivity (BD mania) | > 20% = high           |

## Tier 2 Features (Clinical Context)

| Feature                      | Source               | Clinical Significance                   |
| ---------------------------- | -------------------- | --------------------------------------- | ----------------- |
| `sleep_duration_h`           | Zepp SLEEP           | Total sleep (reduced in mania)          |
| `sleep_duration_var`         | Rolling 7-day window | Sleep timing irregularity (ADHD marker) |
| `sleep_duration_cv`          | Rolling 7-day window | Sleep consistency                       | > 40% = irregular |
| `activity_peak_hour`         | Zepp ACTIVITY_MINUTE | Time of max activity                    |
| `early_morning_activity_pct` | Zepp ACTIVITY_MINUTE | Early awakening (depression)            | > 20% = high      |

## Tier X Features (Device Validation)

| Feature           | Source       | Purpose                              |
| ----------------- | ------------ | ------------------------------------ |
| `apple_hr_mean`   | Apple HR     | Cross-device correlation             |
| `zepp_hr_mean`    | Zepp HR_AUTO | Cross-device correlation             |
| `hr_correlation`  | Both         | Device agreement (0-1)               |
| `hr_bias`         | Both         | Systematic difference (Zepp - Apple) |
| `agreement_score` | Both         | Quality metric (0-100%)              |

## Segmentation (S1-S6)

Cada registro é rotulado com um segmento de dispositivo/firmware:

```
S1 (2023-01-01 → 2023-12-12): iOS 16.3, GTR 2, no Ring
S2 (2023-12-13 → 2024-08-29): iOS 17.0, GTR 4, no Ring
S3 (2024-08-30 → 2024-11-02): iOS 17.0, GTR 4, Ring pre-2.3.1
S4 (2024-11-03 → 2025-04-10): iOS 17.5, GTR 4, Ring pre-2.3.1
S5 (2025-04-11 → 2025-07-25): iOS 17.5, GTR 4, Ring 2.3.1
S6 (2025-07-26 → 2025-11-07): iOS 17.5, GTR 4, Ring 3.1.2.3
```

Permite análise de:

- Device reliability per segment
- Firmware effects
- Cross-device validation per device combination

## Clinical Heuristics (build_heuristic_labels.py v2)

O módulo gera labels com validade clínica usando Tier 1+2:

```python
# Exemplos de regras heurísticas

if rem_latency < 60 min:
    label = "BD_DEPRESSIVE"  # REM latency < 60min é marcador de depressão
    confidence = 0.85

elif nocturnal_activity > 20% AND sleep_duration < 6h:
    label = "BD_MANIC"  # Atividade noturna + redução de sono = mania
    confidence = 0.75

elif activity_variance_std > 80th_percentile AND activity_fragmentation > 1.5:
    label = "ADHD"  # Activity variance = marcador ADHD
    confidence = 0.70

elif sedentary_time > 80% AND sleep_rem_pct < 15%:
    label = "DEPRESSION"  # Alta sedentariness + REM reduzido = depressão
    confidence = 0.65

else:
    label = "CONTROL"
    confidence = 0.50
```

## Data Quality Flags

Para cada dia, são gerados flags de qualidade:

```python
df["missing_hrv"]         # HRV data missing
df["missing_sleep"]       # Sleep data missing
df["missing_activity"]    # Activity data missing
df["data_quality_score"]  # Overall quality (0-100%)
```

## Output

Arquivo: `data/etl/P000001/SNAPSHOT/joined/joined_features_daily_biomarkers.csv`

Colunas:

- `date` - Data (yyyy-mm-dd)
- `segment` - Device segmentation (S1-S6)
- `zepp_hrv_*` - HRV metrics
- `sleep_*` - Sleep metrics
- `activity_*` - Activity metrics
- `nocturnal_*`, `early_morning_*` - Circadian metrics
- `apple_hr_*`, `hr_*` - Cross-device validation
- `missing_*`, `data_quality_score` - Quality flags

## References

**ADHD Biomarkers:**

- Börger et al. (2021) - Activity variability in ADHD
- Thome et al. (2012) - Reduced HRV in ADHD

**BD Biomarkers:**

- Lown et al. (2015) - Nocturnal activity in mania
- Quintana et al. (2012) - HRV in bipolar disorder
- Riemann et al. (2007) - Sleep architecture in BD depression (REM latency)

## Integration with ETL

```bash
# Complete pipeline
make full PID=P000001 SNAPSHOT=2025-11-07
make biomarkers PID=P000001 SNAPSHOT=2025-11-07
make labels PID=P000001 SNAPSHOT=2025-11-07
python run_ml6_engage7.py --pid P000001 --snapshot 2025-11-07
```

## Troubleshooting

### "No valid daily HRV records computed"

- Verificar se Zepp HEARTRATE_AUTO tem dados suficientes
- Precisar de >= 10 HR samples por dia para computar SDNN

### "Activity stage DataFrame is empty"

- Zepp ACTIVITY_STAGE pode não ter dados em período antigo
- Checar range de datas disponíveis

### "Sleep data missing"

- Zepp SLEEP pode ter gap (2024-06-30 em dados antigos)
- Usar apenas período com dados disponíveis

## Next Steps

1. ✅ Implementar Tier 1 features (HRV, Sleep, Activity, Circadian)
2. ✅ Implementar Tier 2 features (Sleep timing, HR trends)
3. ✅ Implementar Tier X validation (Cross-device)
4. ⏳ Reescrever `build_heuristic_labels.py` com clínica
5. ⏳ Cross-segment validation em ML6 (S3 train → S4 test)
6. ⏳ Ring data integration (quando disponível)
