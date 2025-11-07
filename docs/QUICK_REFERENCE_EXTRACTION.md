# Referência Rápida - Comandos de Extração

## Apple INAPP (export.xml)

**Origem:** `data/raw/P000001/apple/export/apple.zip`  
**Destino:** `data/etl/P000001/SNAPSHOT/extracted/apple/inapp/`

**Comando Unificado (extrair Apple + Zepp):**

```bash
make extract PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn
```

**O que extrai:**

- `apple/inapp/apple_health_export/export.xml` (arquivo original)
- `apple/inapp/apple_health_export/export_cda.xml` (parsed)
- CSVs por métrica: `HKQuantityTypeIdentifierHeartRate.csv`, `HKQuantityTypeIdentifierHeartRateVariabilitySDNN.csv`, etc.

**Código:**

- `src/etl_pipeline.py` → `extract_run()`
- `src/cli/etl_runner.py` → CLI entry point
- Automáticamente descobre zips em `data/raw/P000001/apple/export/`

---

## Zepp CLOUD (cloud export com JSON)

**Origem:** `data/raw/P000001/zepp/zepp.zip` (encrypted)  
**Destino:** `data/etl/P000001/SNAPSHOT/extracted/zepp/cloud/`

### Opção 1: Extração de ZIP (se tiver arquivo .zip)

**Comando Unificado (extrair Apple + Zepp):**

```bash
make extract PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn
```

**O que extrai:**

- `zepp/cloud/HEARTRATE_AUTO/` → 430K+ registros HR-auto
- `zepp/cloud/SLEEP/` → com JSON naps embedded
- `zepp/cloud/ACTIVITY_STAGE/` → estágios de atividade
- `zepp/cloud/ACTIVITY_MINUTE/` → dados por minuto

**Código:**

- `src/etl_pipeline.py` → `extract_run()` com pyzipper (suporta AES encryption)
- Automaticamente descobre zips em `data/raw/P000001/zepp/` ou `data/raw/P000001/zepp/export/`
- Requer `ZEPP_ZIP_PASSWORD` (env var ou param)

### Opção 2: Preparação de dados JÁ EXTRAÍDOS (CSVs)

Se já tiver CSVs em `data/raw/P000001/zepp/`, usar:

```bash
make prepare-zepp PID=P000001 SNAPSHOT=2025-11-07
```

**O que faz:**

- Descobre CSVs em `data/raw/P000001/zepp/`
- Cria symlinks (ou copia) para `data/etl/P000001/2025-11-07/extracted/zepp/cloud/`
- Mantém estrutura: HEARTRATE_AUTO/, SLEEP/, ACTIVITY_STAGE/, ACTIVITY_MINUTE/

**Código:**

- `src/cli/prepare_zepp_data.py` → `prepare_zepp_data()`
- Auto-descobre arquivos CSV e JSON
- Suporta symlinks (default) ou cópia com `--symlink false`

---

## Biomarkers (após extração)

**Entrada:** `data/etl/P000001/SNAPSHOT/extracted/{apple,zepp}/`  
**Saída:** `data/etl/P000001/SNAPSHOT/joined/joined_features_daily_biomarkers.csv`

**Comando:**

```bash
make biomarkers PID=P000001 SNAPSHOT=2025-11-07
```

**O que faz:**

- Lê Zepp HR_AUTO → calcula SDNN, RMSSD, pNN50, CV
- Lê Zepp SLEEP (com JSON naps) → calcula sleep architecture, fragmentation
- Lê Zepp ACTIVITY_STAGE → calcula variance, fragmentation
- Lê Zepp para circadian metrics → nocturnal %, timing anomalies
- Valida Apple ↔ Zepp (correlações, agreement scores)
- Gera daily features matrix

**Código:**

- `src/domains/biomarkers/aggregate.py` → `aggregate_daily_biomarkers()`
- `src/cli/extract_biomarkers.py` → CLI wrapper
- Chamado via: `python -m src.cli.extract_biomarkers --participant P000001 --snapshot 2025-11-07 --data-dir data/etl/P000001/2025-11-07/extracted`

---

## Pipeline Completo

### Com Zepp ZIP:

```bash
# 1. Extrair (Apple + Zepp)
make extract PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn

# 2. Computar biomarkers
make biomarkers PID=P000001 SNAPSHOT=2025-11-07

# 3. Gerar labels
make labels PID=P000001 SNAPSHOT=2025-11-07

# 4. Treinar NB2
make nb2 PID=P000001 SNAPSHOT=2025-11-07
```

### Com Zepp CSVs já extraídos:

```bash
# 1. Preparar dados
make prepare-zepp PID=P000001 SNAPSHOT=2025-11-07

# 2-4. Igual acima
make biomarkers PID=P000001 SNAPSHOT=2025-11-07
make labels PID=P000001 SNAPSHOT=2025-11-07
make nb2 PID=P000001 SNAPSHOT=2025-11-07
```

### Ou tudo de uma vez:

```bash
make pipeline PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn
```

---

## Estrutura de Diretórios Esperada

```
data/raw/P000001/
├─ apple/export/
│  └─ apple.zip (contém export.xml)
└─ zepp/
   ├─ zepp.zip (encrypted, com password)
   └─ export/
      └─ zepp.zip (alternative location)

data/etl/P000001/2025-11-07/
├─ extracted/
│  ├─ apple/
│  │  └─ inapp/
│  │     ├─ apple_health_export/
│  │     │  ├─ export.xml
│  │     │  └─ export_cda.xml
│  │     └─ *.csv (por métrica)
│  └─ zepp/
│     └─ cloud/
│        ├─ HEARTRATE_AUTO/
│        ├─ SLEEP/
│        ├─ ACTIVITY_STAGE/
│        └─ ACTIVITY_MINUTE/
└─ joined/
   ├─ joined_features_daily.csv
   ├─ joined_features_daily_biomarkers.csv
   └─ features_daily_labeled.csv
```

---

## Variáveis de Ambiente

```bash
# Para extração Zepp (obrigatório se tiver ZIP encrypted)
export ZEPP_ZIP_PASSWORD=pLOeJaNn

# Ou passar via Makefile
make extract PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn
```

---

## Arquivos Relevantes

- **Extração unificada:** `src/etl_pipeline.py` (função `extract_run()`)
- **CLI extração:** `src/cli/etl_runner.py`
- **Preparação Zepp:** `src/cli/prepare_zepp_data.py`
- **Biomarkers:** `src/domains/biomarkers/`
  - `aggregate.py` - orchestrator
  - `hrv.py` - HRV metrics
  - `sleep.py` - Sleep architecture
  - `activity.py` - Activity metrics
  - `circadian.py` - Circadian rhythms
  - `validators.py` - Cross-device validation
- **Makefile:** `Makefile` (targets: `extract`, `prepare-zepp`, `biomarkers`, `labels`, `nb2`, `pipeline`)
