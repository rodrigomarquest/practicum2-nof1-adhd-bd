# Implementação: Estrutura Vendor/Variant com MAX_RECORDS

Data: 2025-11-06  
Status: ✅ CONCLUÍDA - FASE 1 COMPLETA

## Resumo Executivo

Implementação bem-sucedida da estrutura de vendor/variant para organização de features ETL, com suporte a:

- ✅ Extração paralela de Apple e Zepp para todos os domínios (activity, cardio, sleep)
- ✅ Streaming de XML para lidar com arquivos grandes (export_cda.xml 4GB+)
- ✅ Limitação de registros processados com MAX_RECORDS (testado com 128)
- ✅ Backward compatibility com caminhos legacy
- ✅ 288 registros processados com sucesso em teste

## Arquitetura Implementada

### Estrutura de Diretórios

```
data/etl/<PID>/<SNAPSHOT>/features/
├── activity/
│   ├── apple/
│   │   └── inapp/
│   │       └── features_daily.csv
│   └── zepp/
│       └── cloud/
│           └── features_daily.csv
├── cardio/
│   ├── apple/
│   │   └── inapp/
│   │       └── features_daily.csv
│   └── zepp/
│       └── cloud/
│           └── features_daily.csv
└── sleep/
    ├── apple/ (opcional - se houver dados)
    │   └── inapp/
    │       └── features_daily.csv
    └── zepp/
        └── cloud/
            └── features_daily.csv
```

### Padrão Vendor/Variant

```
<domain>/<vendor>/<variant>/features_daily.csv

Exemplos:
- activity/apple/inapp/features_daily.csv
- activity/zepp/cloud/features_daily.csv
- cardio/apple/inapp/features_daily.csv
- cardio/zepp/cloud/features_daily.csv
- sleep/zepp/cloud/features_daily.csv
```

## Implementação por Domínio

### 1. ACTIVITY (`src/domains/activity/activity_from_extracted.py`)

**Fontes de dados:**

- ✅ Apple: export.xml com extração de atividade (steps, exercise, etc)
- ✅ Zepp: cloud CSVs com dados de ACTIVITY, ACTIVITY_MINUTE

**Funções:**

- `discover_apple()`: Retorna lista de (Path, vendor, variant)
- `load_apple_daily()`: Extrai Apple com MAX_RECORDS
- `load_zepp_activity_daily()`: Extrai Zepp com MAX_RECORDS
- `write_seed()`: Escreve apple/inapp e zepp/cloud separado

**Resultados com MAX_RECORDS=128:**

- apple/inapp: 4 rows (3 dias de dados)
- zepp/cloud: 128 rows (limitado por MAX_RECORDS)
- Total: 132 rows

### 2. CARDIO (`src/domains/cardiovascular/cardio_from_extracted.py`)

**Fontes de dados:**

- ✅ Apple: export.xml com extração de heartrate (NEW!)
- ✅ Zepp: cloud CSVs com dados de HEARTRATE, HEARTRATE_AUTO

**Funções novas:**

- `load_apple_cardio_from_xml()`: Extração streaming de heartrate
  - Usa `iterparse` para processamento memory-efficient
  - Suporta export.xml e export_cda.xml
  - Lxml recovery para arquivos corrompidos
  - Agregação por dia (mean, max, count)

**Resultados com MAX_RECORDS=128:**

- apple/inapp: 1 row (1 dia de dados)
- zepp/cloud: 68 rows
- Total: 69 rows

### 3. SLEEP (`src/domains/sleep/sleep_from_extracted.py`)

**Fontes de dados:**

- ✅ Apple: apple_sleep_intervals.csv (já existente)
- ✅ Zepp: cloud CSVs com dados de SLEEP

**Funções:**

- `load_apple_sleep_daily()`: Carrega e agrega sleep intervals
- `load_zepp_sleep_daily_from_cloud()`: Extrai Zepp com MAX_RECORDS
- `discover_apple_sleep()`: Descobre dados disponíveis
- `discover_zepp_sleep()`: Descobre dados disponíveis

**Resultados com MAX_RECORDS=128:**

- apple/inapp: 0 rows (sem dados neste snapshot)
- zepp/cloud: 87 rows
- Total: 87 rows

## Otimizações Implementadas

### 1. Streaming XML (Memory-efficient)

```python
# Cardio/Activity - Heartrate extraction
it = ET.iterparse(str(xml_path), events=("end",))
for event, elem in it:
    if elem.tag.endswith("Record"):
        typ = elem.get("type")
        if typ == "HKQuantityTypeIdentifierHeartRate":
            # Process heartrate record
            hr_count += 1
            if max_records and hr_count >= max_records:
                break
    elem.clear()  # Clear element to free memory
```

**Benefícios:**

- Processa export_cda.xml (4GB+) sem carregar tudo na memória
- Streaming com `iterparse` vs batch `parse`
- Limpeza contínua de elementos: `elem.clear()`

### 2. MAX_RECORDS Domain-Specific Filtering

```python
# Cardio - count only HR records (not all XML records)
hr_count = 0
if typ == "HKQuantityTypeIdentifierHeartRate":
    hr_count += 1
    if max_records and hr_count >= max_records:
        break

# Activity - count only activity types
if typ in ("StepCount", "Distance", "ActiveEnergy", ...):
    records_processed += 1
    if max_records and records_processed >= max_records:
        break

# Sleep - limit rows from CSV
rows_read = 0
df = df.iloc[:rows_available] if max_records else df
```

**Benefícios:**

- Conta APENAS registros relevantes ao domínio
- Evita processar 6M+ registros XML desnecessários
- Early termination quando MAX_RECORDS atingido

### 3. Backward Compatibility

Todos os seeds continuam escrevendo para:

```
features/<domain>/features_daily.csv (LEGACY)
```

Além de:

```
features/<domain>/<vendor>/<variant>/features_daily.csv (NEW)
```

Permitindo transição gradual sem quebra de código existente.

## Testes Realizados

### Teste 1: MAX_RECORDS=128 em Todos os Domínios

```bash
make etl activity DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
make etl cardio DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
make etl sleep DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
```

**Resultado:**

- ✅ Activity: 132 rows (4 apple + 128 zepp)
- ✅ Cardio: 69 rows (1 apple + 68 zepp)
- ✅ Sleep: 87 rows (0 apple + 87 zepp)
- **TOTAL: 288 registros processados**

### Teste 2: Join com Estrutura Vendor/Variant

```bash
make etl join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06
```

**Resultado:**

- ✅ Descobri arquivos corretos com glob recursivo
- ✅ Processou cardio/zepp/cloud e activity/zepp/cloud
- ✅ Gerou joined_features_daily.csv com merge correto
- ⚠️ Observação: Join escolhe UM arquivo per domínio (não combina apple+zepp ainda)

## Dados Gerados

### Snapshot: P000001/2025-11-06

```
activity/apple/inapp/features_daily.csv: 4 rows
  - Domínios: StepCount, Distance, ActiveEnergy
  - Período: 2021-05-10 a 2021-05-12

activity/zepp/cloud/features_daily.csv: 128 rows
  - Domínio: Zepp activity data (MAX_RECORDS=128)
  - Período: variado

cardio/apple/inapp/features_daily.csv: 1 row
  - Domínio: HKQuantityTypeIdentifierHeartRate
  - Período: 2021-05-14
  - Valores: mean=78.5 bpm, max=104 bpm

cardio/zepp/cloud/features_daily.csv: 68 rows
  - Domínio: Zepp heartrate data (HEARTRATE CSVs)
  - Período: variado

sleep/zepp/cloud/features_daily.csv: 87 rows
  - Domínio: Zepp sleep data (MAX_RECORDS=128)
  - Período: variado
```

## Issues e Observações

### ✅ Resolvidos

- [x] MAX_RECORDS conta apenas registros relevantes do domínio
- [x] Streaming para evitar estouro de memória em XML 4GB+
- [x] Apple heartrate extraído de export.xml (não apenas CSV)
- [x] Backward compatibility com caminhos legacy mantida
- [x] Discovery automático de vendor/variant

### ⚠️ Pendentes (Próxima Fase)

- [ ] Join deveria COMBINAR apple+zepp por domínio (não escolher apenas um)
- [ ] Enriquecimento de features (enriched/prejoin e enriched/postjoin)
- [ ] QC comparativo entre original e enriquecido
- [ ] Otimização para processar arquivo export_cda.xml completo (sem MAX_RECORDS)

### ℹ️ Observações Técnicas

- Apple sleep ausente no snapshot (esperado - nem sempre disponível)
- Zepp zip não processado (requer ZEPP_ZIP_PASSWORD)
- SonarQube flags: Cognitive Complexity (não bloqueador, funcionando)

## Próximas Fases

### FASE 2: Enriquecimento

```
enriched/
├── prejoin/
│   └── <domain>/<vendor>/<variant>/enriched_<domain>.csv
└── postjoin/
    └── <domain>/enriched_<domain>.csv
```

### FASE 3: Join Aprimorado

```
joined/
├── joined_features_daily.csv (global - current)
├── features_activity.csv (per-domain)
├── features_cardio.csv (per-domain)
└── features_sleep.csv (per-domain)
```

### FASE 4: QC Comparativo

```
qc/
└── enriched_compare_<domain>.csv
```

## Arquivos Modificados

### Novos

- (nenhum arquivo novo criado - apenas modificações)

### Modificados

1. `src/domains/activity/activity_from_extracted.py`

   - Adicionado `discover_apple()` com retorno de (Path, vendor, variant)
   - Modificado `write_seed()` para suportar vendor/variant
   - Adicionado escrita de activity/zepp/cloud

2. `src/domains/cardiovascular/cardio_from_extracted.py`

   - Adicionado `load_apple_cardio_from_xml()` com streaming
   - Modificado `load_apple_cardio_daily()` para usar XML
   - Adicionado argumento `--max-records`
   - Modificado main para escrever apple/inapp e zepp/cloud

3. `src/domains/sleep/sleep_from_extracted.py`
   - Adicionado argumento `--max-records` (já existia suporte)
   - Modificado main para escrever sleep/zepp/cloud

## Comandos de Teste

```bash
# Teste individual de cada seed com MAX_RECORDS=128
make etl activity DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
make etl cardio DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128
make etl sleep DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06 MAX_RECORDS=128

# Teste de join
make etl join DRY_RUN=0 PID=P000001 SNAPSHOT=2025-11-06

# Listar arquivos gerados
find data/etl/P000001/2025-11-06/features -name "features_daily.csv" -type f | sort

# Verificar conteúdo
head -5 data/etl/P000001/2025-11-06/features/activity/apple/inapp/features_daily.csv
```

## Conclusão

✅ **IMPLEMENTAÇÃO COMPLETA E FUNCIONAL**

A estrutura vendor/variant foi implementada com sucesso em todos os três domínios (activity, cardio, sleep), com:

- Extração paralela de Apple e Zepp
- Processamento memory-efficient de XML grande
- Limitação controlada de registros processados
- Testes validando 288 registros processados com sucesso
- Backward compatibility mantida
- Join funcionando com nova estrutura

Pronto para próximas fases de enriquecimento e QC.
