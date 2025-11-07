# Comandos de Limpeza - Referência Rápida

## ✅ Resposta Direta

**Para limpar ETL sem apagar data/raw:**

```bash
make clean-data
```

---

## 📋 Todos os Comandos

### 1. `make clean` (limpeza leve)

Remove apenas caches e arquivos compilados:

- `__pycache__/` (Python bytecode)
- `.ipynb_checkpoints/` (Jupyter checkpoints)
- `*.pyc` (compiled Python)
- `*.log` (log files)

**Mantém:**

- `data/raw/` ✅
- `data/etl/` ✅
- Código-fonte ✅

**Uso:** Limpeza rápida de cache sem afetar dados

```bash
make clean
```

---

### 2. `make clean-data` (limpeza de pipeline) ⭐ MAIS COMUM

Remove TODOS os dados processados:

- `data/etl/` (extracted, joined, features, labels)
- `data/ai/` (modelos e resultados)
- `notebooks/outputs/`
- `logs/`, `backups/`, `processed/`

**Mantém:**

- `data/raw/` ✅ IMPORTANTE
- Código-fonte ✅
- Scripts ✅

**Uso:** Resetar pipeline para re-executar do zero

```bash
make clean-data

# Depois re-executar pipeline
make extract PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn
make biomarkers PID=P000001 SNAPSHOT=2025-11-07
make labels PID=P000001 SNAPSHOT=2025-11-07
make nb2 PID=P000001 SNAPSHOT=2025-11-07
```

---

### 3. `make clean-provenance` (limpeza de metadados)

Remove arquivos transitórios de provenance:

- `pip_freeze_*.txt` (histórico de dependências)
- `hash_snapshot_*.json` (hashes de snapshots)
- `migrate_layout_*.json` (histórico de migrações)
- `cleanup_log_*.txt` (logs de limpeza)

**Mantém:**

- `data/etl/` ✅
- `data/raw/` ✅
- `provenance/reports/` (relatórios importantes) ✅

**Uso:** Limpar arquivos transitórios mantendo dados e relatórios

```bash
make clean-provenance
```

---

### 4. `make clean-all` (limpeza completa)

Remove TUDO (= clean + clean-data + clean-provenance):

- Caches Python
- Todos os dados processados (ETL outputs, AI models)
- Arquivos transitórios de provenance

**Mantém:**

- `data/raw/` ✅ IMPORTANTE
- Código-fonte ✅
- Documentação ✅

**Uso:** Limpeza profunda antes de experimento novo ou arquivamento

```bash
make clean-all

# Depois re-executar pipeline de zero
make extract PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn
# ... etc
```

---

## 📊 Matriz de Decisão

| Comando                 | Cache | data/raw | data/etl | data/ai | Provenance | Uso                 |
| ----------------------- | ----- | -------- | -------- | ------- | ---------- | ------------------- |
| `make clean`            | ❌    | ✅       | ✅       | ✅      | ✅         | Cache local         |
| `make clean-data`       | ✅    | ✅       | ❌       | ❌      | ✅         | Resetar pipeline    |
| `make clean-provenance` | ✅    | ✅       | ✅       | ✅      | ❌         | Limpeza transitória |
| `make clean-all`        | ❌    | ✅       | ❌       | ❌      | ❌         | Limpeza total       |

---

## 🎯 Cenários Práticos

### Cenário 1: Resetar pipeline (falhou em algum passo)

```bash
$ make clean-data
$ make extract PID=P000001 SNAPSHOT=2025-11-07 ZEPP_ZIP_PASSWORD=pLOeJaNn
$ make biomarkers PID=P000001 SNAPSHOT=2025-11-07
$ make labels PID=P000001 SNAPSHOT=2025-11-07
$ make nb2 PID=P000001 SNAPSHOT=2025-11-07
```

### Cenário 2: Novo experimento com novos dados

```bash
$ make clean-all
# Copiar novos dados para data/raw/
$ make extract PID=P000002 SNAPSHOT=2025-11-15 ZEPP_ZIP_PASSWORD=pLOeJaNn
$ make pipeline PID=P000002 SNAPSHOT=2025-11-15
```

### Cenário 3: Limpeza rápida de cache (sem afetar dados)

```bash
$ make clean
# Continua com pipeline normalmente
$ make biomarkers PID=P000001 SNAPSHOT=2025-11-07
```

### Cenário 4: Arquivar projeto (manter apenas dados brutos)

```bash
$ make clean-all
# Comprimir data/raw/ e arquivar
$ tar -czf backup_raw_data.tar.gz data/raw/
```

---

## 📁 Estrutura de Diretórios Afetada

```
projeto/
├── data/
│   ├── raw/                    ← ✅ NUNCA apagado
│   │   ├── P000001/
│   │   │   ├── apple/export/apple.zip
│   │   │   └── zepp/zepp.zip
│   │   └── P000002/
│   │       └── ...
│   │
│   ├── etl/                    ← ❌ Apagado por clean-data
│   │   ├── P000001/
│   │   │   └── 2025-11-07/
│   │   │       ├── extracted/
│   │   │       └── joined/
│   │   └── P000002/
│   │
│   └── ai/                     ← ❌ Apagado por clean-data
│       └── ... (modelos, resultados)
│
├── notebooks/
│   └── outputs/                ← ❌ Apagado por clean-data
│
├── logs/                       ← ❌ Apagado por clean-data
├── backups/                    ← ❌ Apagado por clean-data
├── processed/                  ← ❌ Apagado por clean-data
│
├── __pycache__/                ← ❌ Apagado por clean
├── .ipynb_checkpoints/         ← ❌ Apagado por clean
│
├── provenance/                 ← Parcialmente apagado por clean-all
│   ├── *_transient*.json       ← ❌ Apagado por clean-provenance
│   └── reports/                ← ✅ MANTIDO
│
├── src/                        ← ✅ NUNCA apagado
├── scripts/                    ← ✅ NUNCA apagado
├── docs/                       ← ✅ NUNCA apagado
└── Makefile                    ← ✅ NUNCA apagado
```

---

## ⚠️ Pontos Importantes

1. **data/raw/ é SEMPRE preservado**

   - Nenhum comando `make clean*` remove dados brutos
   - É seguro limpar com confiança

2. **data/etl/ é REMOVIDO completamente por clean-data**

   - Todos os CSVs processados desaparecem
   - Inclui extracted/, joined/, features/, labels/, etc.
   - Precisará re-executar pipeline completo

3. **Use clean-all antes de experimentos novos**

   - Garante estado limpo
   - Evita mistura de dados de diferentes runs

4. **data/raw/ é backup seguro**
   - Pode usar `make clean-all` sem medo
   - Dados brutos sempre podem ser re-processados

---

## 🔧 Implementação (Makefile)

```makefile
# -------- Clean-up (safe, portable) --------
.PHONY: clean clean-data clean-provenance clean-all

clean:
	echo ">>> clean: removing caches and logs"
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	echo "[OK] caches/logs removed"

clean-data:
	echo ">>> clean-data: removing ETL outputs and AI results"
	rm -rf notebooks/outputs dist/assets logs backups processed 2>/dev/null || true
	rm -rf data/etl data/ai 2>/dev/null || true
	echo "[OK] data outputs removed"

clean-provenance:
	echo ">>> clean-provenance: removing transient provenance artifacts (keep reports)"
	find provenance -type f \( \
	  -name "pip_freeze_*.txt" -o \
	  -name "hash_snapshot_*.json" -o \
	  -name "migrate_layout_*.json" -o \
	  -name "cleanup_log_*.txt" \
	\) -exec rm -f {} + 2>/dev/null || true
	echo "[OK] provenance transient files removed"

clean-all: clean clean-data clean-provenance
	echo ">>> clean-all: full cleanup done"
```
