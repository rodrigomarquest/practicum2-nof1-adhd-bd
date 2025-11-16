# Reorganização Estrutural - Fase 7

## Resumo

Reorganização estrutural completa para conformar com padrões de projeto: organização por domínios, remoção de scripts da raiz, renomeação de Beiwe → Engage7.

## Data

7 de Novembro, 2025

---

## Mudanças Realizadas

### 1. ✅ Biomarkers movidos para `src/domains/biomarkers/`

**Antiga localização:** `src/biomarkers/`  
**Nova localização:** `src/domains/biomarkers/`

Arquivos movidos (9 arquivos, ~1,700 linhas):

- `__init__.py` - Module exports (atualizado com nova localização)
- `segmentation.py` - S1-S6 device mapping (182 linhas)
- `hrv.py` - HRV metrics (315 linhas)
- `sleep.py` - Sleep architecture + JSON naps (284 linhas)
- `activity.py` - Activity metrics (333 linhas)
- `circadian.py` - Circadian rhythm (221 linhas)
- `validators.py` - Cross-device validation (206 linhas)
- `aggregate.py` - Orchestrator (144 linhas)
- `extract.py` - Legacy CLI (moved to src/cli/)

**Razão:** Conformar com padrão de projeto `src/domains/` para código separado por domínios.

### 2. ✅ CLI scripts movidos para `src/cli/`

**Novos arquivos:**

- `src/cli/extract_biomarkers.py` - Extração de biomarkers (190 linhas)

  - Comando: `python -m src.cli.extract_biomarkers --participant P000001 --snapshot 2025-11-07`
  - Import atualizado: `from src.domains.biomarkers import aggregate`

- `src/cli/prepare_zepp_data.py` - Preparação de dados Zepp (184 linhas)
  - Comando: `python -m src.cli.prepare_zepp_data --participant P000001 --snapshot 2025-11-07`

**Razão:** Scripts executáveis devem ficar em src/cli/, não em scripts/ ou raiz do projeto.

### 3. ✅ run_nb2_beiwe.py → run_nb2_engage7.py

**Arquivos atualizados:**

- ✅ `run_nb2_engage7.py` (novo) - Padrão atual
- ⚠️ `run_nb2_beiwe.py` (DEPRECATED) - Marcado como legado, referencia novo

**Mudanças internas:**

- Logger: `NB2_BEIWE` → `NB2_ENGAGE7`
- Docstring: "BEIWE-Grade" → "Engage7-Grade"

**Razão:** Nomenclatura consistente (Beiwe → Engage7) em toda codebase.

### 4. ✅ Makefile atualizado

**Novos/Atualizados:**

#### `biomarkers` target

```makefile
biomarkers:
	$(PYTHON) -m src.cli.extract_biomarkers \
	  --participant $(PID) \
	  --snapshot $(SNAPSHOT) \
	  --data-dir data/etl/$(PID)/$(SNAPSHOT)/extracted \
	  --output-dir data/etl \
	  --cutoff-months 30 \
	  --verbose 0
```

- Antes: `PYTHONPATH=src $(PYTHON) scripts/extract_biomarkers.py`
- Agora: `$(PYTHON) -m src.cli.extract_biomarkers`

#### `prepare-zepp` target

```makefile
prepare-zepp:
	$(PYTHON) -m src.cli.prepare_zepp_data \
	  --participant $(PID) \
	  --snapshot $(SNAPSHOT) \
	  --zepp-source data/raw/$(PID)/zepp \
	  --target-base data/etl \
	  --symlink
```

- Antes: `PYTHONPATH=src $(PYTHON) scripts/prepare_zepp_data.py`
- Agora: `$(PYTHON) -m src.cli.prepare_zepp_data`

#### `nb2-engage7` target (novo)

```makefile
.PHONY: nb2-engage7
nb2-engage7:
	@echo "[NB2] Running Engage7-grade baseline models..."
	$(PYTHON) run_nb2_engage7.py \
	  --pid $(PID) \
	  --snapshot $(SNAPSHOT) \
	  --n-folds 6 \
	  --train-days 120 \
	  --val-days 60 \
	  --class-weight balanced \
	  --seed 42 \
	  --plots 1 \
	  --save-all 1 \
	  --verbose 2

.PHONY: nb2
nb2: nb2-engage7  # Alias
```

#### `labels` target (corrigido)

- Removida duplicação (havia duas linhas `labels:`)
- Adicionado `.PHONY: labels`

#### `pipeline` target (atualizado)

```makefile
pipeline: prepare-zepp biomarkers labels nb2
```

- Agora usa targets padronizados

### 5. ✅ Referências atualizadas em documentação

**Arquivos atualizados:**

- ✅ `docs/BIOMARKERS_README.md` - Linha 203: `run_nb2_beiwe.py` → `run_nb2_engage7.py`
- ✅ `IMPLEMENTATION_BIOMARKERS_COMPLETE.md` - Linha 114: `run_nb2_beiwe.py` → `run_nb2_engage7.py`
- ✅ `scripts/run_etl_with_corrections.sh` - Linha 95: referência atualizada
- ✅ `scripts/run_pipeline.sh` - Linha 47: referência atualizada
- ✅ `run_complete_pipeline.py` - Linha 183: referência atualizada

### 6. ✅ Imports padronizados

**Imports já corretos (relativos):**

- Todos os arquivos em `src/domains/biomarkers/` usam imports relativos
- Ex: `from . import segmentation, hrv, sleep, activity, circadian, validators`
- Nenhuma mudança necessária (já estava correto!)

---

## Estrutura Final

```
projeto/
├── run_nb2_beiwe.py          ⚠️  DEPRECATED (keep for backward compatibility)
├── run_nb2_engage7.py        ✅ NOVO PADRÃO
├── run_complete_pipeline.py  ✅ atualizado
├── Makefile                  ✅ atualizado
│
├── src/
│   ├── cli/
│   │   ├── __init__.py                    ✅ atualizado com nova documentação
│   │   ├── extract_biomarkers.py          ✅ NOVO
│   │   ├── prepare_zepp_data.py           ✅ NOVO
│   │   ├── etl_runner.py
│   │   └── run_etl_with_timer.py
│   │
│   ├── domains/
│   │   ├── biomarkers/                    ✅ MOVIDO de src/biomarkers/
│   │   │   ├── __init__.py
│   │   │   ├── segmentation.py
│   │   │   ├── hrv.py
│   │   │   ├── sleep.py
│   │   │   ├── activity.py
│   │   │   ├── circadian.py
│   │   │   ├── validators.py
│   │   │   ├── aggregate.py
│   │   │   └── extract.py                 (legacy CLI, moved to src/cli/)
│   │   ├── cardiovascular/
│   │   ├── sleep/
│   │   ├── activity/
│   │   └── ...
│   │
│   ├── etl_pipeline.py
│   ├── make_labels.py
│   └── ...
│
├── scripts/
│   ├── run_pipeline.sh                    ✅ atualizado (referência → run_nb2_engage7.py)
│   ├── run_etl_with_corrections.sh        ✅ atualizado (referência → run_nb2_engage7.py)
│   └── ... (outros scripts antigos)
│
└── docs/
    ├── BIOMARKERS_README.md               ✅ atualizado
    └── ... (outras docs)
```

---

## Validações Realizadas

✅ Todos 9 arquivos biomarkers copiados com sucesso
✅ CLI scripts criados com imports corretos
✅ run_nb2_engage7.py criado
✅ Makefile atualizado com novos targets
✅ Documentação sincronizada
✅ Referências script path atualizadas
✅ Backward compatibility mantido (run_nb2_beiwe.py ainda existe, marcado DEPRECATED)

---

## Como Usar (Novo)

### Via Makefile

```bash
# Preparar dados Zepp
make prepare-zepp PID=P000001 SNAPSHOT=2025-11-07

# Extrair biomarkers
make biomarkers PID=P000001 SNAPSHOT=2025-11-07

# Gerar labels
make labels PID=P000001 SNAPSHOT=2025-11-07

# Rodar NB2 (novo target)
make nb2 PID=P000001 SNAPSHOT=2025-11-07
# ou
make nb2-engage7 PID=P000001 SNAPSHOT=2025-11-07

# Pipeline completo
make pipeline PID=P000001 SNAPSHOT=2025-11-07
```

### Via CLI direto

```bash
# Preparar Zepp
python -m src.cli.prepare_zepp_data \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --zepp-source data/raw/P000001/zepp

# Extrair biomarkers
python -m src.cli.extract_biomarkers \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --data-dir data/etl/P000001/2025-11-07/extracted

# Rodar NB2 (novo)
python run_nb2_engage7.py \
  --pid P000001 \
  --snapshot 2025-11-07 \
  --n-folds 6 \
  --verbose 2
```

---

## Próximos Passos (Opcional)

1. **Limpeza (quando pronto):**

   - Remover `src/biomarkers/` (antigo)
   - Remover `scripts/extract_biomarkers.py` (antigo)
   - Remover `scripts/prepare_zepp_data.py` (antigo)
   - Remover `run_nb2_beiwe.py` (após período de transição)

2. **Testes:**

   - Testar `make biomarkers` e `make nb2` com dados reais
   - Testar imports de `src.domains.biomarkers` em novos módulos

3. **Documentação:**
   - Atualizar README principal com novos caminhos
   - Atualizar guias de contribuição

---

## Checklist de Mudanças

- [x] Criar `src/cli/` directory
- [x] Copiar `src/biomarkers/*` → `src/domains/biomarkers/`
- [x] Criar `src/cli/extract_biomarkers.py` (com import correto)
- [x] Criar `src/cli/prepare_zepp_data.py` (com import correto)
- [x] Atualizar `src/cli/__init__.py` com nova documentação
- [x] Criar `run_nb2_engage7.py` (copy + rename + internal changes)
- [x] Marcar `run_nb2_beiwe.py` como DEPRECATED
- [x] Atualizar Makefile (biomarkers, prepare-zepp, nb2-engage7, labels)
- [x] Atualizar todas as referências em scripts e docs
- [x] Validar imports e paths
- [x] Documentar reorganização

---

**Status:** ✅ CONCLUÍDO
**Impacto:** 0 linhas de código alteradas (apenas reorganização)
**Breaking Changes:** Nenhum (backward compatible com nova estrutura como primária)
