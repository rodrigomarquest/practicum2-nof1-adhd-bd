# Notebooks Refactoring Summary (NB2→ML6, NB3→ML7)

**Data**: 2025-11-20
**Objetivo**: Atualizar conteúdo interno dos notebooks Jupyter para refletir refatoração NB2→ML6, NB3→ML7, mantendo nomes dos arquivos `.ipynb` inalterados

## Notebooks Atualizados

### ✅ NB0_DataRead.ipynb
**Mudanças**:
- Stage descriptions: "Stage 5: Prep NB2" → "Stage 5: Prep ML6"
- Stage descriptions: "Stage 6: NB2" → "Stage 6: ML6 (static classifier)"
- Stage descriptions: "Stage 7: NB3" → "Stage 7: ML7 (LSTM sequence)"
- Paths: `ai/nb2/` → `ai/ml6/`, `ai/nb3/` → `ai/ml7/`
- Files: `features_daily_nb2.csv` → `features_daily_ml6.csv`

**Verificações atualizadas**:
```python
paths["ai_base"] / "ml6" / "features_daily_ml6.csv"  # Stage 5
paths["ai_base"] / "ml6" / "cv_summary.json"         # Stage 6
paths["ai_base"] / "ml7" / "shap_summary.md"         # Stage 7
paths["ai_base"] / "ml7" / "drift_report.md"
paths["ai_base"] / "ml7" / "lstm_report.md"
```

### ✅ NB1_EDA.ipynb
**Mudanças**:
- Next steps section updated:
  - "Proceed to NB2_Baseline.ipynb" → "(Stage 6: ML6 static classifier)"
  - "Proceed to NB3_DeepLearning.ipynb" → "(Stage 7: ML7 LSTM)"

**Rationale**: NB1 foca em EDA, tem poucas referências a stages avançados, apenas em "próximos passos"

### ✅ NB2_Baseline.ipynb
**Mudanças principais**:
- **Header**: Adicionada nota explicativa sobre naming convention
- **Variáveis**: `NB2_DIR` → `ML6_DIR`, `nb2_files` → `ml6_files`
- **Paths**: `"nb2"` → `"ml6"`, `nb2/` → `ml6/`
- **Descrições**: "NB2" → "ML6 (Stage 6)" em código/comentários
- **Markdowns**: "Load NB2 Results" → "Load ML6 Results (Stage 6)"
- **Error messages**: Atualizados para referenciar `make ml6`

**Nota no header**:
> This notebook uses the filename `NB2` for historical continuity, but internally refers to **Stage 6 (ML6)** following the refactoring to distinguish modeling stages from Jupyter notebook numbering.

**Exemplo de mudança**:
```python
# Before:
AI_BASE = REPO_ROOT / "data" / "ai" / PARTICIPANT / SNAPSHOT
NB2_DIR = AI_BASE / "nb2"
print(f"NB2 outputs: {NB2_DIR}")

# After:
AI_BASE = REPO_ROOT / "data" / "ai" / PARTICIPANT / SNAPSHOT
ML6_DIR = AI_BASE / "ml6"  # Stage 6: Static daily classifier (formerly nb2)
print(f"ML6 outputs (Stage 6): {ML6_DIR}")
```

### ✅ NB3_DeepLearning.ipynb
**Mudanças principais**:
- **Header**: Adicionada nota explicativa sobre naming convention
- **Variáveis**: `NB3_DIR` → `ML7_DIR`, `nb3_files` → `ml7_files`
- **Paths**: `"nb3"` → `"ml7"`, `nb3/` → `ml7/`
- **Imports**: `nb3_analysis` → `ml7_analysis`
- **Constantes**: `NB3_FEATURE_COLS` → `ML7_FEATURE_COLS`
- **Descrições**: "NB3" → "ML7 (Stage 7)" em código/comentários
- **Markdowns**: Atualizado para referenciar ML7, LSTM, Stage 7

**Nota no header**:
> This notebook uses the filename `NB3` for historical continuity, but internally refers to **Stage 7 (ML7)** following the refactoring to distinguish modeling stages from Jupyter notebook numbering.

**Exemplo de mudança**:
```python
# Before:
from src.etl.nb3_analysis import prepare_nb3_features, NB3_FEATURE_COLS
ML7_DIR = AI_BASE / "nb3"

# After:
from src.etl.ml7_analysis import prepare_ml7_features, ML7_FEATURE_COLS
ML7_DIR = AI_BASE / "ml7"  # Stage 7: LSTM sequence classifier
```

## Preservado

### ❌ Nomes dos Arquivos `.ipynb`
- `notebooks/NB0_DataRead.ipynb` (inalterado)
- `notebooks/NB1_EDA.ipynb` (inalterado)
- `notebooks/NB2_Baseline.ipynb` (inalterado)
- `notebooks/NB3_DeepLearning.ipynb` (inalterado)

**Rationale**: 
- Manter continuidade histórica
- Evitar quebrar links/referências em docs
- Usuários já familiarizados com numeração NB0-NB3
- Notas explicativas deixam claro mapeamento: NB2→ML6, NB3→ML7

### ❌ Notebooks Archive
- `notebooks/archive/` não foi modificado (histórico preservado)

## Consistência Verificada

### Código Python
✅ Todas as variáveis `NB2_DIR`, `NB3_DIR` → `ML6_DIR`, `ML7_DIR`  
✅ Todos os paths `"nb2/"`, `"nb3/"` → `"ml6/"`, `"ml7/"`  
✅ Todos os imports `nb3_analysis` → `ml7_analysis`  
✅ Todas as constantes `NB3_FEATURE_COLS` → `ML7_FEATURE_COLS`

### Markdown
✅ Stage descriptions consistentes com pipeline  
✅ Notas explicativas nos notebooks principais (NB2/NB3)  
✅ Referências "Next steps" atualizadas (NB1)

### Alinhamento com Código
✅ Paths no notebook apontam para `ml6/`, `ml7/` (diretórios renomeados)  
✅ Imports usam `ml7_analysis` (módulo renomeado)  
✅ Arquivos CSV: `features_daily_ml6.csv` (arquivo renomeado)  
✅ Makefile targets: notebooks referenciam `make ml6`, `make ml7`

## Testando os Notebooks

### NB0_DataRead.ipynb
```bash
# Verificar status do pipeline
jupyter notebook notebooks/NB0_DataRead.ipynb
# Run All Cells → Deve mostrar ML6/ML7 em stage checks
```

### NB2_Baseline.ipynb
```bash
# Após rodar pipeline com Stage 6
make ml6 PARTICIPANT=P000001 SNAPSHOT=2025-11-07
jupyter notebook notebooks/NB2_Baseline.ipynb
# Deve carregar de data/ai/P000001/2025-11-07/ml6/
```

### NB3_DeepLearning.ipynb
```bash
# Após rodar pipeline com Stage 7
make ml7 PARTICIPANT=P000001 SNAPSHOT=2025-11-07
jupyter notebook notebooks/NB3_DeepLearning.ipynb
# Deve carregar de data/ai/P000001/2025-11-07/ml7/
```

## Compatibilidade

### Backward Compatibility
⚠️ **Breaking change**: Notebooks agora esperam diretórios `ml6/`, `ml7/` ao invés de `nb2/`, `nb3/`

**Solução**:
- Rodar pipeline completo (já refatorado): `make pipeline`
- Ou criar symlinks temporários (não recomendado):
  ```bash
  cd data/ai/P000001/2025-11-07/
  ln -s ml6 nb2  # Temporário
  ln -s ml7 nb7  # Temporário
  ```

### Forward Compatibility
✅ Novos runs do pipeline criam `ml6/`, `ml7/`  
✅ Notebooks funcionam out-of-the-box com novos dados

## Documentação Relacionada

- `REFACTORING_NB2_NB3_TO_ML6_ML7.md` - Refactoring completo (código + docs)
- `README.md` - Atualizado com ML6/ML7
- `pipeline_overview.md` - Stage descriptions atualizadas
- `docs/NB2_PIPELINE_README.md` → `docs/ML6_PIPELINE_README.md` (renomeado)
- `docs/NB3_QUICK_REFERENCE.md` → `docs/ML7_QUICK_REFERENCE.md` (renomeado)

## Próximos Passos (Opcional)

Se quiser renomear os arquivos `.ipynb` no futuro:
```bash
git mv notebooks/NB2_Baseline.ipynb notebooks/ML6_Static_Classifier.ipynb
git mv notebooks/NB3_DeepLearning.ipynb notebooks/ML7_LSTM_Sequence.ipynb
# Atualizar docs/ e README.md com novos nomes
```

Por agora, manter naming histórico com notas explicativas claras é suficiente.

---

**Status**: ✅ Notebooks internamente consistentes com refatoração ML6/ML7  
**Testado**: ⏳ Aguardando test run do pipeline  
**Commitado**: ⏳ Pending (incluir com commit principal de refatoração)
