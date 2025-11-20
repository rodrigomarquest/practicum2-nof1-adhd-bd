# Stage 7 (ML7) UnboundLocalError Fix

**Issue**: `UnboundLocalError: cannot access local variable 'df' where it is not associated with a value`

**Root Cause**: 
- Lines após o bloco `if ml6_path.exists():` estavam **fora da indentação**
- Código tentava acessar `df` na linha 627 mesmo quando o arquivo ML6 não existia
- Path incorreto: buscava `features_daily_nb2.csv` ao invés de `features_daily_ml6.csv`

**Fixes Applied**:

1. ✅ **Correção de indentação**: Movidos logs e checks para **dentro do bloco if**
2. ✅ **Adicionado else block**: Tratamento de erro quando ML6 data não existe
3. ✅ **Path corrigido**: `features_daily_nb2.csv` → `features_daily_ml6.csv`

**Before**:
```python
if ml6_path.exists():
    # ... código ...
    df = prepare_ml7_features(df_labeled)
    df = df.sort_values('date').reset_index(drop=True)

logger.info(f"[ML7] Dataset shape: {df.shape}")  # ❌ FORA do if!
```

**After**:
```python
if ml6_path.exists():
    # ... código ...
    df = prepare_ml7_features(df_labeled)
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"[ML7] Dataset shape: {df.shape}")  # ✅ DENTRO do if
    # ... mais logs e checks ...
else:
    logger.error(f"[ML7] Required ML6 data not found: {ml6_path}")
    raise FileNotFoundError(f"ML6 data not found: {ml6_path}")
```

**Testing**:
```bash
# Agora deve funcionar
make pipeline PARTICIPANT=P000001 SNAPSHOT=2025-11-07
```

**Status**: ✅ Fixed, ready to commit
