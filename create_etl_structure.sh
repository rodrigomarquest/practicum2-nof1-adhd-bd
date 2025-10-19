#!/usr/bin/env bash
# ======================================================
# create_etl_structure.sh
# Cria estrutura modular de ETL (common / cardiovascular / apple / zepp / _join)
# sem sobrescrever arquivos existentes.
# ======================================================

set -euo pipefail

write_if_missing() {
  local file="$1"
  shift
  if [ -e "$file" ]; then
    echo "SKIP:   $file (já existe)"
  else
    mkdir -p "$(dirname "$file")"
    echo "$1" > "$file"
    echo "CREATED:$file"
  fi
}

echo "🧩 Criando estrutura modular de ETL..."

# -------------------------
# Diretórios principais
# -------------------------
mkdir -p etl_modules/common \
         etl_modules/cardiovascular/apple \
         etl_modules/cardiovascular/zepp \
         etl_modules/cardiovascular/_join

# -------------------------
# Arquivos comuns
# -------------------------
write_if_missing "etl_modules/common/__init__.py" "\"\"\"Common helpers for ETL modules.\"\"\""
write_if_missing "etl_modules/cardiovascular/__init__.py" "\"\"\"Cardiovascular ETL domain.\"\"\""
write_if_missing "etl_modules/cardiovascular/apple/__init__.py" "\"\"\"Apple providers for cardio domain.\"\"\""
write_if_missing "etl_modules/cardiovascular/zepp/__init__.py" "\"\"\"Zepp providers for cardio domain.\"\"\""
write_if_missing "etl_modules/cardiovascular/_join/__init__.py" "\"\"\"Cross-platform join logic for cardio domain.\"\"\""

# -------------------------
# Config
# -------------------------
write_if_missing "etl_modules/config.py" "from dataclasses import dataclass

@dataclass
class EtlCfg:
    tz: str = \"Europe/Dublin\"
    low_coverage_pct: float = 40.0

@dataclass
class CardioCfg(EtlCfg):
    hr_min_bpm: int = 35
    hr_max_bpm: int = 220
    max_ffill_minutes: int = 5
    spike_bpm_per_min: int = 40
"

# -------------------------
# Common modules
# -------------------------
write_if_missing "etl_modules/common/io.py" "\"\"\"Lightweight I/O utilities (stubs).\"\"\"
import os, warnings, pandas as pd

def read_csv_if_exists(path: str, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, **kwargs) if os.path.exists(path) else None
    except Exception as e:
        warnings.warn(f'read_csv_if_exists falhou: {e}')
        return None

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def to_local_dt(series, tz: str):
    s = pd.to_datetime(series, errors='coerce', utc=True)
    try:
        return s.dt.tz_convert(tz)
    except Exception:
        return s

def date_col(series): return pd.to_datetime(series).dt.date
"

write_if_missing "etl_modules/common/segments.py" "\"\"\"Segment helpers (S1–S6).\"\"\"
import os, pandas as pd
from .io import read_csv_if_exists

def load_segments(snapshot_dir: str):
    path = os.path.join(snapshot_dir, 'version_log_enriched.csv')
    df = read_csv_if_exists(path)
    if df is None or df.empty: return None
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def attach_segment(df, segments):
    if df is None or df.empty or segments is None or segments.empty:
        if df is None: return df
        df = df.copy()
        df['segment_id'] = pd.NA
        return df
    return df.merge(segments[['date','segment_id']], on='date', how='left')
"

write_if_missing "etl_modules/common/adapters.py" "\"\"\"Provider interfaces and registry (Apple, Zepp, etc.).\"\"\"
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional
import pandas as pd

@dataclass
class ProviderContext:
    snapshot_dir: str
    tz: str = 'Europe/Dublin'

class HRProvider(Protocol):
    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]: ...
    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]: ...

_REGISTRY: Dict[str, Dict[str, Any]] = {'cardio': {}}

def register_provider(domain: str, name: str, provider: Any) -> None:
    _REGISTRY.setdefault(domain, {})[name] = provider

def get_providers(domain: str) -> Dict[str, Any]:
    return _REGISTRY.get(domain, {})
"

# -------------------------
# Cardiovascular stubs
# -------------------------
write_if_missing "etl_modules/cardiovascular/apple/loader.py" "\"\"\"Apple cardio loader (stub).\"\"\"
from typing import Optional
import pandas as pd
from etl_modules.common.adapters import ProviderContext, HRProvider, register_provider

class AppleCardio(HRProvider):
    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        return None
    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        return None

register_provider('cardio', 'apple', AppleCardio())
"

write_if_missing "etl_modules/cardiovascular/zepp/loader.py" "\"\"\"Zepp cardio loader (stub).\"\"\"
from typing import Optional
import pandas as pd
from etl_modules.common.adapters import ProviderContext, HRProvider, register_provider

class ZeppCardio(HRProvider):
    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        return None
    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        return None

register_provider('cardio', 'zepp', ZeppCardio())
"

write_if_missing "etl_modules/cardiovascular/_join/join.py" "\"\"\"Cross-platform join rules (stub).\"\"\"
import pandas as pd
def join_hr(hr_sources: list[pd.DataFrame]) -> pd.DataFrame | None: return None
def select_hrv(hrv_sources: list[pd.DataFrame]) -> pd.DataFrame | None: return None
"

write_if_missing "etl_modules/cardiovascular/cardio_features.py" "\"\"\"Cardio feature derivation (stub).\"\"\"
import pandas as pd
from etl_modules.config import CardioCfg
def derive_hr_features(hr,sleep,segments,cfg:CardioCfg)->pd.DataFrame: raise NotImplementedError
def derive_hrv_features(hrv,hr,sleep,segments,cfg:CardioCfg)->pd.DataFrame: raise NotImplementedError
def derive_circadian_features(hr,segments,cfg:CardioCfg)->pd.DataFrame: raise NotImplementedError
def merge_cardio_outputs(hr_feats,hrv_feats,circ_feats)->pd.DataFrame: raise NotImplementedError
def update_features_daily(features_cardio:pd.DataFrame,features_daily_path:str)->str: raise NotImplementedError
"

write_if_missing "etl_modules/cardiovascular/cardio_etl.py" "\"\"\"Cardio ETL orchestrator (stub).\"\"\"
import os,json,pandas as pd
from etl_modules.config import CardioCfg
from etl_modules.common.adapters import get_providers,ProviderContext
from etl_modules.common.io import read_csv_if_exists,ensure_dir
from ._join.join import join_hr,select_hrv
from .cardio_features import (derive_hr_features,derive_hrv_features,
    derive_circadian_features,merge_cardio_outputs,update_features_daily)

def run_stage_cardio(snapshot_dir:str,out_dir:str,cfg:CardioCfg)->dict:
    providers=get_providers('cardio')
    ctx=ProviderContext(snapshot_dir=snapshot_dir,tz=cfg.tz)
    hr_dfs,hrv_dfs=[],[]
    for name,prov in providers.items():
        hr_dfs.append(prov.load_hr(ctx))
        hrv_dfs.append(prov.load_hrv(ctx))
    hr_all=join_hr(hr_dfs); hrv_all=select_hrv(hrv_dfs)
    segments=read_csv_if_exists(os.path.join(snapshot_dir,'version_log_enriched.csv'))
    if segments is not None and not segments.empty:
        segments['date']=pd.to_datetime(segments['date']).dt.date
    ensure_dir(out_dir)
    out_cardio=os.path.join(out_dir,'features_cardiovascular.csv')
    out_daily=os.path.join(out_dir,'features_daily_updated.csv')
    return {'features_cardio':out_cardio,'features_daily_updated':out_daily}
"

echo "✅ Estrutura criada/atualizada com sucesso."
