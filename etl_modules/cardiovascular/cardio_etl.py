"""Cardio ETL orchestrator (stub)."""
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

