"""Zepp cardio loader (stub)."""
from typing import Optional
import pandas as pd
from etl_modules.common.adapters import ProviderContext, HRProvider, register_provider

class ZeppCardio(HRProvider):
    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        return None
    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        return None

register_provider('cardio', 'zepp', ZeppCardio())

