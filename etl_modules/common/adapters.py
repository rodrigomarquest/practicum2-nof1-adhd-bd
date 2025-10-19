"""Provider interfaces and registry (Apple, Zepp, etc.)."""
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

