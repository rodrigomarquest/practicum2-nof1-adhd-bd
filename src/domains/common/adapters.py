# etl_modules/common/adapters.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Adapters / Providers registry (cardio e futuros domínios).

- Define o contrato (CardioProvider) e o contexto (ProviderContext).
- Faz o registro resiliente dos providers disponíveis para cada domínio.
- Para cardio, garante o provider 'apple' (per-metric) e, se existir, 'zepp'.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd


# ----------------------------------------------------------------------
# Contexto comum passado aos providers
# ----------------------------------------------------------------------
@dataclass
class ProviderContext:
    snapshot_dir: str
    tz: Optional[str] = None


# ----------------------------------------------------------------------
# Interfaces base (por domínio)
# ----------------------------------------------------------------------
class CardioProvider:
    """Contrato mínimo para provedores do domínio 'cardio'."""

    name: str = "base"

    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        """
        Retorna DataFrame com colunas:
            ['timestamp', 'bpm']
        ou None se o provider não tiver dados.
        """
        return None

    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        """
        Retorna DataFrame com colunas:
            ['timestamp', 'val', 'metric']   (ex.: metric='hrv_ms')
        ou None se o provider não tiver dados.
        """
        return None


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def get_providers(domain: str) -> Dict[str, CardioProvider]:
    """
    Devolve dict de providers por domínio.
    Para 'cardio', registra:
      - 'apple' (obrigatório no fluxo atual; lê per-metric)
      - 'zepp'  (opcional; depende de existência do módulo)
    """
    providers: Dict[str, CardioProvider] = {}

    if domain == "cardio":
        # Apple (per-metric)
        try:
            from src.etl.cardiovascular.apple.loader import AppleCardioProvider

            providers["apple"] = AppleCardioProvider()
        except Exception as e:
            print(f"[providers] Apple provider unavailable: {e}")

        # Zepp (opcional)
        try:
            from src.etl.cardiovascular.zepp.loader import ZeppCardioProvider

            providers["zepp"] = ZeppCardioProvider()
        except Exception as e:
            print(f"[providers] Zepp provider unavailable: {e}")

    return providers
