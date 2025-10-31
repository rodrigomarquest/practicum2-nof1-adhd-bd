from dataclasses import dataclass

@dataclass
class EtlCfg:
    tz: str = "Europe/Dublin"
    low_coverage_pct: float = 40.0

@dataclass
class CardioCfg(EtlCfg):
    hr_min_bpm: int = 35
    hr_max_bpm: int = 220
    max_ffill_minutes: int = 5
    spike_bpm_per_min: int = 40

