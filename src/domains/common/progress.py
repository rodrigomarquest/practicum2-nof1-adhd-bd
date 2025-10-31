# etl_modules/common/progress.py
from __future__ import annotations
import io, os, time
from contextlib import contextmanager
from typing import Optional, IO

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # fallback sem dependência

class Timer:
    def __init__(self, label: str = "task"):
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0
    def __enter__(self):
        self.start = time.perf_counter()
        print(f"▶ {self.label} ...")
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        status = "OK" if exc is None else "ERROR"
        print(f"⏱ {self.label}: {self.elapsed:.2f}s [{status}]")

class _NoOpBar:
    def __init__(self, total: Optional[int], desc: str):
        self.total = total; self.desc = desc
    def update(self, _n: int):
        # intentionally a no-op when tqdm is not available
        return None
    def close(self):
        # intentionally a no-op when tqdm is not available
        return None
    def __enter__(self):
        return self
    def __exit__(self, *args):
        # no cleanup required for the no-op bar
        return None

@contextmanager
def progress_bar(total: Optional[int], desc: str = "", unit: str = "B"):
    """Context manager that returns a progress bar.

    unit: unit string for tqdm. Default is 'B' (bytes) to preserve
    backward compatibility. If unit == 'B' then unit_scale is True.
    Callers that update by item count should pass unit='items'.
    """
    if tqdm is None:
        bar = _NoOpBar(total, desc)
        yield bar
    else:
        # Explicitly call tqdm with the parameters we need to avoid mypy
        # confusion from dynamic kwargs. For bytes mode enable scaling.
        if unit == "B":
            with tqdm(total=total, desc=desc, unit="B", unit_scale=True) as bar:
                yield bar
        else:
            with tqdm(total=total, desc=desc, unit=unit, unit_scale=False) as bar:
                yield bar

class ProgressFile(io.BufferedReader):
    """File-like que atualiza a barra a cada leitura."""
    def __init__(self, raw: IO[bytes], bar):
        super().__init__(raw)  # type: ignore
        self._bar = bar
    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        b = super().read(size)
        if b:
            self._bar.update(len(b))
        return b
    def readline(self, size: int = -1) -> bytes:  # type: ignore[override]
        b = super().readline(size)
        if b:
            self._bar.update(len(b))
        return b
    def readinto(self, b) -> int:  # type: ignore[override]
        n = super().readinto(b)
        if n and n > 0:
            self._bar.update(n)
        return n

@contextmanager
def progress_open(path: str | os.PathLike[str], desc: str = "Reading file"):
    """Abre arquivo binário com barra de progresso baseada no tamanho."""
    total = None
    try:
        total = os.path.getsize(path)
    except Exception:
        total = None
    # progress_open is explicitly byte-oriented
    with progress_bar(total=total, desc=desc, unit="B") as bar:
        with open(path, "rb") as f:
            pf = ProgressFile(f, bar)
            yield pf
