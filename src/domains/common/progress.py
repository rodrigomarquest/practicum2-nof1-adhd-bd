# etl_modules/common/progress.py
from __future__ import annotations
import io
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional, IO

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # fallback sem dependência


def _should_show_tqdm() -> bool:
    """Determine if tqdm should be shown.
    
    Checks environment variables and TTY detection with fallback:
    - ETL_TQDM=1 forces display
    - ETL_TQDM=0 disables
    - CI environment disables
    - Otherwise check if stdout is a TTY (with fallback for Git Bash/MSYS)
    """
    # Explicit force enable/disable
    if os.getenv("ETL_TQDM") == "1":
        return True
    if os.getenv("ETL_TQDM") == "0":
        return False
    
    # Disable in CI environments
    if os.getenv("CI"):
        return False
    
    # Check if stdout is a TTY
    # On Windows/Git Bash, isatty() may return False even in an interactive terminal
    # So we add additional heuristics: check for common Git Bash/MSYS env vars
    try:
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            return True
    except Exception:
        pass
    
    # Git Bash / MSYS2 detection: check for MSYSTEM or TERM env vars
    if os.getenv("MSYSTEM") or os.getenv("TERM"):
        # Likely in an interactive terminal even if isatty() fails
        return True
    
    return False


class Timer:
    def __init__(self, label: str = "task"):
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        # ASCII-only prefix
        print(f">>> {self.label} ...")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        status = "OK" if exc is None else "ERROR"
        # ASCII-only summary
        print(f"[OK] {self.label}: {self.elapsed:.2f}s [{status}]")


class _NoOpBar:
    def __init__(self, total: Optional[int], desc: str):
        self.total = total
        self.desc = desc

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
    
    Progress bar visibility is controlled by:
    - ETL_TQDM=1 (force show)
    - ETL_TQDM=0 (force hide)
    - Automatic detection via isatty() with Git Bash/MSYS2 fallback
    """
    if tqdm is None:
        bar = _NoOpBar(total, desc)
        yield bar
    else:
        # Decide whether to show the bar based on environment and TTY detection
        disable = not _should_show_tqdm()
        
        # Explicitly call tqdm with the parameters we need to avoid mypy
        # confusion from dynamic kwargs. For bytes mode enable scaling.
        if unit == "B":
            with tqdm(total=total, desc=desc, unit="B", unit_scale=True, disable=disable) as bar:
                yield bar
        else:
            with tqdm(total=total, desc=desc, unit=unit, unit_scale=False, disable=disable) as bar:
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
