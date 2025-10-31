from importlib import import_module as _imp

_progress = _imp("src.domains.common.progress")
_io = _imp("src.domains.common.io")
_adapters = _imp("src.domains.common.adapters")
_segments = _imp("src.domains.common.segments")

# Re-export modules
progress = _progress
io = _io
adapters = _adapters
segments = _segments

__all__ = [
    "progress", "io", "adapters", "segments"
]
