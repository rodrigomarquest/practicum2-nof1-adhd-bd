from importlib import import_module as _imp

loader = _imp("src.domains.cardiovascular.zepp.loader")

__all__ = ["loader"]
