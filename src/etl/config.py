from importlib import import_module as _imp
_module = _imp("src.domains.config")
globals().update({k: getattr(_module, k) for k in dir(_module) if not k.startswith("_")})
