from importlib import import_module as _imp
_mod = _imp('src.domains.cardiovascular.zepp.loader')
globals().update(_mod.__dict__)
del _imp, _mod
__all__ = [k for k in globals().keys() if not k.startswith('__')]
