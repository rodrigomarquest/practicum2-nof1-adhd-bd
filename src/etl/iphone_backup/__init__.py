from importlib import import_module as _imp

iphone_backup = _imp("src.domains.iphone_backup.iphone_backup")
utils = _imp("src.domains.iphone_backup.utils")

__all__ = ["iphone_backup", "utils"]
