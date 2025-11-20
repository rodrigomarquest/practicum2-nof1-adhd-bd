"""
Progress bar utilities using tqdm.

Provides consistent progress bars across the pipeline for long-running operations.
"""

from typing import Optional, Iterable, Any
import sys
from tqdm import tqdm


def create_progress_bar(
    iterable: Optional[Iterable] = None,
    total: Optional[int] = None,
    desc: str = "Processing",
    unit: str = "it",
    disable: bool = False,
    leave: bool = True,
    **kwargs
) -> tqdm:
    """
    Create a standardized progress bar.
    
    Args:
        iterable: Optional iterable to wrap
        total: Total number of iterations (if iterable is None)
        desc: Description to display
        unit: Unit name for the progress bar
        disable: If True, disable progress bar
        leave: If True, leave progress bar after completion
        **kwargs: Additional tqdm parameters
        
    Returns:
        tqdm progress bar instance
        
    Examples:
        # With iterable
        for item in create_progress_bar(items, desc="Processing items"):
            process(item)
            
        # Manual updates
        pbar = create_progress_bar(total=100, desc="Processing")
        for i in range(100):
            process(i)
            pbar.update(1)
        pbar.close()
    """
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
        leave=leave,
        file=sys.stdout,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        **kwargs
    )


def progress_wrapper(func):
    """
    Decorator to add progress bar to functions that return generators.
    
    Example:
        @progress_wrapper
        def process_files(files):
            for f in files:
                yield process(f)
    """
    def wrapper(*args, desc=None, total=None, **kwargs):
        result = func(*args, **kwargs)
        if hasattr(result, '__iter__'):
            desc = desc or f"Running {func.__name__}"
            return create_progress_bar(result, desc=desc, total=total)
        return result
    return wrapper


class ProgressContext:
    """
    Context manager for progress bars.
    
    Example:
        with ProgressContext(total=100, desc="Processing") as pbar:
            for i in range(100):
                process(i)
                pbar.update(1)
    """
    
    def __init__(self, total: int, desc: str = "Processing", **kwargs):
        self.pbar = create_progress_bar(total=total, desc=desc, **kwargs)
    
    def __enter__(self):
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()
        return False
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.pbar.update(n)
    
    def set_description(self, desc: str):
        """Update description."""
        self.pbar.set_description(desc)


def log_progress(message: str, logger=None):
    """
    Log a progress message to both tqdm and logger.
    
    Args:
        message: Message to log
        logger: Optional logger instance (if None, uses print)
    """
    tqdm.write(message)
    if logger:
        logger.info(message)
