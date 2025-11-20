import os


def detect_env():
    """Detect environment and available backends.

    Returns dict: {is_kaggle, data_root, out_root, backend, tf_available, torch_available}
    """
    is_kaggle = os.path.exists("/kaggle/input") or bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    )
    data_root = "/kaggle/input" if is_kaggle else "./data/ai/local"
    out_root = "/kaggle/working/outputs/ML6" if is_kaggle else "notebooks/outputs/ML6"

    tf_available = False
    torch_available = False
    try:
        import tensorflow as _tf  # noqa

        tf_available = True
    except Exception:
        tf_available = False
    try:
        import torch  # noqa

        torch_available = True
    except Exception:
        torch_available = False

    # backend policy
    if is_kaggle:
        backend = "tf" if tf_available else ("none")
    else:
        if tf_available:
            backend = "tf"
        elif torch_available:
            backend = "torch"
        else:
            backend = "none"

    return {
        "is_kaggle": bool(is_kaggle),
        "data_root": data_root,
        "out_root": out_root,
        "backend": backend,
        "tf_available": tf_available,
        "torch_available": torch_available,
    }
