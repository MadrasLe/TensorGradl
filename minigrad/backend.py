import numpy as np
import os

# Default to NumPy
xp = np
backend_name = 'cpu'

try:
    # Try to import cupy if available and not disabled by env var
    if os.getenv("MINIGRAD_DEVICE", "cpu") == "cuda":
        import cupy as cp
        xp = cp
        backend_name = 'cuda'
except ImportError:
    pass

def get_array_module(x):
    """Returns the module (numpy or cupy) for the array x."""
    return xp.get_array_module(x) if backend_name == 'cuda' else np

def to_cpu(x):
    """Moves array x to CPU."""
    if backend_name == 'cuda' and isinstance(x, xp.ndarray):
        return xp.asnumpy(x)
    return x

def to_device(x):
    """Moves array x to the configured device (CPU or GPU)."""
    return xp.array(x)
