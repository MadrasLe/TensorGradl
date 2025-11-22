import ctypes
import numpy as np
import os

# Path to the shared library
lib_path = os.path.join(os.path.dirname(__file__), 'csrc', 'libmatmul.so')

# Load library
try:
    lib = ctypes.CDLL(lib_path)
    
    # Define function signature
    # void c_matmul(float* A, float* B, float* C, int M, int K, int N)
    lib.c_matmul.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # A
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # B
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # C
        ctypes.c_int, # M
        ctypes.c_int, # K
        ctypes.c_int  # N
    ]
except OSError:
    print(f"Warning: Could not load C++ extension at {lib_path}")
    lib = None

def matmul_cpp(a, b):
    """
    Performs matrix multiplication using the custom C++ extension.
    Inputs must be numpy arrays. Returns a numpy array.
    """
    if lib is None:
        raise RuntimeError("C++ extension not loaded.")
        
    # Ensure inputs are float32 (C++ expects float)
    # This might involve a copy if they are float64
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {a.shape} vs {b.shape}"
    
    # Create output array
    c = np.zeros((M, N), dtype=np.float32)
    
    # Call C++ function
    lib.c_matmul(a, b, c, M, K, N)
    
    return c
