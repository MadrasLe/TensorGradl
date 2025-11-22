import numpy as np
import time
from minigrad.accelerator import matmul_cpp

def benchmark_cpp():
    print("--- Benchmark: NumPy vs Custom C++ ---")
    
    size = 512
    print(f"Matrix size: {size}x{size}")
    
    # Generate data
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # 1. NumPy (Optimized BLAS)
    print("1. Running NumPy (OpenBLAS)...")
    t0 = time.time()
    C_np = np.dot(A, B)
    t_np = time.time() - t0
    print(f"   Time NumPy: {t_np:.4f}s")
    
    # 2. Custom C++ (Naive)
    print("2. Running Custom C++ (Naive)...")
    t0 = time.time()
    C_cpp = matmul_cpp(A, B)
    t_cpp = time.time() - t0
    print(f"   Time C++:   {t_cpp:.4f}s")
    
    # Compare
    diff = np.abs(C_np - C_cpp).sum()
    print(f"\nTotal absolute difference: {diff:.4f}")
    
    if diff < 1e-3 * (size * size): # Rough tolerance
        print("SUCCESS: Results match!")
    else:
        print("WARNING: Results diverge significantly.")
        
    print(f"\nNumPy is {t_cpp/t_np:.2f}x faster than naive C++.")

if __name__ == "__main__":
    benchmark_cpp()
