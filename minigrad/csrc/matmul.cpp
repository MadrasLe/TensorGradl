#include <iostream>

extern "C" {
    // A function callable from Python via ctypes
    // A: (M x K), B: (K x N), Output C: (M x N)
    void c_matmul(float* A, float* B, float* C, int M, int K, int N) {
        
        // Naive O(N^3) loop
        // Not optimized for Cache or SIMD, purely educational
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = acc;
            }
        }
    }
}
