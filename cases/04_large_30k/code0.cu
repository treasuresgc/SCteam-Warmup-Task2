#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>

void run(double* A, double* B, double* C, int m, int n, int k)
{
    const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(double);
    const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(double);
    const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(double);

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    cudaMalloc(&d_A, bytes_a);
    cudaMalloc(&d_B, bytes_b);
    cudaMalloc(&d_C, bytes_c);

    cudaMemcpy(d_A, A, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_b, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED);

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                d_B,
                n,
                d_A,
                k,
                &beta,
                d_C,
                n);

    cudaMemcpy(C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
