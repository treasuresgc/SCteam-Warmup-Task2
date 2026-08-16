#include <cuda_runtime.h>

#include <cstdio>
#include <cstddef>

namespace {

__global__ void core(const double* A, const double* B, double* C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}


void run_cuda(const double* A, const double* B, double* C, int m, int n, int k)
{
    size_t dbl = sizeof(double);
    size_t siza = static_cast<size_t>(m) * k * dbl;
    size_t sizb = static_cast<size_t>(k) * n * dbl;
    size_t sizc = static_cast<size_t>(m) * n * dbl;
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    if (cudaMalloc(&d_A, siza) != cudaSuccess ||
        cudaMalloc(&d_B, sizb) != cudaSuccess ||
        cudaMalloc(&d_C, sizc) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        printf("Failed to allocate device memory\n");
        return;
    }

    bool ok = true;
    ok = ok && cudaMemcpy(d_A, A, siza, cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemcpy(d_B, B, sizb, cudaMemcpyHostToDevice) == cudaSuccess;

    if (ok) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        core<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
        ok = cudaGetLastError() == cudaSuccess;
    }

    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
    ok = ok && cudaMemcpy(C, d_C, sizc, cudaMemcpyDeviceToHost) == cudaSuccess;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}  // namespace

void run(double* A, double* B, double* C, int m, int n, int k)
{
    run_cuda(A, B, C, m, n, k);
}
