#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstddef>

namespace {

__global__ void core(const float* A, const float* B, float* C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

bool check_cuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess) {
        printf("%s: %s\n", message, cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool run_cuda(const float* A, const float* B, float* C, int m, int n, int k)
{
    size_t flt = sizeof(float);
    size_t siza = static_cast<size_t>(m) * k * flt;
    size_t sizb = static_cast<size_t>(k) * n * flt;
    size_t sizc = static_cast<size_t>(m) * n * flt;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ok = true;

    ok = ok && check_cuda(cudaMalloc(&d_A, siza), "Failed to allocate device memory for A");
    ok = ok && check_cuda(cudaMalloc(&d_B, sizb), "Failed to allocate device memory for B");
    ok = ok && check_cuda(cudaMalloc(&d_C, sizc), "Failed to allocate device memory for C");
    ok = ok && check_cuda(cudaMemcpy(d_A, A, siza, cudaMemcpyHostToDevice), "Failed to copy A to device");
    ok = ok && check_cuda(cudaMemcpy(d_B, B, sizb, cudaMemcpyHostToDevice), "Failed to copy B to device");

    if (ok) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        core<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }

    ok = ok && check_cuda(cudaMemcpy(C, d_C, sizc, cudaMemcpyDeviceToHost), "Failed to copy result to host");

    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return ok;
}

}  // namespace

void print(const float* C, int m, int n)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }
}

void run(float* A, float* B, float* C, int m, int n, int k)
{
    run_cuda(A, B, C, m, n, k);
}
