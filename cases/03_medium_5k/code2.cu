#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int CtaM = 8;
constexpr int CtaN = 32;
constexpr int CtaK = 32;

__global__ void matmul_kernel(const double* A, const double* B, double* C, int m, int n, int k)
{
    __shared__ double As[CtaM][CtaK], Bs[CtaK][CtaN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * CtaM + ty;
    const int col = blockIdx.x * CtaN + tx;
    double sum = 0.0;

    for (int t = 0; t < k; t += CtaK)
    {
        As[ty][tx] = A[row * k + t + tx];

        #pragma unroll
        for (int i = ty; i < CtaK; i += CtaM) {
            Bs[i][tx] = B[(t + i) * n + col];
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < CtaK; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

}  // namespace

void run(double* A, double* B, double* C, int m, int n, int k)
{
    const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(double);
    const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(double);
    const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(double);

    double* d_A;
    double* d_B;
    double* d_C;
    cudaMalloc(&d_A, bytes_a);
    cudaMalloc(&d_B, bytes_b);
    cudaMalloc(&d_C, bytes_c);

    cudaMemcpy(d_A, A, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_b, cudaMemcpyHostToDevice);

    dim3 block(CtaN, CtaM);
    dim3 grid(n / CtaN, m / CtaM);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
