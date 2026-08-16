#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int CtaM = 128;
constexpr int CtaN = 128;
constexpr int CtaK = 32;
constexpr int ThreadsX = 32;
constexpr int ThreadsY = 8;
constexpr int ThreadM = 16;
constexpr int ThreadN = 4;
constexpr int ThreadsPerBlock = ThreadsX * ThreadsY;
constexpr int LoadsPerThread = (CtaM * CtaK) / ThreadsPerBlock;
constexpr size_t SharedBytes =
    (static_cast<size_t>(CtaM) * CtaK + static_cast<size_t>(CtaK) * CtaN) * sizeof(double);

__global__ void matmul_kernel(const double* A, const double* B, double* C, int m, int n, int k)
{
    extern __shared__ double shared[];
    double* As = shared;
    double* Bs = shared + static_cast<size_t>(CtaM) * CtaK;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int tile_row = blockIdx.y * CtaM;
    const int tile_col = blockIdx.x * CtaN;
    const int row = tile_row + ty * ThreadM;
    const int col = tile_col + tx * ThreadN;

    double sum[ThreadM][ThreadN] = {};

    for (int t = 0; t < k; t += CtaK)
    {
        #pragma unroll
        for (int i = 0; i < LoadsPerThread; ++i) {
            const int load_idx = i * ThreadsPerBlock + tid;

            const int a_row = load_idx >> 5;
            const int a_col = load_idx & (CtaK - 1);
            As[a_row * CtaK + a_col] = A[(tile_row + a_row) * k + t + a_col];

            const int b_row = load_idx >> 7;
            const int b_col = load_idx & (CtaN - 1);
            Bs[b_row * CtaN + b_col] = B[(t + b_row) * n + tile_col + b_col];
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < CtaK; ++kk) {
            const double b0 = Bs[kk * CtaN + tx * ThreadN + 0];
            const double b1 = Bs[kk * CtaN + tx * ThreadN + 1];
            const double b2 = Bs[kk * CtaN + tx * ThreadN + 2];
            const double b3 = Bs[kk * CtaN + tx * ThreadN + 3];

            #pragma unroll
            for (int i = 0; i < ThreadM; ++i) {
                const double a = As[(ty * ThreadM + i) * CtaK + kk];
                sum[i][0] += a * b0;
                sum[i][1] += a * b1;
                sum[i][2] += a * b2;
                sum[i][3] += a * b3;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < ThreadM; ++i) {
        #pragma unroll
        for (int j = 0; j < ThreadN; ++j) {
            C[(row + i) * n + col + j] = sum[i][j];
        }
    }
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

    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(SharedBytes));

    dim3 block(ThreadsX, ThreadsY);
    dim3 grid(n / CtaN, m / CtaM);
    matmul_kernel<<<grid, block, SharedBytes>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
