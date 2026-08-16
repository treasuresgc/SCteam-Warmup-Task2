#include <cuda_runtime.h>

#include <cstddef>

#include <mma.h>
using namespace nvcuda;

constexpr int ThreadsX = 32;
constexpr int ThreadsY = 16;
constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 4;
constexpr int CtaM = WMMA_M;
constexpr int CtaN = ThreadsY * WMMA_N;
constexpr int CtaK = 16;
constexpr int ThreadsPerBlock = ThreadsX * ThreadsY;

namespace {

__global__ void matmul_kernel(const double* A, const double* B, double* C, int m, int n, int k)
{
    __shared__ double As[CtaM][CtaK];
    __shared__ double Bs[CtaK][CtaN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int tile_m = blockIdx.y * CtaM;
    const int tile_n = blockIdx.x * CtaN;
    const int warp_n = ty * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> c_frag;

    wmma::fill_fragment(c_frag, 0.0);

    for (int tile_k = 0; tile_k < k; tile_k += CtaK) {
        for (int idx = tid; idx < CtaM * CtaK; idx += ThreadsPerBlock) {
            const int row = idx / CtaK;
            const int col = idx % CtaK;
            As[row][col] = A[static_cast<size_t>(tile_m + row) * k + tile_k + col];
        }

        for (int idx = tid; idx < CtaK * CtaN; idx += ThreadsPerBlock) {
            const int row = idx / CtaN;
            const int col = idx % CtaN;
            Bs[row][col] = B[static_cast<size_t>(tile_k + row) * n + tile_n + col];
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < CtaK; kk += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &As[0][kk], CtaK);
            wmma::load_matrix_sync(b_frag, &Bs[kk][warp_n], CtaN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    wmma::store_matrix_sync(C + static_cast<size_t>(tile_m) * n + tile_n + warp_n,
                            c_frag,
                            n,
                            wmma::mem_row_major);
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

    dim3 block(ThreadsX, ThreadsY);
    dim3 grid(n / CtaN, m / CtaM);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
