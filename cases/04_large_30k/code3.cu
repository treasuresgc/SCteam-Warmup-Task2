#include <cuda_runtime.h>

#include <cstddef>

#include <mma.h>
using namespace nvcuda;

constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 4;

constexpr int CtaM = 128;
constexpr int CtaN = 128;
constexpr int CtaK = 16;

constexpr int WarpTileM = 32;
constexpr int WarpTileN = 64;
constexpr int WarpCountM = CtaM / WarpTileM;
constexpr int WarpCountN = CtaN / WarpTileN;
constexpr int WarpTilesM = WarpTileM / WMMA_M;
constexpr int WarpTilesN = WarpTileN / WMMA_N;

constexpr int ThreadsX = 32;
constexpr int ThreadsY = WarpCountM * WarpCountN;
constexpr int ThreadsPerBlock = ThreadsX * ThreadsY;

static_assert(WarpCountM * WarpTileM == CtaM, "Warp tile M must divide CTA tile M");
static_assert(WarpCountN * WarpTileN == CtaN, "Warp tile N must divide CTA tile N");
static_assert(WarpTilesM * WMMA_M == WarpTileM, "WMMA tile M must divide warp tile M");
static_assert(WarpTilesN * WMMA_N == WarpTileN, "WMMA tile N must divide warp tile N");
static_assert(ThreadsX == 32 && ThreadsY == 8, "code3.cu expects blockDim to be (32, 8)");

namespace {

__global__ void matmul_kernel(const double* A, const double* B, double* C, int m, int n, int k)
{
    __shared__ __align__(32) double As[CtaM][CtaK];
    __shared__ __align__(32) double Bs[CtaK][CtaN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int tile_m = blockIdx.y * CtaM;
    const int tile_n = blockIdx.x * CtaN;
    const int warp_m = ty / WarpCountN;
    const int warp_n = ty % WarpCountN;
    const int warp_row = warp_m * WarpTileM;
    const int warp_col = warp_n * WarpTileN;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag[WarpTilesM];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag[WarpTilesN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> c_frag[WarpTilesM][WarpTilesN];

    #pragma unroll
    for (int wm = 0; wm < WarpTilesM; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WarpTilesN; ++wn) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0);
        }
    }

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
            #pragma unroll
            for (int wm = 0; wm < WarpTilesM; ++wm) {
                wmma::load_matrix_sync(a_frag[wm], &As[warp_row + wm * WMMA_M][kk], CtaK);
            }

            #pragma unroll
            for (int wn = 0; wn < WarpTilesN; ++wn) {
                wmma::load_matrix_sync(b_frag[wn], &Bs[kk][warp_col + wn * WMMA_N], CtaN);
            }

            #pragma unroll
            for (int wm = 0; wm < WarpTilesM; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WarpTilesN; ++wn) {
                    wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int wm = 0; wm < WarpTilesM; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WarpTilesN; ++wn) {
            wmma::store_matrix_sync(C + static_cast<size_t>(tile_m + warp_row + wm * WMMA_M) * n +
                                        tile_n + warp_col + wn * WMMA_N,
                                    c_frag[wm][wn],
                                    n,
                                    wmma::mem_row_major);
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

    dim3 block(ThreadsX, ThreadsY);
    dim3 grid(n / CtaN, m / CtaM);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
