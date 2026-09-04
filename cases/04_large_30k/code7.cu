#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include <cstddef>

#include <mma.h>
using namespace nvcuda;

constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 4;

constexpr int CtaM = 128;
constexpr int CtaN = 128;
constexpr int CtaK = 16;

constexpr int WarpTileM = 16;
constexpr int WarpTileN = 32;
constexpr int WarpCountM = CtaM / WarpTileM;
constexpr int WarpCountN = CtaN / WarpTileN;
constexpr int WarpTilesM = WarpTileM / WMMA_M;
constexpr int WarpTilesN = WarpTileN / WMMA_N;

constexpr int ThreadsX = 32;
constexpr int ThreadsY = WarpCountM * WarpCountN;
constexpr int ThreadsPerBlock = ThreadsX * ThreadsY;

constexpr int VecWidth = 2;

// Pack A as 8x4 blocks and B as 4x8 blocks so every WMMA fragment starts
// at a contiguous block boundary in shared memory.
constexpr int ABlockM = WMMA_M;
constexpr int ABlockK = WMMA_K;
constexpr int BBlockK = WMMA_K;
constexpr int BBlockN = WMMA_N;

constexpr int ABlockCountM = CtaM / ABlockM;
constexpr int ABlockCountK = CtaK / ABlockK;
constexpr int BBlockCountK = CtaK / BBlockK;
constexpr int BBlockCountN = CtaN / BBlockN;

constexpr int ABlockVecPerRow = ABlockK / VecWidth;
constexpr int BBlockVecPerRow = BBlockN / VecWidth;
constexpr int ABlockVec = ABlockM * ABlockVecPerRow;
constexpr int BBlockVec = BBlockK * BBlockVecPerRow;
constexpr int ABlockDoubles = ABlockM * ABlockK;
constexpr int BBlockDoubles = BBlockK * BBlockN;
constexpr int AStageVec = ABlockCountM * ABlockCountK * ABlockVec;
constexpr int BStageVec = BBlockCountK * BBlockCountN * BBlockVec;
constexpr size_t SharedBytes = static_cast<size_t>(2) * (AStageVec + BStageVec) * sizeof(double2);

static_assert(WarpCountM * WarpTileM == CtaM, "Warp tile M must divide CTA tile M");
static_assert(WarpCountN * WarpTileN == CtaN, "Warp tile N must divide CTA tile N");
static_assert(WarpTilesM * WMMA_M == WarpTileM, "WMMA tile M must divide warp tile M");
static_assert(WarpTilesN * WMMA_N == WarpTileN, "WMMA tile N must divide warp tile N");
static_assert(ThreadsX == 32 && ThreadsY == 32, "code7.cu expects blockDim to be (32, 32)");
static_assert(CtaK % VecWidth == 0 && CtaN % VecWidth == 0, "double2 vectorization requires even tile widths");
static_assert(AStageVec == ThreadsPerBlock, "A packed stage expects one double2 copy per thread");
static_assert(BStageVec == ThreadsPerBlock, "B packed stage expects one double2 copy per thread");
static_assert(ABlockM == WMMA_M && ABlockK == WMMA_K, "A block must match the WMMA A fragment");
static_assert(BBlockK == WMMA_K && BBlockN == WMMA_N, "B block must match the WMMA B fragment");
static_assert(SharedBytes == 65536, "Packed double-buffered shared staging should use 64 KiB");

namespace {

__device__ __forceinline__ void prefetch_tile(double2* As_stage,
                                              double2* Bs_stage,
                                              const double2* A_vec,
                                              const double2* B_vec,
                                              int tile_m,
                                              int tile_n,
                                              int tile_k,
                                              int a_src_vec_ldm,
                                              int b_src_vec_ldm,
                                              int tid)
{
    const int a_tile_vec_col = tile_k >> 1;
    const int b_tile_vec_col = tile_n >> 1;

    const int a_block = tid >> 4;
    const int a_lane = tid & 15;
    const int a_block_row = a_block >> 2;
    const int a_block_col = a_block & 3;
    const int a_row = (a_block_row << 3) + (a_lane >> 1);
    const int a_vec_col = (a_block_col << 1) + (a_lane & 1);

    __pipeline_memcpy_async(&As_stage[static_cast<size_t>(a_block) * ABlockVec + a_lane],
                            &A_vec[static_cast<size_t>(tile_m + a_row) * a_src_vec_ldm + a_tile_vec_col + a_vec_col],
                            sizeof(double2));

    const int b_block = tid >> 4;
    const int b_lane = tid & 15;
    const int b_block_row = b_block >> 4;
    const int b_block_col = b_block & 15;
    const int b_row = (b_block_row << 2) + (b_lane >> 2);
    const int b_vec_col = (b_block_col << 2) + (b_lane & 3);

    __pipeline_memcpy_async(&Bs_stage[static_cast<size_t>(b_block) * BBlockVec + b_lane],
                            &B_vec[static_cast<size_t>(tile_k + b_row) * b_src_vec_ldm + b_tile_vec_col + b_vec_col],
                            sizeof(double2));
}

__global__ __launch_bounds__(ThreadsPerBlock, 1) void matmul_kernel(const double* __restrict__ A,
                                                                    const double* __restrict__ B,
                                                                    double* __restrict__ C,
                                                                    int m,
                                                                    int n,
                                                                    int k)
{
    extern __shared__ __align__(32) double2 smem[];
    const auto* A_vec = reinterpret_cast<const double2*>(A);
    const auto* B_vec = reinterpret_cast<const double2*>(B);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int tile_m = blockIdx.y << 7;
    const int tile_n = blockIdx.x << 7;
    const int a_src_vec_ldm = k >> 1;
    const int b_src_vec_ldm = n >> 1;
    const int num_tiles = k >> 4;
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

    if (num_tiles == 0) {
        return;
    }

    prefetch_tile(smem, smem + 2 * AStageVec, A_vec, B_vec, tile_m, tile_n, 0, a_src_vec_ldm, b_src_vec_ldm, tid);
    __pipeline_commit();
    if (num_tiles > 1) {
        prefetch_tile(smem + AStageVec, smem + 2 * AStageVec + BStageVec, A_vec, B_vec, tile_m, tile_n, CtaK, a_src_vec_ldm, b_src_vec_ldm, tid);
        __pipeline_commit();
    }

    __pipeline_wait_prior(num_tiles > 1 ? 1 : 0);
    __syncthreads();

    #pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int stage = tile & 1;
        const int tile_k = tile << 4;
        double2* As_stage = smem + static_cast<size_t>(stage) * AStageVec;
        double2* Bs_stage = smem + 2 * AStageVec + static_cast<size_t>(stage) * BStageVec;
        const double* As_tile = reinterpret_cast<const double*>(As_stage);
        const double* Bs_tile = reinterpret_cast<const double*>(Bs_stage);

        #pragma unroll
        for (int kk = 0; kk < CtaK; kk += WMMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WarpTilesM; ++wm) {
                const int a_row = warp_row + wm * WMMA_M;
                const int a_block = (a_row >> 3) * ABlockCountK + (kk >> 2);
                wmma::load_matrix_sync(a_frag[wm], As_tile + static_cast<size_t>(a_block) * ABlockDoubles, ABlockK);
            }

            #pragma unroll
            for (int wn = 0; wn < WarpTilesN; ++wn) {
                const int b_col = warp_col + wn * WMMA_N;
                const int b_block = (kk >> 2) * BBlockCountN + (b_col >> 3);
                wmma::load_matrix_sync(b_frag[wn], Bs_tile + static_cast<size_t>(b_block) * BBlockDoubles, BBlockN);
            }

            #pragma unroll
            for (int wm = 0; wm < WarpTilesM; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WarpTilesN; ++wn) {
                    wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
                }
            }
        }

        if (tile + 2 < num_tiles) {
            __syncthreads();
            prefetch_tile(As_stage,
                          Bs_stage,
                          A_vec,
                          B_vec,
                          tile_m,
                          tile_n,
                          tile_k + 2 * CtaK,
                          a_src_vec_ldm,
                          b_src_vec_ldm,
                          tid);
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
        } else if (tile + 1 < num_tiles) {
            __syncthreads();
            __pipeline_wait_prior(0);
            __syncthreads();
        }
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
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SharedBytes));
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    matmul_kernel<<<grid, block, SharedBytes>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
