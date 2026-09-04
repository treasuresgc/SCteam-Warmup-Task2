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

constexpr int WarpTileM = 32;
constexpr int WarpTileN = 32;
constexpr int WarpCountM = CtaM / WarpTileM;
constexpr int WarpCountN = CtaN / WarpTileN;
constexpr int WarpTilesM = WarpTileM / WMMA_M;
constexpr int WarpTilesN = WarpTileN / WMMA_N;

constexpr int ThreadsX = 32;
constexpr int ThreadsY = WarpCountM * WarpCountN;
constexpr int ThreadsPerBlock = ThreadsX * ThreadsY;
constexpr int VecWidth = 2;
constexpr int CtaKVec = CtaK / VecWidth;
constexpr int CtaNVec = CtaN / VecWidth;
constexpr int AsLdm = CtaK + 2;
constexpr int BsLdm = CtaN + 2;
constexpr int AsVecLdm = AsLdm / VecWidth;
constexpr int BsVecLdm = BsLdm / VecWidth;
constexpr int AsStageVec = CtaM * AsVecLdm;
constexpr int BsStageVec = CtaK * BsVecLdm;
constexpr size_t SharedBytes = static_cast<size_t>(2) * (AsStageVec + BsStageVec) * sizeof(double2);

static_assert(WarpCountM * WarpTileM == CtaM, "Warp tile M must divide CTA tile M");
static_assert(WarpCountN * WarpTileN == CtaN, "Warp tile N must divide CTA tile N");
static_assert(WarpTilesM * WMMA_M == WarpTileM, "WMMA tile M must divide warp tile M");
static_assert(WarpTilesN * WMMA_N == WarpTileN, "WMMA tile N must divide warp tile N");
static_assert(ThreadsX == 32 && ThreadsY == 16, "code5.cu expects blockDim to be (32, 16)");
static_assert(CtaK % VecWidth == 0 && CtaN % VecWidth == 0, "double2 vectorization requires even tile widths");
static_assert(AsLdm % VecWidth == 0 && BsLdm % VecWidth == 0, "double2 vectorization requires even shared strides");
static_assert(CtaM * CtaKVec == ThreadsPerBlock * 2, "A tile staging expects two double2 copies per thread");
static_assert(CtaK * CtaNVec == ThreadsPerBlock * 2, "B tile staging expects two double2 copies per thread");

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
    const int a_tile_vec_col = tile_k / VecWidth;
    const int b_tile_vec_col = tile_n / VecWidth;

    const int a_idx0 = tid;
    const int a_idx1 = tid + ThreadsPerBlock;
    const int a_row0 = a_idx0 / CtaKVec;
    const int a_col0 = a_idx0 % CtaKVec;
    const int a_row1 = a_idx1 / CtaKVec;
    const int a_col1 = a_idx1 % CtaKVec;

    __pipeline_memcpy_async(&As_stage[static_cast<size_t>(a_row0) * AsVecLdm + a_col0],
                            &A_vec[static_cast<size_t>(tile_m + a_row0) * a_src_vec_ldm + a_tile_vec_col + a_col0],
                            sizeof(double2));
    __pipeline_memcpy_async(&As_stage[static_cast<size_t>(a_row1) * AsVecLdm + a_col1],
                            &A_vec[static_cast<size_t>(tile_m + a_row1) * a_src_vec_ldm + a_tile_vec_col + a_col1],
                            sizeof(double2));

    const int b_idx0 = tid;
    const int b_idx1 = tid + ThreadsPerBlock;
    const int b_row0 = b_idx0 / CtaNVec;
    const int b_col0 = b_idx0 % CtaNVec;
    const int b_row1 = b_idx1 / CtaNVec;
    const int b_col1 = b_idx1 % CtaNVec;

    __pipeline_memcpy_async(&Bs_stage[static_cast<size_t>(b_row0) * BsVecLdm + b_col0],
                            &B_vec[static_cast<size_t>(tile_k + b_row0) * b_src_vec_ldm + b_tile_vec_col + b_col0],
                            sizeof(double2));
    __pipeline_memcpy_async(&Bs_stage[static_cast<size_t>(b_row1) * BsVecLdm + b_col1],
                            &B_vec[static_cast<size_t>(tile_k + b_row1) * b_src_vec_ldm + b_tile_vec_col + b_col1],
                            sizeof(double2));
}

__global__ void matmul_kernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C,
                              int m,
                              int n,
                              int k)
{
    extern __shared__ __align__(32) double2 smem[];
    double2* As_stage[2] = {
        smem,
        smem + AsStageVec,
    };
    double2* Bs_stage[2] = {
        smem + 2 * AsStageVec,
        smem + 2 * AsStageVec + BsStageVec,
    };
    double* As_tile[2] = {
        reinterpret_cast<double*>(As_stage[0]),
        reinterpret_cast<double*>(As_stage[1]),
    };
    double* Bs_tile[2] = {
        reinterpret_cast<double*>(Bs_stage[0]),
        reinterpret_cast<double*>(Bs_stage[1]),
    };
    const auto* A_vec = reinterpret_cast<const double2*>(A);
    const auto* B_vec = reinterpret_cast<const double2*>(B);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int tile_m = blockIdx.y * CtaM;
    const int tile_n = blockIdx.x * CtaN;
    const int a_src_vec_ldm = k / VecWidth;
    const int b_src_vec_ldm = n / VecWidth;
    const int num_tiles = k / CtaK;
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

    prefetch_tile(As_stage[0], Bs_stage[0], A_vec, B_vec, tile_m, tile_n, 0, a_src_vec_ldm, b_src_vec_ldm, tid);
    __pipeline_commit();
    if (num_tiles > 1) {
        prefetch_tile(As_stage[1], Bs_stage[1], A_vec, B_vec, tile_m, tile_n, CtaK, a_src_vec_ldm, b_src_vec_ldm, tid);
        __pipeline_commit();
    }

    __pipeline_wait_prior(num_tiles > 1 ? 1 : 0);
    __syncthreads();

    #pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int stage = tile & 1;
        const int tile_k = tile * CtaK;

        #pragma unroll
        for (int kk = 0; kk < CtaK; kk += WMMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WarpTilesM; ++wm) {
                wmma::load_matrix_sync(a_frag[wm],
                                       &As_tile[stage][static_cast<size_t>(warp_row + wm * WMMA_M) * AsLdm + kk],
                                       AsLdm);
            }

            #pragma unroll
            for (int wn = 0; wn < WarpTilesN; ++wn) {
                wmma::load_matrix_sync(b_frag[wn],
                                       &Bs_tile[stage][static_cast<size_t>(kk) * BsLdm + warp_col + wn * WMMA_N],
                                       BsLdm);
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
            prefetch_tile(As_stage[stage], Bs_stage[stage], A_vec, B_vec, tile_m, tile_n, tile_k + 2 * CtaK, a_src_vec_ldm, b_src_vec_ldm, tid);
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
