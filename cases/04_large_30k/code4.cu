#include <cuda_runtime.h>
#include <mma.h>

#include <chrono>
#include <cstdio>
#include <cstddef>

namespace {

using namespace nvcuda;

constexpr int WMMA_M = 8;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 4;

#ifndef WARPS_M
#define WARPS_M 2
#endif

#ifndef WARPS_N
#define WARPS_N 4
#endif

constexpr int kWarpsM = WARPS_M;
constexpr int kWarpsN = WARPS_N;
constexpr int kWarpsPerBlock = kWarpsM * kWarpsN;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kBlockM = kWarpsM * WMMA_M;
constexpr int kBlockN = kWarpsN * WMMA_N;

static_assert(kWarpsPerBlock > 0, "WARPS_M * WARPS_N must be positive");
static_assert(kThreadsPerBlock <= 1024, "Tensor Core block has too many threads");

__global__ void copy_a_padded(const double* src, double* dst, int m, int k, int kpad)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(m) * k;
    if (idx < total) {
        int row = static_cast<int>(idx / k);
        int col = static_cast<int>(idx - static_cast<size_t>(row) * k);
        dst[static_cast<size_t>(row) * kpad + col] = src[idx];
    }
}

__global__ void copy_b_padded(const double* src, double* dst, int k, int n, int npad)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(k) * n;
    if (idx < total) {
        int row = static_cast<int>(idx / n);
        int col = static_cast<int>(idx - static_cast<size_t>(row) * n);
        dst[static_cast<size_t>(row) * npad + col] = src[idx];
    }
}

__global__ void copy_c_unpadded(const double* src, double* dst, int m, int n, int npad)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(m) * n;
    if (idx < total) {
        int row = static_cast<int>(idx / n);
        int col = static_cast<int>(idx - static_cast<size_t>(row) * n);
        dst[idx] = src[static_cast<size_t>(row) * npad + col];
    }
}

__global__ void wmma_fp64_core(const double* A, const double* B, double* C, int m, int n, int k, int* mode)
{
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        *mode = 1;
    }

    int warp_id = threadIdx.x >> 5;
    int warp_m = warp_id / kWarpsN;
    int warp_n = warp_id - warp_m * kWarpsN;

    int tile_m = (blockIdx.y * kWarpsM + warp_m) * WMMA_M;
    int tile_n = (blockIdx.x * kWarpsN + warp_n) * WMMA_N;

    if (tile_m >= m || tile_n >= n) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0);

    for (int kk = 0; kk < k; kk += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(tile_m) * k + kk, k);
        wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(kk) * n + tile_n, n);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + static_cast<size_t>(tile_m) * n + tile_n, acc_frag, n, wmma::mem_row_major);
}

bool check_cuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess) {
        printf("%s: %s\n", message, cudaGetErrorString(status));
        return false;
    }
    return true;
}

int round_up(int x, int step)
{
    return (x + step - 1) / step * step;
}

const char* fp64_kernel_label(int mode)
{
    if (mode != 0) {
        return "Tensor Core FP64";
    }
    return "FP64 fallback";
}

bool run_tensor_core_aligned(const double* A, const double* B, double* C, int m, int n, int k)
{
    size_t dbl = sizeof(double);
    size_t siza = static_cast<size_t>(m) * k * dbl;
    size_t sizb = static_cast<size_t>(k) * n * dbl;
    size_t sizc = static_cast<size_t>(m) * n * dbl;
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    int* d_mode = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ok = true;

    ok = ok && check_cuda(cudaMalloc(&d_A, siza), "Failed to allocate device memory for A");
    ok = ok && check_cuda(cudaMalloc(&d_B, sizb), "Failed to allocate device memory for B");
    ok = ok && check_cuda(cudaMalloc(&d_C, sizc), "Failed to allocate device memory for C");
    ok = ok && check_cuda(cudaMalloc(&d_mode, sizeof(int)), "Failed to allocate device memory for kernel mode");
    ok = ok && check_cuda(cudaMemcpy(d_A, A, siza, cudaMemcpyHostToDevice), "Failed to copy A to device");
    ok = ok && check_cuda(cudaMemcpy(d_B, B, sizb, cudaMemcpyHostToDevice), "Failed to copy B to device");
    ok = ok && check_cuda(cudaMemset(d_mode, 0, sizeof(int)), "Failed to clear kernel mode");

    if (ok) {
        dim3 block(kThreadsPerBlock);
        dim3 grid((n + kBlockN - 1) / kBlockN, (m + kBlockM - 1) / kBlockM);

        ok = ok && check_cuda(cudaEventCreate(&start), "Failed to create CUDA start event");
        ok = ok && check_cuda(cudaEventCreate(&stop), "Failed to create CUDA stop event");
        ok = ok && check_cuda(cudaEventRecord(start), "Failed to start CUDA timing");
        wmma_fp64_core<<<grid, block>>>(d_A, d_B, d_C, m, n, k, d_mode);
        ok = ok && check_cuda(cudaGetLastError(), "Failed to launch FP64 Tensor Core kernel");
        ok = ok && check_cuda(cudaEventRecord(stop), "Failed to stop CUDA timing");
        ok = ok && check_cuda(cudaEventSynchronize(stop), "Failed to synchronize CUDA timing");

        if (ok) {
            float elapsed_ms = 0.0f;
            int kernel_mode = 0;
            ok = ok && check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "Failed to calculate CUDA elapsed time");
            ok = ok && check_cuda(cudaMemcpy(&kernel_mode, d_mode, sizeof(kernel_mode), cudaMemcpyDeviceToHost),
                                  "Failed to copy kernel mode to host");
            if (ok) {
                printf("%s kernel time: %.3f ms\n", fp64_kernel_label(kernel_mode), elapsed_ms);
            }
        }
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
    cudaFree(d_mode);
    return ok;
}

bool run_tensor_core_padded(const double* A, const double* B, double* C, int m, int n, int k)
{
    int mpad = round_up(m, WMMA_M);
    int npad = round_up(n, WMMA_N);
    int kpad = round_up(k, WMMA_K);

    size_t dbl = sizeof(double);
    size_t siza = static_cast<size_t>(m) * k * dbl;
    size_t sizb = static_cast<size_t>(k) * n * dbl;
    size_t sizc = static_cast<size_t>(m) * n * dbl;
    size_t siza_pad = static_cast<size_t>(mpad) * kpad * dbl;
    size_t sizb_pad = static_cast<size_t>(kpad) * npad * dbl;
    size_t sizc_pad = static_cast<size_t>(mpad) * npad * dbl;

    double* d_A_src = nullptr;
    double* d_B_src = nullptr;
    double* d_C_dst = nullptr;
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    int* d_mode = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ok = true;

    ok = ok && check_cuda(cudaMalloc(&d_A_src, siza), "Failed to allocate source device memory for A");
    ok = ok && check_cuda(cudaMalloc(&d_B_src, sizb), "Failed to allocate source device memory for B");
    ok = ok && check_cuda(cudaMalloc(&d_C_dst, sizc), "Failed to allocate output device memory for C");
    ok = ok && check_cuda(cudaMalloc(&d_A, siza_pad), "Failed to allocate padded device memory for A");
    ok = ok && check_cuda(cudaMalloc(&d_B, sizb_pad), "Failed to allocate padded device memory for B");
    ok = ok && check_cuda(cudaMalloc(&d_C, sizc_pad), "Failed to allocate padded device memory for C");
    ok = ok && check_cuda(cudaMalloc(&d_mode, sizeof(int)), "Failed to allocate device memory for kernel mode");
    ok = ok && check_cuda(cudaMemcpy(d_A_src, A, siza, cudaMemcpyHostToDevice), "Failed to copy A to device");
    ok = ok && check_cuda(cudaMemcpy(d_B_src, B, sizb, cudaMemcpyHostToDevice), "Failed to copy B to device");
    ok = ok && check_cuda(cudaMemset(d_A, 0, siza_pad), "Failed to clear padded A");
    ok = ok && check_cuda(cudaMemset(d_B, 0, sizb_pad), "Failed to clear padded B");
    ok = ok && check_cuda(cudaMemset(d_mode, 0, sizeof(int)), "Failed to clear kernel mode");

    if (ok) {
        constexpr int copy_threads = 256;
        dim3 copy_a_grid((static_cast<size_t>(m) * k + copy_threads - 1) / copy_threads);
        dim3 copy_b_grid((static_cast<size_t>(k) * n + copy_threads - 1) / copy_threads);
        dim3 copy_c_grid((static_cast<size_t>(m) * n + copy_threads - 1) / copy_threads);
        dim3 wmma_block(kThreadsPerBlock);
        dim3 wmma_grid((npad + kBlockN - 1) / kBlockN, (mpad + kBlockM - 1) / kBlockM);

        ok = ok && check_cuda(cudaEventCreate(&start), "Failed to create CUDA start event");
        ok = ok && check_cuda(cudaEventCreate(&stop), "Failed to create CUDA stop event");
        ok = ok && check_cuda(cudaEventRecord(start), "Failed to start CUDA timing");
        copy_a_padded<<<copy_a_grid, copy_threads>>>(d_A_src, d_A, m, k, kpad);
        copy_b_padded<<<copy_b_grid, copy_threads>>>(d_B_src, d_B, k, n, npad);
        wmma_fp64_core<<<wmma_grid, wmma_block>>>(d_A, d_B, d_C, mpad, npad, kpad, d_mode);
        copy_c_unpadded<<<copy_c_grid, copy_threads>>>(d_C, d_C_dst, m, n, npad);
        ok = ok && check_cuda(cudaGetLastError(), "Failed to launch padded FP64 Tensor Core pipeline");
        ok = ok && check_cuda(cudaEventRecord(stop), "Failed to stop CUDA timing");
        ok = ok && check_cuda(cudaEventSynchronize(stop), "Failed to synchronize CUDA timing");

        if (ok) {
            float elapsed_ms = 0.0f;
            int kernel_mode = 0;
            ok = ok && check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "Failed to calculate CUDA elapsed time");
            ok = ok && check_cuda(cudaMemcpy(&kernel_mode, d_mode, sizeof(kernel_mode), cudaMemcpyDeviceToHost),
                                  "Failed to copy kernel mode to host");
            if (ok) {
                printf("%s padded pipeline time: %.3f ms\n", fp64_kernel_label(kernel_mode), elapsed_ms);
            }
        }
    }

    ok = ok && check_cuda(cudaMemcpy(C, d_C_dst, sizc, cudaMemcpyDeviceToHost), "Failed to copy result to host");

    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    cudaFree(d_A_src);
    cudaFree(d_B_src);
    cudaFree(d_C_dst);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_mode);
    return ok;
}

}  // namespace

void run(double* A, double* B, double* C, int m, int n, int k)
{
    bool aligned = (m % WMMA_M == 0) && (n % WMMA_N == 0) && (k % WMMA_K == 0);
    if (aligned) {
        run_tensor_core_aligned(A, B, C, m, n, k);
    } else {
        run_tensor_core_padded(A, B, C, m, n, k);
    }
}
