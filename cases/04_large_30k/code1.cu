#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdio>

#include <mma.h>
using namespace nvcuda;

const int ThreadsX = 32;
const int ThreadsY = 16;
const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 4;

namespace {

__global__ void matmul_kernel(const double* A, const double* B, double* C, int m, int n, int k)
{
    int warp_id = threadIdx.y;
    // int lane_id = threadIdx.x;

    int row = (blockIdx.y) * WMMA_M;
    int col = (blockIdx.x * ThreadsY + warp_id) * WMMA_N;

    wmma::fragment<
        wmma::matrix_a,
        8,8,4,double,wmma::row_major
        > a_frag;
    wmma::fragment<
        wmma::matrix_b,
        8,8,4,double,wmma::row_major
        > b_frag;
    wmma::fragment<
        wmma::accumulator,
        8,8,4,double
        > c_frag;

    wmma::fill_fragment(c_frag, 0.0);

    for (int i = 0; i < k; i += WMMA_K) {
        const double* Aptr = A + row*k + i;
        const double* Bptr = B + i*n + col;
        wmma::load_matrix_sync(a_frag, Aptr, k);
        wmma::load_matrix_sync(b_frag, Bptr, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    double* Cptr = C + row*n + col;
    wmma::store_matrix_sync(Cptr, c_frag, n, wmma::mem_row_major);

}  // namespace

}

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
    dim3 grid(n / ThreadsY / WMMA_N, m / WMMA_M);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
