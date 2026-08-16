#include <random>
#include <cstddef>
#include <vector>

constexpr int M = 2048000;
constexpr int N = 2048000;
constexpr int K = 2048000;

void run(float* A, float* B, float* C, int m, int n, int k);

int main() {
    std::mt19937 rng(20260606);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(K) * N);
    std::vector<float> C(static_cast<std::size_t>(M) * N, 0.0f);

    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    run(A.data(), B.data(), C.data(), M, N, K);
    return 0;
}
