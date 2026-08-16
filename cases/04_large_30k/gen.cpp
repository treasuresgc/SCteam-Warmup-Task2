#include <random>
#include <vector>

constexpr int M = 32768;
constexpr int N = 32768;
constexpr int K = 32768;

void run(double* A, double* B, double* C, int m, int n, int k);

int main() {
    std::mt19937 rng(20260606);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> A(static_cast<size_t>(M) * K);
    std::vector<double> B(static_cast<size_t>(K) * N);
    std::vector<double> C(static_cast<size_t>(M) * N, 0.0);

    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    run(A.data(), B.data(), C.data(), M, N, K);
    return 0;
}
