#include <random>
#include <vector>

constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 32;

void run(double* A, double* B, double* C, int m, int n, int k);

int main() {
    std::mt19937 rng(20260606);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> A(M * K);
    std::vector<double> B(K * N);
    std::vector<double> C(M * N, 0.0);

    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    run(A.data(), B.data(), C.data(), M, N, K);
    return 0;
}
