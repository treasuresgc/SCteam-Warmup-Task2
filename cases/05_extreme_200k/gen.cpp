#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr std::uint64_t kM = 200000;
constexpr std::uint64_t kN = 200000;
constexpr std::uint64_t kK = 200000;
constexpr std::size_t kBlockRows = 64;
constexpr std::uint64_t kSeedA = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t kSeedB = 0x13198a2e03707344ULL;
constexpr double kInv2Pow53 = 1.0 / 9007199254740992.0;

static_assert(sizeof(double) == 8, "FP64 is required here");

std::uint64_t splitmix64(std::uint64_t& state)
{
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double to_unit_double(std::uint64_t bits)
{
    return static_cast<double>(bits >> 11) * kInv2Pow53;
}

struct Progress {
    std::uint64_t total_bytes = 0;
    std::uint64_t done_bytes = 0;
    std::uint64_t next_mark = 5;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    void advance(std::uint64_t bytes, const char* label, std::uint64_t rows_done, std::uint64_t rows_total)
    {
        done_bytes += bytes;
        while (next_mark <= 100) {
            const std::uint64_t threshold = total_bytes * next_mark / 100;
            if (done_bytes < threshold) {
                break;
            }
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            std::printf("[%3llu%%] %s %llu/%llu rows elapsed=%llds\n",
                        static_cast<unsigned long long>(next_mark),
                        label,
                        static_cast<unsigned long long>(rows_done),
                        static_cast<unsigned long long>(rows_total),
                        static_cast<long long>(elapsed));
            std::fflush(stdout);
            next_mark += 5;
        }
    }
};

void fill_block(double* dst, std::uint64_t row_begin, std::uint64_t rows, std::uint64_t cols, std::uint64_t seed)
{
    for (std::uint64_t r = 0; r < rows; ++r) {
        std::uint64_t state = seed + (row_begin + r + 1) * 0x9e3779b97f4a7c15ULL;
        double* row = dst + static_cast<std::size_t>(r * cols);
        for (std::uint64_t c = 0; c < cols; ++c) {
            row[c] = to_unit_double(splitmix64(state)) - 0.5;
        }
    }
}

bool write_matrix(const std::filesystem::path& path,
                  const char* label,
                  std::uint64_t rows,
                  std::uint64_t cols,
                  std::uint64_t seed,
                  Progress& progress)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::printf("open failed: %s\n", path.string().c_str());
        return false;
    }

    const std::size_t block_elems = static_cast<std::size_t>(kBlockRows) * static_cast<std::size_t>(cols);
    std::vector<double> block(block_elems);

    for (std::uint64_t row0 = 0; row0 < rows; row0 += kBlockRows) {
        const std::uint64_t cur_rows = std::min<std::uint64_t>(kBlockRows, rows - row0);
        fill_block(block.data(), row0, cur_rows, cols, seed);

        const std::uint64_t bytes = cur_rows * cols * sizeof(double);
        out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(bytes));
        if (!out) {
            std::printf("write failed: %s\n", path.string().c_str());
            return false;
        }

        progress.advance(bytes, label, row0 + cur_rows, rows);
    }

    out.flush();
    if (!out) {
        std::printf("flush failed: %s\n", path.string().c_str());
        return false;
    }

    return true;
}

bool write_meta(const std::filesystem::path& path,
                const std::filesystem::path& a_path,
                const std::filesystem::path& b_path)
{
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        std::printf("open failed: %s\n", path.string().c_str());
        return false;
    }

    out << "M=" << kM << '\n';
    out << "N=" << kN << '\n';
    out << "K=" << kK << '\n';
    out << "dtype=float64\n";
    out << "layout=row-major\n";
    out << "A=" << a_path.filename().string() << '\n';
    out << "B=" << b_path.filename().string() << '\n';
    out << "seed_a=" << kSeedA << '\n';
    out << "seed_b=" << kSeedB << '\n';
    out << "block_rows=" << kBlockRows << '\n';
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv)
{
    std::error_code ec;
    std::filesystem::path exe_path = std::filesystem::absolute(argv[0], ec);
    if (ec) {
        exe_path = std::filesystem::path(argv[0]);
    }

    std::filesystem::path base_dir = exe_path.parent_path();
    if (base_dir.empty()) {
        base_dir = ".";
    }

    const std::filesystem::path data_dir = base_dir / "data";
    std::filesystem::create_directories(data_dir, ec);
    if (ec) {
        std::printf("mkdir failed: %s\n", data_dir.string().c_str());
        return 1;
    }

    const std::filesystem::path a_path = data_dir / "A.bin";
    const std::filesystem::path b_path = data_dir / "B.bin";
    const std::filesystem::path meta_path = data_dir / "meta.txt";

    const std::uint64_t matrix_bytes = kM * kK * sizeof(double);
    const std::uint64_t total_bytes = matrix_bytes * 2;
    const double gib = 1024.0 * 1024.0 * 1024.0;

    std::printf("FP64 generator start\n");
    std::printf("output: %s\n", data_dir.string().c_str());
    std::printf("A/B each: %.2f GiB, total: %.2f GiB\n",
                static_cast<double>(matrix_bytes) / gib,
                static_cast<double>(total_bytes) / gib);
    std::fflush(stdout);

    Progress progress;
    progress.total_bytes = total_bytes;

    std::printf("writing %s\n", a_path.filename().string().c_str());
    std::fflush(stdout);
    if (!write_matrix(a_path, "A.bin", kM, kK, kSeedA, progress)) {
        return 1;
    }

    std::printf("writing %s\n", b_path.filename().string().c_str());
    std::fflush(stdout);
    if (!write_matrix(b_path, "B.bin", kK, kN, kSeedB, progress)) {
        return 1;
    }

    if (!write_meta(meta_path, a_path, b_path)) {
        return 1;
    }

    std::printf("done\n");
    return 0;
}
