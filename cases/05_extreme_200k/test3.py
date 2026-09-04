#!/usr/bin/env python3
import torch


M = 32768
N = 32768
K = 32768
REPEATS = 3
DTYPE = torch.float64


def tflops(seconds):
    return 2.0 * M * N * K / seconds / 1e12


def bench(device_index):
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    print(f"GPU{device_index}: {torch.cuda.get_device_name(device_index)}")
    print(f"M={M} N={N} K={K} dtype={DTYPE}")

    A = torch.ones((M, K), device=device, dtype=DTYPE)
    B = torch.ones((K, N), device=device, dtype=DTYPE)
    C = torch.empty((M, N), device=device, dtype=DTYPE)

    torch.matmul(A, B, out=C)
    torch.cuda.synchronize()

    times = []
    for i in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.matmul(A, B, out=C)
        stop.record()
        torch.cuda.synchronize()

        seconds = start.elapsed_time(stop) / 1000.0
        times.append(seconds)
        print(f"iter {i + 1}: {seconds:.3f}s, {tflops(seconds):.3f} TFLOPS")

    print(f"best: {min(times):.3f}s, avg: {sum(times) / len(times):.3f}s, C[0,0]={float(C[0, 0])}")
    print()


def main():
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")

    for device_index in range(torch.cuda.device_count()):
        bench(device_index)


if __name__ == "__main__":
    main()
