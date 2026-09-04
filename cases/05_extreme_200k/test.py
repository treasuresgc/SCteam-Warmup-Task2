#!/usr/bin/env python3
import torch


BYTES = 512 * 1024 * 1024
REPEATS = 10
DTYPE = torch.uint8


def gbps(num_bytes, ms):
    return num_bytes / (ms / 1000.0) / (1024 * 1024 * 1024)


def measure_copy(src, dst):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    dst.copy_(src, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def bench_h2d(device_index):
    torch.cuda.set_device(device_index)
    src = torch.empty(BYTES, dtype=DTYPE, pin_memory=True)
    dst = torch.empty(BYTES, dtype=DTYPE, device=f"cuda:{device_index}")
    src.fill_(1)
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()

    total_ms = 0.0
    for _ in range(REPEATS):
        total_ms += measure_copy(src, dst)

    print(f"GPU{device_index} H2D: {gbps(BYTES, total_ms / REPEATS):.2f} GB/s")


def bench_d2h(device_index):
    torch.cuda.set_device(device_index)
    src = torch.empty(BYTES, dtype=DTYPE, device=f"cuda:{device_index}")
    dst = torch.empty(BYTES, dtype=DTYPE, pin_memory=True)
    src.fill_(1)
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()

    total_ms = 0.0
    for _ in range(REPEATS):
        total_ms += measure_copy(src, dst)

    print(f"GPU{device_index} D2H: {gbps(BYTES, total_ms / REPEATS):.2f} GB/s")


def main():
    bench_h2d(0)
    bench_d2h(0)
    bench_h2d(1)
    bench_d2h(1)


if __name__ == "__main__":
    main()
