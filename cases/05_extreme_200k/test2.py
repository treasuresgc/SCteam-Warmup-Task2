#!/usr/bin/env python3
import argparse
import mmap
import os
import threading
import time
from pathlib import Path


PAGE_SIZE = 4096
DEFAULT_SIZE_MB = 1024
DEFAULT_BLOCK_MB = 16
DEFAULT_JOBS = [1, 2, 4, 8]


def gbps(num_bytes, seconds):
    return num_bytes / seconds / (1024 * 1024 * 1024)


def align_down(value, step):
    return value - (value % step)


def make_buffer(size_bytes):
    buf = mmap.mmap(-1, size_bytes, access=mmap.ACCESS_WRITE)
    view = memoryview(buf)
    for offset in range(0, size_bytes, PAGE_SIZE):
        view[offset] = 0
    return buf


def open_fd(path, write, direct):
    flags = os.O_RDONLY
    if write:
        flags = os.O_CREAT | os.O_RDWR
    if direct and hasattr(os, "O_DIRECT"):
        flags |= os.O_DIRECT
    return os.open(path, flags, 0o644)


def split_work(size_bytes, jobs, block_bytes):
    jobs = max(1, min(jobs, size_bytes // block_bytes))
    per_job = align_down(size_bytes // jobs, block_bytes)
    total = per_job * jobs
    return jobs, per_job, total


def bench_write(path, size_bytes, block_bytes, jobs, direct):
    setup_fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.ftruncate(setup_fd, size_bytes)
    os.close(setup_fd)

    jobs, per_job, total = split_work(size_bytes, jobs, block_bytes)
    barrier = threading.Barrier(jobs + 1)

    def worker(job_id):
        fd = open_fd(path, True, direct)
        buf = make_buffer(block_bytes)
        barrier.wait()
        base = job_id * per_job
        end = base + per_job
        offset = base
        while offset < end:
            os.pwritev(fd, [buf], offset)
            offset += block_bytes
        os.close(fd)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(jobs)]
    for thread in threads:
        thread.start()
    barrier.wait()
    start = time.perf_counter()
    for thread in threads:
        thread.join()
    end = time.perf_counter()
    return total, end - start


def bench_read(path, size_bytes, block_bytes, jobs, direct):
    jobs, per_job, total = split_work(size_bytes, jobs, block_bytes)
    barrier = threading.Barrier(jobs + 1)

    def worker(job_id):
        fd = open_fd(path, False, direct)
        buf = make_buffer(block_bytes)
        barrier.wait()
        base = job_id * per_job
        end = base + per_job
        offset = base
        while offset < end:
            os.preadv(fd, [buf], offset)
            offset += block_bytes
        os.close(fd)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(jobs)]
    for thread in threads:
        thread.start()
    barrier.wait()
    start = time.perf_counter()
    for thread in threads:
        thread.join()
    end = time.perf_counter()
    return total, end - start


def parse_jobs(values):
    return [int(v) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path(__file__).with_name("io_bandwidth_test.bin"))
    parser.add_argument("--size-mb", type=int, default=DEFAULT_SIZE_MB)
    parser.add_argument("--block-mb", type=int, default=DEFAULT_BLOCK_MB)
    parser.add_argument("--jobs", type=int, nargs="+", default=DEFAULT_JOBS)
    parser.add_argument("--mode", choices=["read", "write", "both"], default="both")
    parser.add_argument("--buffered", action="store_true")
    args = parser.parse_args()

    size_bytes = align_down(args.size_mb * 1024 * 1024, PAGE_SIZE)
    block_bytes = align_down(args.block_mb * 1024 * 1024, PAGE_SIZE)
    if size_bytes <= 0 or block_bytes <= 0:
        raise SystemExit("size-mb and block-mb must be positive")
    if size_bytes < block_bytes:
        raise SystemExit("size-mb must be >= block-mb")

    direct = not args.buffered and hasattr(os, "O_DIRECT")

    best_write = (0.0, 0)
    best_read = (0.0, 0)

    print(f"path={args.path}")
    print(f"size={size_bytes // (1024 * 1024)} MiB block={block_bytes // (1024 * 1024)} MiB direct={direct}")

    for job_count in parse_jobs(args.jobs):
        if args.mode in ("write", "both"):
            total, seconds = bench_write(args.path, size_bytes, block_bytes, job_count, direct)
            rate = gbps(total, seconds)
            print(f"write jobs={job_count}: {rate:.2f} GB/s")
            if rate > best_write[0]:
                best_write = (rate, job_count)

        if args.mode in ("read", "both"):
            total, seconds = bench_read(args.path, size_bytes, block_bytes, job_count, direct)
            rate = gbps(total, seconds)
            print(f"read  jobs={job_count}: {rate:.2f} GB/s")
            if rate > best_read[0]:
                best_read = (rate, job_count)

    if args.mode in ("write", "both"):
        print(f"best write: {best_write[0]:.2f} GB/s at jobs={best_write[1]}")
    if args.mode in ("read", "both"):
        print(f"best read : {best_read[0]:.2f} GB/s at jobs={best_read[1]}")


if __name__ == "__main__":
    main()
