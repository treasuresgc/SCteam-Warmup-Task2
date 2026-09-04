#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import threading
from pathlib import Path

import numpy as np
import torch


DEFAULT_M = 200_000
DEFAULT_N = 200_000
DEFAULT_K = 200_000
DEFAULT_BIG_BLOCK = 100_000
DEFAULT_TILE = 25_000
DTYPE = np.float64
TORCH_DTYPE = torch.float64

QUADRANT_STEPS = {
    "C00": (("A00", "B00"), ("A01", "B10")),
    "C01": (("A00", "B01"), ("A01", "B11")),
    "C11": (("A10", "B01"), ("A11", "B11")),
    "C10": (("A10", "B00"), ("A11", "B10")),
}

QUADRANT_BLOCKS = {
    "C00": (0, 0),
    "C01": (0, 1),
    "C11": (1, 1),
    "C10": (1, 0),
}

BLOCK_POSITIONS = {
    "A00": (0, 0),
    "A01": (0, 1),
    "A10": (1, 0),
    "A11": (1, 1),
    "B00": (0, 0),
    "B01": (0, 1),
    "B10": (1, 0),
    "B11": (1, 1),
}


def parse_meta(path: Path) -> dict[str, str]:
    meta = {}
    if not path.exists():
        return meta

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def matrix_shape(meta: dict[str, str]) -> tuple[int, int, int]:
    rows = int(meta.get("M", DEFAULT_M))
    cols = int(meta.get("N", DEFAULT_N))
    inner = int(meta.get("K", DEFAULT_K))
    return rows, cols, inner


def block_view(matrix: np.memmap, block_row: int, block_col: int, big_block: int) -> np.ndarray:
    row0 = block_row * big_block
    col0 = block_col * big_block
    return matrix[row0 : row0 + big_block, col0 : col0 + big_block]


def tile_view(block: np.ndarray, tile_row: int, tile_col: int, tile: int) -> np.ndarray:
    row0 = tile_row * tile
    col0 = tile_col * tile
    return block[row0 : row0 + tile, col0 : col0 + tile]


def make_live_blocks(
    a_mm: np.memmap,
    b_mm: np.memmap,
    c_mm: np.memmap,
    quadrant: str,
    big_block: int,
) -> dict[str, np.ndarray]:
    c_row_block, c_col_block = QUADRANT_BLOCKS[quadrant]
    live = {
        "A00": block_view(a_mm, *BLOCK_POSITIONS["A00"], big_block),
        "A01": block_view(a_mm, *BLOCK_POSITIONS["A01"], big_block),
        "A10": block_view(a_mm, *BLOCK_POSITIONS["A10"], big_block),
        "A11": block_view(a_mm, *BLOCK_POSITIONS["A11"], big_block),
        "B00": block_view(b_mm, *BLOCK_POSITIONS["B00"], big_block),
        "B01": block_view(b_mm, *BLOCK_POSITIONS["B01"], big_block),
        "B10": block_view(b_mm, *BLOCK_POSITIONS["B10"], big_block),
        "B11": block_view(b_mm, *BLOCK_POSITIONS["B11"], big_block),
        "C": block_view(c_mm, c_row_block, c_col_block, big_block),
    }
    return live


def split_rows(num_rows: int, workers: int) -> list[tuple[int, int]]:
    if workers <= 0:
        return []
    chunk = math.ceil(num_rows / workers)
    ranges = []
    start = 0
    while start < num_rows:
        stop = min(start + chunk, num_rows)
        ranges.append((start, stop))
        start = stop
    return ranges


def copy_tile(dst: np.ndarray, src: np.ndarray, tile_row: int, tile_col: int, tile: int) -> None:
    np.copyto(dst, tile_view(src, tile_row, tile_col, tile))


def worker_quadrant(
    device_index: int,
    quadrant: str,
    live: dict[str, np.ndarray],
    row_begin: int,
    row_end: int,
    tile: int,
    tiles_per_big_block: int,
    errors: list[tuple[int, str, BaseException]],
) -> None:
    try:
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")

        a_host = torch.empty((tile, tile), device="cpu", dtype=TORCH_DTYPE, pin_memory=True)
        b_host = torch.empty((tile, tile), device="cpu", dtype=TORCH_DTYPE, pin_memory=True)
        c_host = torch.empty((tile, tile), device="cpu", dtype=TORCH_DTYPE, pin_memory=True)

        a_cpu = a_host.numpy()
        b_cpu = b_host.numpy()
        c_cpu = c_host.numpy()

        a_gpu = torch.empty((tile, tile), device=device, dtype=TORCH_DTYPE)
        b_gpu = torch.empty((tile, tile), device=device, dtype=TORCH_DTYPE)
        c_gpu = torch.zeros((tile, tile), device=device, dtype=TORCH_DTYPE)

        for local_row in range(row_begin, row_end):
            for local_col in range(tiles_per_big_block):
                c_gpu.zero_()
                for a_name, b_name in QUADRANT_STEPS[quadrant]:
                    a_block = live[a_name]
                    b_block = live[b_name]
                    for inner_k in range(tiles_per_big_block):
                        copy_tile(a_cpu, a_block, local_row, inner_k, tile)
                        copy_tile(b_cpu, b_block, inner_k, local_col, tile)
                        a_gpu.copy_(a_host)
                        b_gpu.copy_(b_host)
                        c_gpu.addmm_(a_gpu, b_gpu, beta=1.0, alpha=1.0)

                c_host.copy_(c_gpu)
                out_row0 = local_row * tile
                out_col0 = local_col * tile
                live["C"][out_row0 : out_row0 + tile, out_col0 : out_col0 + tile] = c_cpu
    except BaseException as exc:  # pragma: no cover - propagated to main thread
        errors.append((device_index, quadrant, exc))


def run_quadrant(
    quadrant: str,
    devices: list[int],
    live: dict[str, np.ndarray],
    tile: int,
    big_block: int,
) -> None:
    tiles_per_big_block = big_block // tile
    row_ranges = split_rows(tiles_per_big_block, len(devices))

    errors: list[tuple[int, str, BaseException]] = []
    threads: list[threading.Thread] = []

    print(f"{quadrant}: {' + '.join(f'{a}*{b}' for a, b in QUADRANT_STEPS[quadrant])}")

    for device_index, (row_begin, row_end) in zip(devices, row_ranges):
        if row_begin >= row_end:
            continue
        thread = threading.Thread(
            target=worker_quadrant,
            args=(device_index, quadrant, live, row_begin, row_end, tile, tiles_per_big_block, errors),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if errors:
        device_index, quadrant_name, exc = errors[0]
        raise RuntimeError(f"GPU{device_index} failed while computing {quadrant_name}") from exc


def resolve_devices(requested: list[int] | None) -> list[int]:
    available = torch.cuda.device_count()
    if available <= 0:
        raise SystemExit("CUDA is required for cases/05_extreme_200k/code.py")

    if requested is None:
        requested = list(range(min(2, available)))

    devices = []
    for device in requested:
        if device < 0 or device >= available:
            raise SystemExit(f"device index {device} is not available")
        if device not in devices:
            devices.append(device)
    if not devices:
        raise SystemExit("no CUDA devices selected")
    return devices


def load_matrices(data_dir: Path, rows: int, cols: int) -> tuple[np.memmap, np.memmap, np.memmap]:
    a_path = data_dir / "A.bin"
    b_path = data_dir / "B.bin"
    c_path = data_dir / "C.bin"

    if not a_path.exists():
        raise SystemExit(f"missing input: {a_path}")
    if not b_path.exists():
        raise SystemExit(f"missing input: {b_path}")

    a_mm = np.memmap(a_path, mode="r", dtype=DTYPE, shape=(rows, cols))
    b_mm = np.memmap(b_path, mode="r", dtype=DTYPE, shape=(cols, cols))
    c_mm = np.memmap(c_path, mode="w+", dtype=DTYPE, shape=(rows, cols))
    return a_mm, b_mm, c_mm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--big-block", type=int, default=DEFAULT_BIG_BLOCK)
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--devices", type=int, nargs="+")
    args = parser.parse_args()

    meta = parse_meta(args.data_dir / "meta.txt")
    rows, cols, inner = matrix_shape(meta)
    if rows != cols or cols != inner:
        raise SystemExit("code.py only handles square matrices")
    if rows != 2 * args.big_block:
        raise SystemExit(f"matrix size {rows} does not match 2 * big_block {args.big_block}")
    if args.big_block % args.tile != 0:
        raise SystemExit("big-block size must be divisible by tile size")

    devices = resolve_devices(args.devices)
    a_mm, b_mm, c_mm = load_matrices(args.data_dir, rows, cols)

    for quadrant in ("C00", "C01", "C11", "C10"):
        live = make_live_blocks(a_mm, b_mm, c_mm, quadrant, args.big_block)
        run_quadrant(quadrant, devices, live, args.tile, args.big_block)
        c_mm.flush()

    print(f"wrote {args.data_dir / 'C.bin'}")


if __name__ == "__main__":
    main()
