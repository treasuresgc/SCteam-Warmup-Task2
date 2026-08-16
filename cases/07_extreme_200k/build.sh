#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT=/usr/local/cuda-13.1

nvcc gen.cpp code.cu -O2 -std=c++17 \
    -I"${CUDA_ROOT}/targets/x86_64-linux/include" \
    -L"${CUDA_ROOT}/targets/x86_64-linux/lib" \
    -lcublas \
    -Xlinker -rpath -Xlinker "${CUDA_ROOT}/targets/x86_64-linux/lib" \
    -o main
