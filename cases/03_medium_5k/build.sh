#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "x" ]]; then
    NVCC="${NVCC:-nvcc}"
    if ! command -v "${NVCC}" >/dev/null 2>&1 && [[ -x /usr/local/cuda-13.1/bin/nvcc ]]; then
        NVCC=/usr/local/cuda-13.1/bin/nvcc
    fi

    "${NVCC}" gen.cpp codex.cu -O2 -std=c++17 -arch=sm_80 -o mainx
    ./mainx
    exit 0
fi

NVCC="${NVCC:-nvcc}"
if ! command -v "${NVCC}" >/dev/null 2>&1 && [[ -x /usr/local/cuda-13.1/bin/nvcc ]]; then
    NVCC=/usr/local/cuda-13.1/bin/nvcc
fi

"${NVCC}" gen.cpp code.cu -O2 -std=c++17 -arch=sm_80 -o main
"${NVCC}" gen.cpp code2.cu -O2 -std=c++17 -arch=sm_80 -o main2

if [[ "${1:-}" == "profile" ]]; then
    tag=$(date +%Y%m%d-%H%M%S)

    nsys profile -o "report-${tag}-main" ./main
    nsys export -t sqlite -o "report-${tag}-main.sqlite" "report-${tag}-main.nsys-rep"

    nsys profile -o "report-${tag}-main2" ./main2
    nsys export -t sqlite -o "report-${tag}-main2.sqlite" "report-${tag}-main2.nsys-rep"
fi
