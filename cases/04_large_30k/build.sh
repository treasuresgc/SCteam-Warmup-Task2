#!/usr/bin/env bash
set -euo pipefail

NVCC="${NVCC:-nvcc}"
if ! command -v "${NVCC}" >/dev/null 2>&1 && [[ -x /usr/local/cuda-13.1/bin/nvcc ]]; then
    NVCC=/usr/local/cuda-13.1/bin/nvcc
fi

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-13.1}"

for src in code0.cu code1.cu code2.cu code3.cu code4.cu code5.cu code6.cu code7.cu; do
    [[ -e "${src}" ]] || continue
    suffix="${src#code}"
    suffix="${suffix%.cu}"

    "${NVCC}" gen.cpp "${src}" -O2 -std=c++17 -arch=sm_80 \
        -I"${CUDA_ROOT}/targets/x86_64-linux/include" \
        -L"${CUDA_ROOT}/targets/x86_64-linux/lib" \
        -lcublasLt \
        -lcublas \
        -Xlinker -rpath -Xlinker "${CUDA_ROOT}/targets/x86_64-linux/lib" \
        -o "main${suffix}"
done

src="code_old.cu"
suffix="${src#code}"
suffix="${suffix%.cu}"

"${NVCC}" gen.cpp "${src}" -O2 -std=c++17 -arch=sm_80 \
    -I"${CUDA_ROOT}/targets/x86_64-linux/include" \
    -L"${CUDA_ROOT}/targets/x86_64-linux/lib" \
    -lcublas \
    -Xlinker -rpath -Xlinker "${CUDA_ROOT}/targets/x86_64-linux/lib" \
    -o "main${suffix}"
