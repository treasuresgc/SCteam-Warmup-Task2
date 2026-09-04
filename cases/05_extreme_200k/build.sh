#!/usr/bin/env bash
set -euo pipefail

CXX="${CXX:-g++}"
if ! command -v "${CXX}" >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    CXX=g++
fi

"${CXX}" gen.cpp -O2 -std=c++17 -o main
