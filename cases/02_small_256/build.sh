#!/usr/bin/env bash
set -euo pipefail

nvcc gen.cpp code.cu -O2 -std=c++17 -o main
