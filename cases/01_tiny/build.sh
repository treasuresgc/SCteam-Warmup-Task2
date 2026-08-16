#!/usr/bin/env bash
set -euo pipefail

g++ gen.cpp code.cpp -O2 -std=c++17 -o main
