# filelist.md

说明：以下按当前工作树整理。`cases/05_extreme_200k/` 是当前极限规模分支，旧的 `cases/07_extreme_200k/` 已从工作树移除，不再列入清单。

## 根目录
- `.gitignore`：忽略本地 metadata、编译产物、Nsight 报告和临时文件。
- `.gitattributes`：统一仓库文本文件的自动识别与换行规则。
- `Readme.md`：项目总览，说明各 case 的规模、实现路线和构建方式。
- `AGENT.md`：后续 agent 的工作规范、日志要求和检查清单。
- `GPU_CONFIG.md`：当前可见 GPU、CUDA 版本和硬件资源记录。

## `cases/01_tiny`
- `build.sh`：用 `g++` 编译 `gen.cpp` 和 `code.cpp` 生成 `main`。
- `gen.cpp`：生成 32x32 的双精度随机矩阵并调用 `run(...)`。
- `code.cpp`：CPU 三重循环的最小规模矩阵乘法基线。

## `cases/02_small_256`
- `build.sh`：用 `nvcc` 编译 `gen.cpp` 和 `code.cu`。
- `gen.cpp`：生成 256x256 的双精度输入并调用 `run(...)`。
- `code.cu`：基础 CUDA 版本，一个线程负责一个输出元素。

## `cases/03_medium_5k`
- `build.sh`：默认编译基础 CUDA 版和 shared-memory 版，并支持 profiling 输出。
- `gen.cpp`：生成 4096x4096 的双精度输入并调用 `run(...)`。
- `code.cu`：基础 CUDA core 版本，未做共享内存分块。
- `code2.cu`：`8x32x32` 的 shared-memory CTA tile 版本。
- `code3.cu`：历史的大 CTA / register tiling 原型，保留用于实验对比，不在默认构建里编译。
- `experiment_summary.md`：记录 `03_medium_5k` 的实验过程、性能数据和结论。

## `cases/04_large_30k`
- `build.sh`：逐个编译 `code0.cu` 到 `code7.cu` 以及 `code_old.cu`，生成 `main*` 系列可执行文件。
- `gen.cpp`：生成 32768x32768 的双精度输入并调用 `run(...)`。
- `code0.cu`：cuBLAS 基线实现。
- `code1.cu`：直接使用 WMMA / Tensor Core 的实现。
- `code2.cu`：在 WMMA 基础上加入 shared-memory staging 的版本。
- `code3.cu`：`128x128x16` 的大 CTA WMMA 版本，warp tile 进一步放大。
- `code4.cu`：带 padding 和 async 预取的双缓冲 WMMA 版本。
- `code5.cu`：packed shared-memory 的双缓冲 WMMA 版本，采用更激进的块化加载。
- `code6.cu`：另一版 packed 双缓冲 WMMA 变体，调整了 warp / 块划分和搬运方式。
- `code7.cu`：`__launch_bounds__` 约束下的 packed WMMA 变体，按每线程一个 `double2` 组织搬运。
- `code_old.cu`：旧的 CUDA-core shared-memory tiling 版本，作为历史 fallback 保留。

## `cases/05_extreme_200k`
- `build.sh`：只编译 `gen.cpp`，生成磁盘数据生成器 `main`。
- `gen.cpp`：生成 200000x200000 的 FP64 矩阵并写入 `data/A.bin`、`data/B.bin` 和 `data/meta.txt`。
- `code.py`：两张 GPU 的分块矩阵乘法 baseline，按 `C00 -> C01 -> C11 -> C10` 顺序计算。
- `test.py`：H2D / D2H 带宽测试脚本。
- `test2.py`：磁盘顺序读写带宽测试脚本，支持 direct I/O 和多线程。
- `test3.py`：`torch.matmul` 性能基准脚本，测 FP64 `32768x32768` 乘法。

## `cases/test`
- `test.cu`：最小 CUDA hello-world / smoke test。

## `logs`
- `logs/codex-*.md`：按时间记录的工作日志，记录每次修改、命令和结果，不属于算法源码。

## 附录：生成物与缓存
- `cases/05_extreme_200k/data/`：运行 `gen.cpp` 和 `code.py` 产生的矩阵与结果文件。
- `cases/05_extreme_200k/__pycache__/`：Python 字节码缓存。
- `cases/05_extreme_200k/gen`：`gen.cpp` 编出来的可执行文件。
- `cases/test/tesgt`：测试用可执行文件产物。
- `testfile`：根目录数据文件，非源码。
- `cases/05_extreme_200k/testfile`：数据文件，非源码。
- `cases/05_extreme_200k/io_bandwidth_test.bin`：`test2.py` 的磁盘带宽测试文件。
- `cases/*/main*`：各 case 的构建产物。
- `cases/*/report*.nsys-rep` / `cases/*/report*.sqlite`：Nsight Systems profiling 产物。
