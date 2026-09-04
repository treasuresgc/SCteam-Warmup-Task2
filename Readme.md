# Warmup Task2 项目说明

这个项目围绕矩阵乘法 `C = A * B` 展开，目标是从基础 CPU 实现逐步过渡到 CUDA GPU 实现，并在更大矩阵规模上尝试不同的性能优化策略。代码被放在 `cases/` 下的多个子目录中，每个目录对应一个输入规模或实验阶段。

## 项目在完成什么

项目的核心工作是计算两个稠密矩阵的乘积：

- `A` 的形状是 `m x k`
- `B` 的形状是 `k x n`
- `C` 的形状是 `m x n`
- 所有矩阵都按行优先的一维数组存储

每个 case 中的 `gen.cpp` 通常负责生成固定规模的输入矩阵，并调用对应实现文件中的 `run(...)` 函数完成矩阵乘法；极限规模 case 也可能先把输入写到磁盘，再由后续脚本读取。不同目录和不同 `code*.cu` / `code*.py` 文件展示了从朴素算法到 GPU 优化版本的演进。

## 目录结构

```text
cases/
  01_tiny/          32 x 32 x 32，CPU 版本，用于最小规模验证
  02_small_256/     256 x 256 x 256，基础 CUDA 版本
  03_medium_5k/     4096 x 4096 x 4096，多个 CUDA 优化版本和 Nsight 报告
  04_large_30k/     32768 x 32768 x 32768，较大规模 CUDA 优化实验
  05_extreme_200k/  200000 x 200000 的 FP64 磁盘分块实验
  test/             CUDA hello-world 测试程序
```

各目录通常包含：

- `gen.cpp`：生成输入矩阵；极限规模 case 也可能直接把输入落盘，供后续脚本读取。
- `code.cpp` / `code.cu` / `code.py`：矩阵乘法实现。
- `build.sh`：编译当前目录下的实现。
- `main`、`main0`、`main1`、`main2`、`main3`、`main4`：已编译出的可执行文件，具体取决于当前 case 的 `build.sh`。
- `report*.nsys-rep`、`report*.sqlite`：Nsight Systems 性能分析产物。

## 仓库约定

仓库根目录已配置 `.gitignore` 和 `.gitattributes`。
默认只保留源码、文档和日志，编译产物、Nsight Systems 报告、本地工具缓存和常见临时文件不会进入 Git 提交。

## 实现路线

### 1. CPU 朴素实现

`cases/01_tiny/code.cpp` 使用三重循环直接计算矩阵乘法：

```cpp
for i in m:
  for j in n:
    for p in k:
      C[i, j] += A[i, p] * B[p, j]
```

这是最容易理解的基准实现，适合验证计算逻辑，但复杂度是 `O(m * n * k)`，规模变大后速度会很慢。

### 2. 基础 CUDA 实现

`cases/02_small_256/code.cu` 和 `cases/03_medium_5k/code.cu` 中，一个 CUDA thread 负责计算 `C` 中的一个元素。kernel 使用二维 grid/block 映射矩阵行列：

- `threadIdx/blockIdx` 映射到 `row` 和 `col`
- 每个线程沿着 `k` 维度做完整点积
- 使用 `cudaMalloc`、`cudaMemcpy` 完成主机和设备之间的数据传输

这个版本已经把不同输出元素之间的并行性放到 GPU 上，但每个线程仍然从全局内存重复读取 `A` 和 `B`，内存访问效率不高。

### 3. 共享内存分块

`cases/03_medium_5k/code2.cu` 使用 `8 x 32 x 32` CTA tile，把 `A` 的 `8 x 32` 子块和 `B` 的 `32 x 32` 子块加载到 shared memory 后再计算。这样可以减少对 global memory 的重复访问，提高数据复用率。

这一版的关键点是：

- 每个 thread block 处理 `C` 的一个 `8 x 32` 子块
- 每轮加载一块 `A` 和一块 `B` 到 shared memory
- 每个 block 使用 `32 x 8` 个线程，每个线程只累加 `1` 个输出
- 通过 `__syncthreads()` 保证块内线程同步
- 当前实现面向本 case 的 `4096 x 4096 x 4096` 对齐尺寸，不保留非整除边界路径或 register tiling

### 4. 大规模 WMMA / Tensor Core 路线

`cases/04_large_30k/code0.cu` 使用 cuBLAS 作为基线实现。`cases/04_large_30k/code1.cu` 是直接 WMMA 版本。`cases/04_large_30k/code2.cu` 在 `code1.cu` 基础上加入 shared memory，`CtaK = 16`。`cases/04_large_30k/code3.cu` 是 `128 x 128 x 16` 的大 CTA WMMA 版本。`cases/04_large_30k/code4.cu` 保留更完整的 FP64 WMMA / padding 路径。`cases/04_large_30k/code5.cu` 是 row-major 的 cuBLASLt fast path。

主要特征：

- 使用 `nvcuda::wmma` fragment
- tile 形状为 `8 x 8 x 4`
- `code2.cu` 采用 shared memory 的 `8 x 16` / `16 x 128` CTA tile staging
- `code3.cu` 采用 shared memory 的 `128 x 128 x 16` CTA tile，`32 x 8` block；warp tile 为 `32 x 64`，8 个 warp 直接覆盖完整 CTA
- `04_large_30k` 的 WMMA 构建显式使用 `-arch=sm_80`

这一部分面向支持 FP64 Tensor Core 的 NVIDIA GPU，尤其是 Ampere 及更新架构。

## 各 case 的矩阵规模

| 目录 | 数据类型 | M | N | K | 主要用途 |
| --- | --- | ---: | ---: | ---: | --- |
| `cases/01_tiny` | `double` | 32 | 32 | 32 | CPU 正确性和流程验证 |
| `cases/02_small_256` | `double` | 256 | 256 | 256 | 基础 CUDA kernel |
| `cases/03_medium_5k` | `double` | 4096 | 4096 | 4096 | CUDA 优化对比和 profiling |
| `cases/04_large_30k` | `double` | 32768 | 32768 | 32768 | 大规模 FP64 优化实验 |
| `cases/05_extreme_200k` | `double` | 200000 | 200000 | 200000 | FP64 磁盘生成 + 分块计算流水线 |

注意：`cases/05_extreme_200k` 现在是 FP64 磁盘分支。`gen.cpp` 只负责把 `A/B` 写到磁盘，后续的 `code.py` 再从磁盘读取并完成 `C` 的计算与导出。

## 构建和运行

每个 case 目录都有自己的 `build.sh`。进入对应目录后执行：

```bash
./build.sh
```
然后运行该目录实际生成的可执行文件。

`cases/03_medium_5k/build.sh` 会生成两个版本：

- `main`：基础 CUDA 版本，对应 `code.cu`
- `main2`：shared memory 分块版本，对应 `code2.cu`

`cases/04_large_30k/build.sh` 会生成：

- `main0`：cuBLAS 基线版本，对应 `code0.cu`
- `main1`：直接 WMMA 版本，对应 `code1.cu`
- `main2`：shared memory WMMA 版本，对应 `code2.cu`
- `main3`：大 CTA WMMA 版本，对应 `code3.cu`
- `main4`：FP64 Tensor Core / WMMA 版本，对应 `code4.cu`
- `main5`：cuBLASLt fast path，对应 `code5.cu`

`cases/05_extreme_200k/build.sh` 会生成：

- `main`：FP64 `gen.cpp` 生成器，把 `A.bin` / `B.bin` 写到 `data/` 目录下，顺带生成 `meta.txt`

`cases/05_extreme_200k/code.py` 是 FP64 两卡分块实现，读取 `data/A.bin` 和 `data/B.bin`，按 `C00 -> C01 -> C11 -> C10` 的顺序计算，并把结果写到 `data/C.bin`。

`cases/05_extreme_200k/test.py` 是一个更直接的带宽测试脚本，只测 CPU/GPU 复制时间，不做矩阵乘法。

`cases/05_extreme_200k/test2.py` 是一个纯 Python I/O 带宽测试脚本，默认用 direct I/O 顺序读写同一个测试文件，并支持多线程扫 jobs。

`cases/05_extreme_200k/test3.py` 是一个 FP64 `torch.matmul` 基准脚本，默认测 `32768 x 32768 x 32768`，并逐个测试当前可见 GPU。

`code_old.cu` 是历史备份，不参与当前构建。

## 性能分析文件

`cases/03_medium_5k` 和 `cases/04_large_30k` 中的 `report*.nsys-rep` 与 `report*.sqlite` 是 Nsight Systems 生成的性能分析结果。它们可用于查看：

- CUDA kernel 执行耗时
- host-to-device 和 device-to-host 拷贝耗时
- kernel launch 时间线
- 不同实现版本之间的性能差异

如果需要重新采集，可以在对应目录下使用类似命令：

```bash
nsys profile -o report ./main0
```

## 依赖环境

运行 CUDA 版本需要：

- NVIDIA GPU
- CUDA Toolkit
- `nvcc`
- 支持目标 kernel 的 GPU 架构

其中 FP64 Tensor Core / WMMA 版本需要较新的 GPU 架构支持。`cases/04_large_30k/build.sh` 使用了 `-arch=sm_80`，表示面向 Ampere 架构编译。

`cases/05_extreme_200k` 的构建阶段只需要 `g++` 编译 `gen.cpp`；计算阶段用 `python3` 跑 `code.py`，并依赖 PyTorch + CUDA runtime 在两张 GPU 上执行。

## 总结

这个项目是一个 CUDA 矩阵乘法优化练习。它先用 CPU 朴素三重循环建立基础逻辑，再逐步引入 GPU 并行、shared memory 分块、每线程多输出和 FP64 Tensor Core / WMMA。目录中的 Nsight Systems 报告则用于分析这些实现的实际运行时间和瓶颈。
