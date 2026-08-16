# 03_medium_5k 历史实验总结

## 目标

在单卡 A800 上优化 `4096 x 4096 x 4096` 的双精度矩阵乘法。本目录当前聚焦手写 CUDA core 路径，对比基础线程映射、shared memory CTA tile，以及更大的 CTA tile + register tiling。
这里记录的是曾经在 `03_medium_5k` 上做过的实验，其中大 CTA 版本后来迁移到了 `04_large_30k/code3.cu`。

## A800 相关配置

| 参数 | 数值 |
| --- | ---: |
| GPU 架构 | Ampere GA100 |
| Compute Capability | 8.0 |
| SM 个数 | 108 |
| FP32 CUDA cores / SM | 64 |
| FP32 CUDA cores 总数 | 6912 |
| FP64 CUDA cores / SM | 32 |
| FP64 CUDA cores 总数 | 3456 |
| 32-bit registers / SM | 65536 |
| 32-bit registers / GPU | 7077888 |
| 折算 32-bit registers / FP32 CUDA core | 1024 |
| 最大 32-bit registers / thread | 255 |
| Tensor Cores / SM | 4 |
| Tensor Cores 总数 | 432 |
| L2 cache | 40 MiB，41943040 bytes |
| Shared memory / SM | 164 KiB，167936 bytes |
| 默认 shared memory / block | 48 KiB，49152 bytes |
| Opt-in shared memory / block | 约 160 KiB，163840 bytes |
| Warp size | 32 threads |
| Maximum resident blocks / SM | 32 |

## 1. BlockDim 和基础 CTA tile

首先选取 `BlockDim`。A800 的 warp size 是 `32`，因此让 `BlockDim.x = 32` 可以使一个 warp 访问一行中连续的 32 列，更符合全局内存访问和缓存行为。由于单个 block 最多 1024 个线程，`BlockDim.y` 可以取到 32；实际测试中不同 `y` 取值差距不大，因此当前使用：

```text
BlockDim = (32, 8)
```

基础 CUDA core 版本的 GPU 时间约为：

```text
61,050,273 ns
```

随后加入 CTA tile，将 `A` 和 `B` 的子块放入 shared memory。`code2.cu` 当前使用：

```cpp
constexpr int CtaM = 8;
constexpr int CtaN = 32;
constexpr int CtaK = 32;
```

该版本每个 block 处理 `8 x 32` 个输出，每轮加载 `8 x 32` 的 `A` tile 和 `32 x 32` 的 `B` tile。测试时间约为：

```text
54,768,645 ns
```

这个提升并不大，主要原因是该 tile 的算术强度仍然较低，global memory 读写压力还没有被充分摊薄。

## 2. Register tiling 和大 CTA tile

对 CTA tile 记为 `m x n x k`。每轮需要读取 `m * k + k * n` 个 FP64，约为：

```text
8 * k * (m + n) bytes
```

计算量约为：

```text
2 * m * n * k FLOPs
```

因此算术强度为：

```text
I = m * n / (4 * (m + n))
```

在 `m * n` 固定时，`m` 和 `n` 越接近，算术强度越高。另一方面，每线程输出数量受寄存器限制约束。A800 最大 `255` 个 32-bit registers / thread，而一个 FP64 累加器占用两个 32-bit registers，因此单线程 FP64 输出数量不能过大。按 `32 x 8 = 256` 个线程估算，CTA 输出数量控制在：

```text
m * n <= 32 * 8 * 64 = 16384
```

因此选取：

```text
CtaM = 128
CtaN = 128
CtaK = 32
```

其中 `CtaK = 32` 可以减少循环轮数，同时 shared memory 用量为：

```text
(128 * 32 + 32 * 128) * sizeof(double) = 65536 bytes
```

该用量超过默认 48 KiB shared memory / block，因此 `code3.cu` 需要通过 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` 打开动态 shared memory opt-in。

大 CTA 版本每个线程计算 `16 x 4 = 64` 个 FP64 输出。测试时间约为：

```text
22,262,756 ns
```

提升幅度明显。估算上，`8 x 32` tile 的算术强度约为：

```text
I = 8 * 32 / (4 * (8 + 32)) = 1.6 FLOP/byte
```

而 `128 x 128` tile 的算术强度约为：

```text
I = 128 * 128 / (4 * (128 + 128)) = 16 FLOP/byte
```

结合 A800 的经验临界点约 `6.2 FLOP/byte`，大 CTA 版本更接近 compute bound，因此优化效果明显。

## 当前结论

`03_medium_5k` 当前保留两个 CUDA core 版本：

- `code.cu`：基础 CUDA core 版本。
- `code2.cu`：`8 x 32 x 32` shared memory CTA tile。

`code3.cu` 的 `128 x 128 x 32` 大 CTA tile + register tiling 实现已迁移到 `04_large_30k/code3.cu`。

`code4.cu` 的 FP64 WMMA / Tensor Core 实验版本已从本目录移除，本目录后续只保留基础 CUDA core 和 shared memory CTA tile。

## 风险和后续工作

- `code2.cu` 和 `code3.cu` 当前面向 `4096 x 4096 x 4096` 对齐尺寸，不包含通用边界路径。
- `code3.cu` 编译时 `ptxas` 报告寄存器使用达到 `255 registers/thread`，虽然没有 spill，但 occupancy 会受到明显限制。
- 目前仍缺少 CPU reference 或 cuBLAS reference 正确性验证。
