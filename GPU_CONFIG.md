# GPU Configuration

记录时间：2026-08-02 20:49 CST

本文档记录当前机器可见的 GPU 配置，以及 CUDA 矩阵乘法优化时需要关注的硬件参数。

## 当前 GPU 清单

通过以下命令查询：

```bash
conda run -n Matmul nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,memory.used,compute_cap,mig.mode.current,driver_version --format=csv,noheader,nounits
```

查询结果：

| GPU | 型号 | UUID | Bus ID | Compute Capability | 总显存 | 空闲显存 | 已用显存 | MIG | Driver |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | NVIDIA A800 80GB PCIe | `GPU-985f4b24-f86e-d290-a4c6-351507e29d36` | `00000000:8A:00.0` | 8.0 | 81920 MiB | 81176 MiB | 0 MiB | Disabled | 610.43.02 |
| 1 | NVIDIA A800 80GB PCIe | `GPU-aac4934f-6b11-6baa-acd1-2c76f4085695` | `00000000:D6:00.0` | 8.0 | 81920 MiB | 81176 MiB | 0 MiB | Disabled | 610.43.02 |

## 单卡关键规格

以下规格按 NVIDIA A800 80GB PCIe / Ampere GA100 / Compute Capability 8.0 记录。

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
| Maximum number of resident blocks / SM (`resident_blocks_per_SM`) | 32 |

说明：

- 这里的 `CUDA cores / SM` 指常规 CUDA core 计算单元，不包含 Tensor Core。
- 对本项目的 FP64 矩阵乘法，常规 FP64 CUDA cores 是 `32 / SM`；`code4.cu` 系列使用的 FP64 WMMA / Tensor Core 路径还要单独考虑 Tensor Core 支持。
- NVIDIA 的寄存器资源按 SM 管理，不是每个 CUDA core 私有固定分配。`1024 registers / FP32 CUDA core` 是用 `65536 registers / SM / 64 FP32 cores / SM` 得到的折算值，实际 occupancy 分析应使用 `registers / thread`、`threads / block`、`blocks / SM` 和 `65536 registers / SM`。
- `resident_blocks_per_SM` 这里指每个 SM 最多可同时驻留的线程块数。对于 Compute Capability 8.0，NVIDIA 官方表给出的上限是 `32`；实际 kernel 能达到的 active blocks per SM 还会受寄存器、shared memory 和线程数限制。
- Shared memory 与 L1 cache 在 Ampere 上存在可配置 carveout。实际 kernel 可用 shared memory 还会受编译参数、动态 shared memory、occupancy 和 opt-in 设置影响。

## CUDA 环境

CUDA Toolkit 13.1 已安装：

```bash
/usr/local/cuda-13.1/bin/nvcc --version
```

输出版本：

```text
Cuda compilation tools, release 13.1, V13.1.115
```

`nvidia-smi` 报告的驱动和运行时信息：

```text
Driver/KMD Version: 610.43.02
CUDA UMD Version: 13.3
```

## 当前环境限制

当前工作沙箱内没有暴露以下设备节点：

```text
/dev/nvidia0
/dev/nvidia1
/dev/nvidiactl
/dev/nvidia-uvm
```

因此直接运行 CUDA runtime 查询程序时，`cudaGetDeviceCount` 返回：

```text
cudaGetDeviceCount failed: no CUDA-capable device is detected
```

这意味着：

- `nvidia-smi` 能通过 `conda run -n Matmul` 读取 NVML GPU 清单。
- 当前沙箱不能直接运行依赖 CUDA runtime 设备枚举的程序。
- SM 数、shared memory、L2 cache 等细节在本文档中按 A800 80GB PCIe / GA100 规格记录，而不是从当前沙箱里的 `cudaDeviceProp` 直接读取。

## 对本项目的含义

- `cases/04_large_30k/build.sh` 使用 `-arch=sm_80`，与 A800 的 Compute Capability 8.0 匹配。
- FP64 WMMA / Tensor Core 实验可以面向 `sm_80` 编译，但实际运行仍需要确保 CUDA runtime 能访问 GPU。
- 每张卡 80 GiB 级别显存足够运行 `03_medium_5k`，但不够直接运行 `07_extreme_200k` 当前的 `2048000 x 2048000` 配置。
- 针对 shared memory tile 优化时，应以每 SM 164 KiB、单 block 默认 48 KiB、opt-in 约 160 KiB 作为资源上限。
- 针对寄存器压力优化时，应优先关注编译器报告的每线程寄存器数。例如 `registers/thread * threads/block` 不能超过每 block/SM 的寄存器限制，并且会影响每 SM 可同时驻留的 block/warp 数。

## 参考

- NVIDIA Ampere / A100 架构说明：https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- CUDA Compute Capability 说明：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
- NVIDIA Developer Forum 关于 A800 80GB L2 cache 的 `deviceQuery` 讨论：https://forums.developer.nvidia.com/t/l2cache-size-of-a800-80gb/289934
