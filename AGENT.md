# Agent 工作说明

本项目是 CUDA 矩阵乘法优化练习，当前已有 CPU 朴素实现、基础 CUDA 实现、shared memory 分块版本、每线程多输出版本，以及大规模目录中的 cuBLAS 基线和 FP64 Tensor Core / WMMA 实验版本。后续参与工作的 Agent 应先阅读根目录 `Readme.md`，再根据本文件继续推进。

## 强制要求：所有 Agent 必须写 log

所有参与本项目工作的 Agent 都必须记录工作日志。没有日志的改动视为不可追踪改动。

建议在 `logs/` 目录下新增独立日志文件，命名格式：

```text
logs/<agent-name>-YYYYMMDD-HHMM.md
```

每次工作至少记录：

- Agent 名称或标识。
- 开始时间和结束时间。
- 修改了哪些文件。
- 运行了哪些构建、测试或 profiling 命令。
- 命令是否成功；失败时写明错误信息和下一步建议。
- 尚未解决的问题和风险。

如果 `logs/` 目录不存在，先创建该目录。不要把日志写进 Nsight 报告文件、二进制产物或临时输出里。

## 当前未完成或需要继续确认的部分

### 1. 缺少正确性验证

当前各 case 的 `gen.cpp` 只生成输入并调用 `run(...)`，没有把 GPU 结果与 CPU reference 或 cuBLAS reference 做数值比较。后续应补充：

- 小规模 CPU reference 校验。
- `float` 和 `double` 的误差阈值。
- 各 CUDA 版本输出一致性比较。
- 对非整除 tile 尺寸的边界测试。

### 2. 缺少统一 benchmark 脚本

项目中已有多个 `build.sh` 和 Nsight Systems 报告，但没有统一脚本来复现实验。后续可补充：

- 一键构建全部可运行 case 的脚本。
- 分别运行各 case 实际生成的可执行文件，并记录 benchmark 结果。
- 自动收集 kernel 时间、总运行时间和设备信息。
- 固定输出目录，避免覆盖已有 profiling 结果。

### 3. `cases/07_extreme_200k` 规模不可直接运行

`cases/07_extreme_200k/gen.cpp` 当前实际配置是：

- `M = 2048000`
- `N = 2048000`
- `K = 2048000`

这个规模会产生 TB 级内存需求，普通机器无法直接运行。该目录需要明确后续方向：

- 改成真实可运行的压力测试规模。
- 或实现分块/流式矩阵乘法。
- 或改为 cuBLAS/多 GPU 实验。
- 或仅保留为极限规模占位，并在脚本中默认禁止执行。

另外，该目录的 `build.sh` 链接了 `-lcublas`，但当前 `code.cu` 没有实际调用 cuBLAS API。不要在文档或日志中声称它已经是 cuBLAS 实现。

### 4. Tensor Core / WMMA 版本需要架构约束

`cases/04_large_30k/code0.cu` 是 cuBLAS 基线实现，`cases/04_large_30k/code1.cu`、`cases/04_large_30k/code2.cu` 和 `cases/04_large_30k/code4.cu` 是 WMMA 相关实现。`03_medium_5k` 已移除 `code4.cu`，当前只保留 CUDA core 优化路径。后续 Agent 修改这部分时必须确认：

- 编译架构是否支持目标 WMMA 指令。
- `-arch=sm_80` 或更高架构是否适合当前机器。
- 如果重新加入低架构 fallback，是否仍能编译和运行。
- 如果重新加入或修改 padding 路径，在非对齐维度下是否正确。

### 5. 边界写入需要重点检查

`code3.cu` 系列每个线程计算 `2x2` 输出。当前 case 的矩阵尺寸是偶数且能整除主要 tile，因此一般不会触发边界问题。但 `run(...)` 接口本身接收任意 `m`、`n`、`k`，后续如果要把实现泛化，必须检查：

- `col + 1 < n` 时才能写第二列。
- `row + 1 < m` 时才能写第二行。
- 不能只用 `if (row < m && col < n)` 包住四个输出写入。

### 6. Profiling 产物需要管理

当前仓库中已有多个 `report*.nsys-rep` 和 `report*.sqlite`。后续 Agent 重新 profiling 时应：

- 不覆盖已有报告，除非任务明确要求。
- 记录 GPU 型号、CUDA 版本、编译参数和运行命令。
- 把新报告编号或按日期命名。
- 在日志中说明新报告对应哪个可执行文件。

## 工作流程建议

1. 阅读 `Readme.md` 和本文件。
2. 写入或创建本次工作的 log 文件。
3. 明确本次只处理一个主题，例如 correctness、benchmark、WMMA、extreme case 或文档。
4. 修改前先检查相关目录的 `build.sh`、`gen.cpp` 和 `code*.cu`。
5. 修改后尽量运行最小可行验证。
6. 把验证结果、未运行原因或失败原因写入 log。

## 构建注意事项

- `cases/01_tiny` 使用 `g++`。
- 其他 CUDA case 主要使用 `nvcc`。
- `cases/03_medium_5k` 和 `cases/04_large_30k` 编译时指定 `-arch=sm_80`。
- `cases/07_extreme_200k` 指定 `/usr/local/cuda-13.1`，不同环境可能不存在该路径。

不要默认运行大规模 case，尤其是 `cases/04_large_30k` 和 `cases/07_extreme_200k`。如果需要运行，先确认机器内存、显存、CUDA 版本和 GPU 架构。

## 提交前检查

每个 Agent 完成工作前应确认：

- 已更新自己的 log。
- 没有误删已有 profiling 报告。
- 没有提交无关二进制产物。
- 文档描述与源码一致。
- 如果未运行测试，已在 log 中说明原因。
