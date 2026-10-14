# 8-GPU P2P bandwidth test

直接运行的单机 GPU P2P 带宽测试。**不需要下载模型，不依赖 Python、PyTorch、FlashInfer 或 NCCL。** 只需要 NVIDIA 驱动、CUDA Toolkit（包含 `nvcc`）和 CUDA 支持的 C++ 编译器。

默认使用 8 张可见 GPU，逐个测试全部 **56 个有向 GPU 对**，输出 8×8 带宽矩阵。一次只有一对 GPU 在传输，不测八卡并发总带宽

## 一键运行

```bash
git clone https://github.com/lishunyang12/gpu-p2p-bandwidth.git
cd gpu-p2p-bandwidth
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
```

脚本在目标机器上编译，无预编译的 x86 二进制；ARM64 主机同样使用本机 CUDA 工具链。目标机器的驱动与 Toolkit 必须支持其 GPU。

若 `nvcc` 不在 PATH 中：

```bash
CUDA_HOME=/usr/local/cuda bash run.sh
```

只测一个方向，例如 GPU 0 → GPU 1：

```bash
bash run.sh --pair 0 1
```

只有两张卡时可以运行 `bash run.sh --gpus 2`。GPU 编号均为 `CUDA_VISIBLE_DEVICES` 映射后的可见序号；日志同时记录 PCI bus ID，方便对应物理 GPU。

## 输出与测量口径

每次结果写入独立的 `results/<UTC时间>-<进程号>/` 目录，可将整个目录发回：

- `benchmark.txt`：GPU 型号、PCI bus ID、逐对结果和带宽矩阵。
- `bandwidth.csv`：每对 GPU 的中位数、各 trial 的带宽和正确性状态。
- `topology.txt`、`p2p-read.txt`、`p2p-write.txt`：`nvidia-smi` 拓扑与 P2P 能力信息（命令可用时）。
- `nvidia-smi.txt`、`environment.txt`：驱动、CUDA、运行参数及设备可见性。

默认每次拷贝 **256 MiB**，每对 GPU 预热 3 次，测量 3 个 trial，每个 trial 连续拷贝 20 次。CUDA event 包住同一 stream 上的整批 `cudaMemcpyPeerAsync`，带宽为 `拷贝字节数 × 次数 / GPU耗时`，使用十进制 **GB/s**。矩阵显示各 trial 带宽的中位数：行是源 GPU，列是目标 GPU，单向字节只计算一次。

每对 GPU 在计时前后都会校验整个缓冲区的非均匀数据。分配、初始化、主机传输、校验和预热不计入带宽。脚本检查双向 peer-access 能力并启用 peer access；不支持的对记录 `unsupported`，不会作为主机中转带宽填入矩阵。退出码：0 表示所有请求的 GPU 对通过，2 表示存在不支持的 GPU 对，1 表示运行或校验错误。

可以增大数据量与重复数：

```bash
bash run.sh --size-mib 512 --warmup 5 --iterations 50 --trials 5
```

默认每个活动 GPU 分配 256 MiB，主机校验缓冲区合计约 512 MiB。应在 GPU 空闲时测量，以避免其他任务影响结果。

## PCIe / NVLink

这是 CUDA P2P 实际路径的带宽测试。**`cudaMemcpyPeerAsync` 不提供强制绕过 NVLink、只走 PCIe 的开关。** 如果目标八卡机器使用 NVLink/NVSwitch，测得的是该硬件实际选择的 P2P 路径，不能标成 PCIe 带宽。请结合保存的拓扑信息解释结果，必要时由机器管理员确认物理链路。

测量方式参考 [NVIDIA CUDA 多 GPU 编程文档](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)。

## 验证状态

已通过 CUDA 13.0 编译、命令行参数与 shell 语法检查。2026-10-14 在本机空闲的 5 张 GPU 上执行默认参数测试：20 个有向 GPU 对全部完成，所有完整缓冲区校验通过，覆盖 PXB、NODE 和 SYS 拓扑。另完成两卡短测试。其余 GPU 当时有活动负载，因此尚未完成整机八卡验证，也未在同事的目标机器上测试；目标机器的运行结果是最终验证。
