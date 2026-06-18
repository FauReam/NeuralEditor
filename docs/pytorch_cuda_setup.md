# PyTorch CUDA 版安装记录

## 环境概况

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 4070 (12 GB) |
| Driver | 591.86, CUDA 13.1 |
| Python | 3.11.9, venv: `C:\Users\46326\NeuralEditor\.venv` |
| 目标 | `torch>=2.1.0` + CUDA 支持 |
| 所需 wheel | `torch-2.5.1+cu121-cp311-cp311-win_amd64.whl` (~2.4 GB) |

---

## 问题历程

### 问题 1: 清华源只提供 CPU 版

清华源（`pypi.tuna.tsinghua.edu.cn`）装了 `torch-2.12.0+cpu`，无 CUDA DLL。

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
# → False
```

**根因：** 国内 PyPI 镜像不包含 CUDA tag 的 PyTorch wheel。

---

### 问题 2: PyTorch 官方下载频繁断流

`download.pytorch.org` 直连 2.4 GB 文件，下载中间 `IncompleteRead`：

```
Downloading torch-2.5.1+cu121-cp311-cp311-win_amd64.whl (2449.4 MB)
  465 MB / 2449 MB ── IncompleteRead
ERROR: ProtocolError: Connection broken
```

pip 自带的下载器不支持断点续传，中断后必须重头开始。

---

### 问题 3: aria2 未安装

```powershell
aria2c → CommandNotFoundException
```

需要 `winget install aria2` 或 `scoop install aria2` 后才能用。

---

### 问题 4: 内联 Python 脚本编码问题

`-c` 传入的字符串中 `%2B` 被 PowerShell/Python 解析器误识别：

```
SyntaxError: invalid decimal literal
```

**解决：** 写成独立 `.py` 脚本文件执行。

---

## 推荐方案（最终结论）

### 方案 A: aria2 分段下载（最快）

```powershell
# 安装 aria2
winget install aria2

# 下载 (16 线程, 断点续传)
aria2c -x 16 -s 16 -k 1M -o torch_cu121.whl `
  "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl" `
  -d "$env:TEMP"

# 安装
.\.venv\Scripts\python.exe -m pip install `
  "$env:TEMP\torch_cu121.whl" --force-reinstall
```

### 方案 B: 浏览器下载 + 本地安装

1. 浏览器打开: `https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl`
2. 下载到 `C:\Users\46326\NeuralEditor\`
3. 安装:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  C:\Users\46326\NeuralEditor\torch-2.5.1+cu121-cp311-cp311-win_amd64.whl `
  --force-reinstall
```

### 方案 C: 代理加速

```powershell
.\.venv\Scripts\python.exe -m pip install torch `
  --index-url https://download.pytorch.org/whl/cu121 `
  --proxy http://127.0.0.1:7890 --force-reinstall
```

---

## 验证命令

```powershell
.\.venv\Scripts\python.exe -c `
  "import torch; `
   print('CUDA:', torch.cuda.is_available()); `
   print('GPU:', torch.cuda.get_device_name(0)); `
   print('VRAM:', torch.cuda.get_device_properties(0).total_mem / 1e9, 'GB')"
```

预期输出:

```
CUDA: True
GPU: NVIDIA GeForce RTX 4070
VRAM: 11.99 GB
```

---

## 已知断点文件

`C:\Users\46326\AppData\Local\Temp\torch_cu121.whl` — 已下载约 352 MB，aria2 可从此字节续传。
