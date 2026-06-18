---
name: neural-editor-ops
description: NeuralEditor / Heartscape Engine 运维操作。当用户提到 NeuralEditor、Heartscape、启动服务、tunnel、cloudflared、API Key、ROME、MEMIT、LoRA、GPU、服务异常、502、健康检查、模型加载时自动加载。加载后第一步：执行健康检查 localhost:8765/api/health，根据结果决定是否干预。
metadata:
  short-description: NeuralEditor 自进化运维 skill - 自动健康检查
  project-root: C:/Users/46326/NeuralEditor
  auto-trigger: true
  companion: references/commands.md
---

# NeuralEditor 运维 Skill (自动触发版)

**加载后第一步**: 执行 Invoke-RestMethod http://localhost:8765/api/health
- 如果返回 status: ok，报告用户服务正常，然后继续用户的其它任务
- 如果失败，自动进入 Skillopt 5 步诊断修复流程


## 自动触发机制

此 skill 通过以下机制实现无提示自动加载：
1. description 字段覆盖所有操作关键词 — agent 扫描 skill 列表时匹配率极高
2. project-root 设为 C:/Users/46326/NeuralEditor — agent 在该目录下工作时优先匹配
3. 加载即检查 — skill 第一条指令是健康检查，确保 agent 不遗漏服务状态
4. companion file references/commands.md — 提供精确可复制命令，减少 agent 试错
5. 项目级副本 — 同时存在于 ~/.codewhale/skills/ 和 NeuralEditor/.agents/skills/，双重保障

## 项目基础

| 项 | 值 |
|------|------|
| 项目路径 | C:/Users/46326/NeuralEditor |
| 虚拟环境 | .venv/Scripts/python.exe |
| 服务端口 | 8765 |
| 隧道工具 | cloudflared.exe (推荐) / ngrok.exe (中国大陆不可用) |
| LLM | models/Qwen2.5-7B-Instruct/ |
| GPU | RTX 4070, 12GB VRAM, 4-bit 量化 |
| 配置文件 | config/settings.yaml (LLM参数), config/characters/, config/scenes/ |
| 数据目录 | data/saves/, data/memories/(Chroma), data/sessions/ |

## 已知陷阱 (agent 必读)

当 agent cwd 在 C:/Windows/System32 时须注意:
- PowerShell 不支持 && — 用 ; 或分次执行
- 必须用 .venv/Scripts/python.exe 而非系统 python
- 启动服务必须用 python -c 方式，禁止 python -m src.web.server
- cloudflared 在校园网可能因 IPv6 失败 — 重试或加 --protocol http2
- ngrok 在中国大陆 DNS 劫持到 127.0.0.1 — 用 cloudflared 替代
- PowerShell 对 $、{}、中文引号、反引号敏感 — 精确命令见 references/commands.md
- 每次执行命令前先 cd C:/Users/46326/NeuralEditor


## Skillopt 5 步自进化运维管道

    Step 1: 诊断 -> Step 2: 策略选择 -> Step 3: 干预执行 -> Step 4: 验证 -> Step 5: 记录学习

### Step 1: 诊断 (类比 Skillopt DataPipeline)

采集指标:
    Invoke-RestMethod -Uri http://localhost:8765/api/health -ErrorAction SilentlyContinue
    Get-Process -Name python -ErrorAction SilentlyContinue | Select Id, CPU, WorkingSet64
    .venv/Scripts/python.exe -c import torch; print(torch.cuda.is_available())
    .venv/Scripts/python.exe -c import torch; print(torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0)
    Get-Content tunnel_url.txt -ErrorAction SilentlyContinue

诊断分类矩阵:
| health | process | LLM | tunnel | 结论 | 策略 |
|:---:|:---:|:---:|:---:|---------|------|
| OK | alive | loaded | OK | 正常 | 跳过 |
| OK | alive | not_loaded | - | LLM加载中 | 等待 |
| FAIL | alive | - | - | 端口异常 | warm_restart |
| FAIL | dead | - | - | 服务未运行 | cold_start |
| OK | alive | loaded | FAIL | 隧道异常 | tunnel_restart |

### Step 2: 策略选择 (类比 Skillopt ModelFactory)

| 诊断 | 策略 | 动作 |
|------|------|------|
| 服务未运行 | cold_start | 完整启动 (Step 3-A) |
| 服务无响应 | warm_restart | 杀进程重启 (Step 3-B) |
| LLM未加载 | wait_or_cuda_fix | 等待或修复CUDA (Step 3-C) |
| 隧道异常 | tunnel_restart | 重启隧道 (Step 3-D) |
| 频繁OOM | scale_down | 降低上下文长度 |
| 性能劣化 | benchmark | 对比配置性能 |

### Step 3: 干预执行

精确命令见 references/commands.md。
策略概要:

3-A cold_start: Stop-Process python,cloudflared -> sleep 2 -> 启动 .venv/Scripts/python.exe -c ... -> 轮询 health 直到 OK
3-B warm_restart: Stop-Process python -> sleep 3 -> 同 cold_start 启动
3-C LLM修复: 检查 torch.cuda.is_available() -> 若 False 则检查 torch.__version__ 是否含 +cpu -> 是则重装 CUDA 版 PyTorch
3-D 隧道重启: Stop-Process cloudflared -> sleep 2 -> Start-Process cloudflared.exe tunnel --url http://localhost:8765

### Step 4: 验证 (类比 Skillopt evaluate_model)

    Invoke-RestMethod http://localhost:8765/api/health  (期望: status=ok)
    .venv/Scripts/python.exe -c import requests,json; r=requests.post('http://localhost:8765/api/romance/new',json={chr(125)); print(r.json())  (期望: has_llm=true)
    Get-Content tunnel_url.txt | Invoke-RestMethod (期望: 同本地)
    nvidia-smi 或 GPU 内存检查 (期望: VRAM less than 8GB)

精确命令见 references/commands.md。

### Step 5: 记录学习 (自进化核心)

记录到 data/ops_history.jsonl:
    timestamp, trigger, diagnosis, strategy, actions, verification, time_to_recovery_s

聚合分析:
- 某类故障频繁 -> 建议预防措施
- 某策略 MTTR 最短 -> 提升为默认
- 新故障模式 -> 扩展诊断矩阵


## 常用命令速查

详细精确命令见 **references/commands.md**，以下为概要:

| 操作 | 概要 |
|------|------|
| 健康检查 | Invoke-RestMethod localhost:8765/api/health |
| 启动服务 | .venv/Scripts/python.exe -c ...from web.server import main; main() |
| 启动隧道 | cloudflared.exe tunnel --url http://localhost:8765 |
| 停止所有 | Stop-Process -Name python,cloudflared -Force |
| 生成Key | .venv/Scripts/python.exe -c ...APIKeyManager().generate_key() |
| GPU检查 | .venv/Scripts/python.exe -c import torch; print(torch.cuda.is_available()) |
| ROME编辑 | .venv/Scripts/python.exe scripts/editing/rome_edit.py --config ... |
| MEMIT编辑 | .venv/Scripts/python.exe scripts/editing/memit_edit.py --config ... |
| LoRA训练 | .venv/Scripts/python.exe scripts/train_lora.py --config ... |

## 自进化学习流程

1. 模式识别: 读取 data/ops_history.jsonl -> Counter 分析 diagnosis/strategy 频率 -> 计算 MTTR
2. 策略优化: MTTR 最长策略 -> 分析瓶颈; 成功率最低配对 -> 替换; 新模式 -> 扩展矩阵
3. 自动化建议: 定时健康检查 / 自动重启阈值 (连续3次失败) / VRAM预警 (大于80%)

## 安全边界

- 绝不直接操作 data/api_keys.json 和 data/admin_password.hash，始终通过 APIKeyManager
- 绝不将 API Key 或 admin 密码写入日志或终端
- 绝不在未经用户确认时执行 Stop-Process -Force
- 模型编辑 (ROME/MEMIT) 修改权重，操作前确认备份
- Cloudflare Tunnel URL 不加密，敏感数据在应用层加密

## 文件索引

- SKILL.md (本文件): 策略、流程、触发机制
- references/commands.md: 精确可执行命令 (免解析、直接复制)

