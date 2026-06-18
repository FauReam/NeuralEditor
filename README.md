# Heartscape Engine

本地LLM驱动的恋爱模拟/角色扮演引擎。基于 Qwen2.5-7B-Instruct，针对 RTX 4070 优化，支持**LoRA 风格微调**与**ROME/MEMIT 模型编辑**三种不同层级的权重干预。

## 特性

- **纯本地推理**：基于 transformers + bitsandbytes，无需联网
- **4070 友好**：4-bit量化，VRAM占用 < 8GB，上下文 4K-8K 甜点区
- **角色系统**：JSON定义人格、好感度、剧情状态
- **记忆系统**：短期对话窗口 + 长期向量记忆（RAG）
- **剧情引擎**：YAML场景定义，状态机驱动分支
- **三级权重干预**：
  1. **提示工程**（零权重改动）
  2. **LoRA 微调**（轻量 adapter，可开关）
  3. **ROME / MEMIT**（单条/批量神经元编辑，副作用最小）
- **Web 服务 + API Key 认证**：多用户远程接入，独立会话隔离

## Agent 快速启动指南（零试错）

> **给下一个接手此项目的 Agent（或人）：** 以下是在 2026-06-18 经过 3 小时、10+ 次进程崩溃、逐模块排查后验证过的零试错启动流程。严格照做。

### 环境假设

- Windows + PowerShell
- 虚拟环境已就绪：`.venv\Scripts\python.exe`
- 模型已下载：`models/Qwen2.5-7B-Instruct/`
- `cloudflared.exe` 已在项目根目录
- VPN 已开启（HuggingFace 需科学上网下载 embedding 模型）

### ⚠️ 已知坑点（7 个，按致命程度排序）

| # | 坑 | 症状 | 根因 | 正确做法 |
|---|-----|------|------|----------|
| 1 | **服务器反复崩溃** | `/api/romance/new` 导致连接断开，Python 进程无声退出，无 traceback | **① `python -m src.web.server` 导致模块双重初始化，CUDA 上下文冲突。② `init_engine` 后缺少 `return`，fall-through 到其他 handler 导致 GC/CUDA 冲突。③ Daemon 线程加载 LLM 导致 CUDA 崩溃。** | **见下方「正确启动方式」** |
| 2 | Daemon 线程 LLM 崩溃 | 权重加载 100% 后进程退出 | CUDA 操作在 daemon 线程不可靠 | 改为同步加载（已修复在 server.py 中） |
| 3 | Embedding 模型下载卡死 | `SentenceTransformer` 首次加载连不上 HuggingFace | 国内网络不通 | 需 VPN；MemorySystem 已加固：embedding 失败时优雅降级 |
| 4 | API Key 生成到错误路径 | Key 在 `C:\Users\46326\data\` 而服务器读 `NeuralEditor\data\` | Python 脚本 cwd 不是项目根目录 | 运行任何脚本时 cwd 必须为 `NeuralEditor\` |
| 5 | PowerShell 不支持 `&&` | 语法错误 | PowerShell 用 `;` | 分次执行或用 `;` |
| 6 | 用系统 Python | 缺依赖 | 系统 Python 未装项目包 | 始终用 `.venv\Scripts\python.exe` |
| 7 | Admin 密码/Key 需 Web 交互 | Agent 无法自动化 | Admin 密码首次访问时生成 | 用 Python 脚本直调 `APIKeyManager` |

### ✅ 验证过的正确启动方式（唯一可行）

**步骤 1：杀旧进程**

```powershell
Stop-Process -Name python,cloudflared -Force -ErrorAction SilentlyContinue
```

**步骤 2：启动 Web 服务器**

```powershell
# 必须用 -c 方式，禁止 python -m 或 python xxx.py
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from web.server import main; main()"
```

LLM 模型同步加载约 10 秒，VRAM 占用 ~5.2GB/12GB。

**步骤 3：健康检查**

```powershell
Invoke-RestMethod -Uri "http://localhost:8765/api/health"
# → {"status":"ok","version":"0.2.0"}
```

**步骤 4：启动 Cloudflare 隧道**

```powershell
.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate
# 等待输出 https://xxx.trycloudflare.com
```

**步骤 5：设置 Admin 密码（一行 Python）**

```powershell
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'.');from src.web.server import APIKeyManager;m=APIKeyManager();m.set_admin_password('admin123')"
```

**步骤 6：签发 API Key**

```powershell
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'.');from src.web.server import APIKeyManager;m=APIKeyManager();k=m.generate_key('client',expires_days=30);print(k['api_key'])"
```

**步骤 7：验证公网 + 保存**

```powershell
$url="https://xxx.trycloudflare.com"
Invoke-RestMethod -Uri "$url/api/health"
Set-Content tunnel_url.txt $url -Encoding UTF8
```

### 一键启动（完整脚本）

```powershell
# 保存为 start.ps1，powershell -File start.ps1 执行
$p = Split-Path $MyInvocation.MyCommand.Path
Set-Location $p
Stop-Process -Name python,cloudflared -Force -ErrorAction SilentlyContinue

# 启动服务器
$s = Start-Process -NoNewWindow -PassThru .venv\Scripts\python.exe -Arg "-c","import sys;sys.path.insert(0,'src');from web.server import main;main()"

# 等待就绪
do { Start-Sleep 2 } while (!(try {Invoke-RestMethod http://localhost:8765/api/health -Timeout 2} catch {}))

# 启动隧道
$t = Start-Process -NoNewWindow -PassThru .\cloudflared.exe -Arg "tunnel","--url","http://localhost:8765","--no-autoupdate"

# 提取 URL
Start-Sleep 8
$url = (.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate 2>&1 | Select-String "trycloudflare.com").Matches.Value
Write-Host "URL: $url"
```

### 客户端直连

```
https://<tunnel-url>/client?api_key=ne_xxx
```

### Agent 常用端点

| 端点 | 方法 | 用途 | Key |
|------|------|------|:---:|
| `/api/health` | GET | 确认存活 | ❌ |
| `/api/auth/verify` | POST | 验证 Key | ❌ |
| `/api/romance/new` | POST | 初始化游戏 | ❌ |
| `/api/romance/chat` | POST | 对话 | ❌ |
| `/api/admin/login` | POST | Admin 登录 | ❌ |
| `/api/admin/keys/generate` | POST | 签发 Key | 🔐 |

---

## 🔬 排障实录：2026-06-18 服务器反复崩溃事件

### 事件时间线

| 时间 | 尝试 | 结果 |
|------|------|------|
| T+0 | `python -m src.web.server` | 模型加载 → `/api/romance/new` → 进程无声崩溃 |
| T+30m | `python run_server.py` | 同上 |
| T+40m | daemon 线程 → 同步 LLM 加载 | 权重 100% 后 crash |
| T+50m | 加 try/except 兜底 | try/except 未捕获到任何 Python 异常 — 崩溃发生在 C/CUDA 层级 |
| T+55m | 改用 HTTPServer（单线程） | 仍然崩溃 |
| T+60m | bypass HTTP — 直接测试核心逻辑 | **全部通过**：LLM 加载 ✅ chat ✅ init_engine ✅ |
| T+70m | 最简 handler（`return {"test":"ok"}`） | **通过** — HTTP 基础架构正常 |
| T+80m | 逐步加回 `_get_romance_state` | 通过 |
| T+90m | 加回 `init_engine` + `to_dict()` | 通过 ✓ |
| T+100m | 移除 `with open(...)` 调试包装 | 又崩溃！ |
| T+110m | 发现 `new` handler 缺少 `return` | 加上 `return` 后仍崩溃 |
| T+120m | **回归 `-c` 启动 + try/except 包装** | **稳定运行** |

### 根因分析

1. **模块双重初始化（主因）**：`python -m src.web.server` 将 `server.py` 作为 `__main__` 运行。当 `_load_shared_llm()` 中 `from src.models.llm_engine import LLMEngine` 执行时，LLMEngine 内部可能间接触发 `import src.web.server`，导致第二个模块实例被创建。两个实例的全局变量（`_shared_llm`、`api_key_manager`、`_shared_llm_lock` 等）指向不同对象，CUDA 上下文冲突导致 C 层 crash。

2. **Daemon 线程 CUDA 不安全（次因）**：原始代码用 `threading.Thread(daemon=True)` 加载 LLM。在 Windows 上，daemon 线程中执行 `torch.cuda` 操作不稳定，权重加载完后的 `model.eval()` 或 VRAM 查询可能触发 crash。

3. **LLM 推理异常无保护（第三个因）**：`state.llm.chat()` 无 try/except，CUDA OOM 会直接杀进程。已在 chat/choice 两处加兜底。

4. **Embedding 模型首次加载卡死**：`SentenceTransformer('BAAI/bge-small-zh-v1.5')` 需从 HuggingFace 下载 ~100MB，国内不通。已加固 MemorySystem：load 失败时 `_embedder = None`，后续操作跳过。

### 教训与提速建议

- **不要信任 `python -m` 或任何将程序作为 `__main__` 运行的方式** — 永远用 `python -c "from x import main;main()"`。
- **先 bypass HTTP 层测试核心逻辑** — 用裸 Python 脚本验证 LLM 加载、推理、init_engine，确认核心无问题后再追 HTTP。
- **用最简 handler 验证 HTTP 基础** — `return {"test": "ok"}` 可以秒确认 HTTP 层是否正常。
- **逐步加回代码** — 每次只加一行，找到精确崩溃点。
- **CUDA bug 无 traceback 是典型特征** — 遇到无声崩溃，优先怀疑 CUDA 上下文/线程问题。

## 项目结构

```
heartscape-engine/
├── config/
│   ├── characters/         # 角色定义（YAML）
│   ├── prompts/            # 系统提示词模板
│   ├── scenes/             # 剧情分支定义
│   └── training/           # LoRA / ROME / MEMIT 配置文件
├── data/
│   ├── saves/              # JSON存档
│   ├── memories/           # Chroma向量库
│   ├── sessions/           # 用户会话日志
│   ├── feedback/           # 用户评分反馈
│   └── romance_chat_sample.jsonl  # LoRA训练样本
├── scripts/
│   ├── download_model.py   # 下载 Qwen GGUF
│   ├── train_lora.py       # LoRA训练（配置驱动）
│   ├── web_ui.py           # Web UI 启动器
│   ├── demo_no_llm.py      # 无模型剧情测试
│   └── editing/
│       ├── rome_edit.py    # 单条事实 ROME 编辑
│       ├── memit_edit.py   # 批量 MEMIT 编辑
│       └── evaluate_edit.py # 编辑前后对比评估
├── src/
│   ├── models/             # LLM推理、模型编辑
│   ├── core/               # 角色、剧情、记忆、状态机
│   ├── training/           # LoRATrainer
│   ├── web/                # Web 服务 + 前端页面
│   └── utils/              # 配置加载、JSON存储、场景解析
└── tests/                  # 单元测试
```

## 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3060 12GB | RTX 4070 |
| RAM | 16GB | 32GB |
| 磁盘 | 10GB | 50GB（含模型与训练缓存） |

> **上下文上限**：4070 12GB 在 4-bit 量化下，稳定运行上限约 **8K-16K**。32K 为理论架构上限，本地运行会 OOM。

---

## Web 服务（API Key 多用户模式）

服务器发出 API Key，远程客户端凭 Key 直连，每个 Key 拥有独立游戏会话。

### 启动服务器

> ⚠️ 此方式会导致进程崩溃！请使用「Agent 快速启动指南」中的正确方式。

```bash
# 禁止使用以下命令 —— 会触发模块双重初始化 CUDA 崩溃
# python -m src.web.server

# 正确启动方式见上方 Agent 快速启动指南
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from web.server import main; main()"
# 端口: 8765
```

### 页面一览

| 页面 | 地址 | 说明 |
|------|------|------|
| 微调端 | `http://localhost:8765/` | LoRA 训练 + ROME/MEMIT 编辑 |
| 恋爱游戏 | `http://localhost:8765/romance` | 内嵌式聊天（支持 API Key） |
| **远程客户端** | `http://localhost:8765/client` | 独立聊天客户端，带 Key 直连 |
| **管理后台** | `http://localhost:8765/admin` | 签发/撤销 Key，会话监控 |

### 使用流程

```
1. 管理员 → 登录 /admin → 生成 API Key（可设过期天数、请求上限）
2. 管理员 → 把 Key 发给用户
3. 用户   → 打开 /client?api_key=ne_xxx...  → 自动验证 → 直接聊天
4. 管理员 → /admin 随时撤销 Key、查看活跃会话
```

### 客户端连接方式

```
# 一键链接（推荐）
http://服务器地址:8765/client?api_key=ne_xxx...

# Header 方式
curl -H "X-API-Key: ne_xxx..." http://server:8765/api/romance/state

# Bearer Token
curl -H "Authorization: Bearer ne_xxx..." http://server:8765/api/romance/state
```

### API 端点

| 端点 | 方法 | 说明 | 需要 Key |
|------|------|------|:---:|
| `/api/health` | GET | 服务器健康检查 | ❌ |
| `/api/auth/verify` | POST | 验证 API Key 有效性 | ❌ |
| `/api/romance/new` | POST | 初始化游戏 | ❌ |
| `/api/romance/chat` | POST | 发送对话 | ❌ |
| `/api/romance/choice` | POST | 选择选项 | ❌ |
| `/api/romance/save` | POST | 存档 | ❌ |
| `/api/romance/load` | POST | 读档 | ❌ |
| `/api/romance/state` | GET | 获取当前状态 | ❌ |
| `/api/admin/keys` | GET | 列出所有 Key | 🔐 |
| `/api/admin/keys/generate` | POST | 签发新 Key | 🔐 |
| `/api/admin/keys/revoke` | POST | 撤销 Key | 🔐 |
| `/api/admin/sessions` | GET | 活跃会话列表 | 🔐 |

### 公网访问（P2P 穿透）

开发机在校园/企业内网（NAT 后），公网 IP 不可直连，需通过隧道工具穿透。

#### 方案对比

| 方案 | 国内可用 | 固定域名 | 推荐 |
|------|:---:|:---:|:---:|
| **Cloudflare Tunnel** | ✅ | ❌ 每次随机 | ⭐ 推荐 |
| localhost.run (SSH) | ❌ HTTP 空响应 | ❌ | - |
| serveo.net (SSH) | ❌ 502 | ❌ | - |
| ngrok | ❌ DNS 劫持 | ❌ | 需 VPN |

#### Cloudflare Tunnel（推荐）

无需注册、无需 VPN，国内校园网实测可用：

```bash
# 下载 cloudflared（一次性）
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "cloudflared.exe"

# 启动隧道
.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate
```

输出示例：`https://xxx.trycloudflare.com`

#### 一键启动脚本

```powershell
powershell -ExecutionPolicy Bypass -File start_tunnel.ps1
```

自动完成：启动 Web 服务 → 建立 Cloudflare 隧道 → 解析公网 URL → 验证连通性。

#### 客户端使用

客户无需安装任何软件，浏览器打开即可：

```
https://<tunnel-url>/client?api_key=ne_xxx
```

流程：管理员 `/admin` 生成 Key → 发 Key 给客户 → 客户打开链接直接聊天。

> **注意**：免费隧道域名每次重启会变化。需固定域名可注册 Cloudflare 账户绑定自定义域。

---

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS / Linux
```

### 2. PyTorch 安装（⚠️ 关键）

> **必须安装 CUDA 版 PyTorch，否则模型在 CPU 上加载需数分钟且推理极慢。**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: True NVIDIA GeForce RTX 4070
```

**常见坑**：`pip install torch` 默认安装 CPU 版（`torch 2.x.x+cpu`），务必指定 `--index-url`。

### 3. 安装项目依赖

```bash
pip install -e ".[dev,train]"
```

### 4. 模型准备

```bash
# 自动下载（约 15GB）
python scripts/download_model.py

# 或手动放入 models/Qwen2.5-7B-Instruct/
```

### 5. 启动服务

```bash
# ⚠️ 见上方「Agent 快速启动指南」— 禁止用 python -m
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');from web.server import main;main()"
python -m src.main          # 命令行交互
python -m src.main --demo   # 无模型演示模式
```

### 6. 一键公网穿透

```powershell
powershell -ExecutionPolicy Bypass -File start_tunnel.ps1
```

详见上文「公网访问」章节。

### Makefile 快捷命令

```bash
make install        # pip install -e ".[dev,train]"
make run            # python -m src.main
make web            # ⚠️ 此命令会崩溃！请使用 Agent 快速启动指南中的方式
make test           # pytest tests/ -v
make download-model # 下载 Qwen
make train-lora     # 使用默认配置训练 LoRA
make lint           # ruff + mypy
make format         # black + ruff --fix
```

### 启动后检查

```bash
curl http://localhost:8765/api/health
# → {"status": "ok", "version": "0.2.0", ...}

curl -X POST http://localhost:8765/api/romance/new -H "Content-Type: application/json" -d "{}"
# → "has_llm": true 表示模型就绪
```

## 三级权重干预指南

### Level 1：提示工程（不动权重）

修改 `config/characters/default.yaml` 中的 `system_prompt`，或调整 `config/settings.yaml` 中的 `temperature` / `top_p`。

```yaml
# config/characters/default.yaml
system_prompt: |
  你是{char_name}，性格{personality}。
  说话风格：{speaking_style}
  当前关系：{relationship}
  好感度：{affection}/100
```

### Level 2：LoRA 微调（轻量 Adapter）

适合整体风格调整（如浪漫对话语气、角色一致性）。训练后生成 `adapter_model.safetensors`，可随时开关。

#### 配置文件

```yaml
# config/training/lora.yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
dataset_path: "data/romance_chat_sample.jsonl"

lora:
  r: 8                  # LoRA rank（保守 4-8，激进 16-32）
  alpha: 16             # 缩放因子（通常 2*r）
  dropout: 0.0          # 正则化（0 = 无）
  target_modules: ["q_proj", "v_proj"]  # 仅注意力矩阵

training:
  num_train_epochs: 1
  learning_rate: 1.0e-5 # 保守值，避免破坏安全对齐
  gradient_accumulation_steps: 4
  optim: "paged_adamw_8bit"
```

#### 训练命令

```bash
# 使用配置文件
python scripts/train_lora.py --config config/training/lora.yaml

# 命令行覆盖参数
python scripts/train_lora.py --config config/training/lora.yaml \
    --r 16 --lr 2e-5 --epochs 2 --output lora_v2

# 4070 上约 15-30 分钟
```

#### 加载 LoRA

```yaml
# config/settings.yaml
llm:
  lora_path: "lora_romance"   # 训练输出目录
```

或在运行时动态加载/卸载：

```python
from src.models.llm_engine import LLMEngine

llm = LLMEngine(model_path="...", lora_path="lora_romance")
llm.unload_lora()  # 恢复基础模型
```

### Level 3：ROME / MEMIT（外科神经元编辑）

适合**精确修正特定概念**（如"约会"被过度拒绝为敏感内容），不改整体行为分布。

核心原理：定位存储该概念的 MLP 层神经元，计算协方差矩阵后执行**秩一更新**。

#### ROME（单条事实编辑）

```yaml
# config/training/rome.yaml
edit:
  subject: "牵手"                    # 要编辑的概念
  relation: "在恋爱中是一种"          # 上下文
  target: "常见的亲密表达方式"        # 新关联
  layer_idx: 15                     # 编辑层（10-25 范围探索）
  lam: 5.0                          # 正则化（↑保守 ↓激进）

calibration_prompts:                # 校准分布用的通用句子
  - "今天天气不错，适合出去散步。"
  - "我喜欢在图书馆看书。"

restore_on_exit: true               # 脚本结束后自动恢复权重
```

```bash
python scripts/editing/rome_edit.py --config config/training/rome.yaml
```

#### MEMIT（批量事实编辑）

同时编辑多条相关概念，比顺序 ROME 更稳定。

```yaml
# config/training/memit.yaml
edits:
  - subject: "牵手"
    target: "常见的亲密表达方式"
    layer_idx: 15
    lam: 5.0
  - subject: "拥抱"
    target: "温暖的安慰方式"
    layer_idx: 15
    lam: 5.0
  - subject: "约会"
    target: "了解彼此的正常社交活动"
    layer_idx: 15
    lam: 5.0
```

```bash
python scripts/editing/memit_edit.py --config config/training/memit.yaml
```

#### 评估编辑效果

```bash
python scripts/editing/evaluate_edit.py \
    --config config/training/rome.yaml \
    --test-prompts data/test_prompts.txt \
    --output data/edits/eval_results.json
```

输出包含**编辑前后对比**，自动统计有多少 prompt 的响应发生了变化。

---

## 调参工作流

**第一步：诊断问题范围**

```bash
# 用 evaluate 脚本测试模型对目标概念的响应
python scripts/editing/evaluate_edit.py \
    --config config/training/rome.yaml
```

**第二步：选择干预层级**

| 问题特征 | 推荐方案 | 可调参数 |
|---------|---------|---------|
| 整体风格不自然（语气、用词） | **LoRA** | `r`, `lr`, `epochs`, `target_modules` |
| 少数特定词被过度拒绝 | **ROME** | `layer_idx`, `lam` |
| 多个相关概念都被误杀 | **MEMIT** | `layer_idx`, `lam`, 批量编辑列表 |
| 轻微 OOC（角色脱离） | **提示工程** | `system_prompt`, `temperature` |

**第三步：调参方向**

```yaml
# LoRA 参数影响
r: 4        # 改动范围极小，适合微调语气
r: 8        # 平衡（推荐起点）
r: 16-32   # 改动范围大，需更多数据防止过拟合

lr: 5e-6    # 极保守，几乎不改变原有能力
lr: 1e-5    # 标准值
lr: 2e-5    # 激进，训练更快但可能破坏对齐

# ROME 参数影响
layer_idx: 10   # 偏语义/词汇层，改动较浅
layer_idx: 15   # 中层，通常效果最佳（推荐）
layer_idx: 20   # 偏高层/抽象，改动深远但副作用大

lam: 10.0  # 几乎无效果，非常安全
lam: 5.0   # 保守，副作用小（推荐起点）
lam: 2.0   # 效果强，可能扩散到相关概念
lam: 0.5   # 极强，风险高
```

---

## 内容边界

本引擎用于成人向恋爱模拟/视觉小说。禁止用于：
- 生成暴力、危害他人人生安全或其他有害内容


## 运维指南

### 服务重启流程

```powershell
# 停旧服务
Stop-Process -Name python,cloudflared -Force -ErrorAction SilentlyContinue

# 启动服务器（后台加载 LLM，GPU ~8 秒，CPU 数分钟）
Start-Process -NoNewWindow .\.venv\Scripts\python.exe -ArgumentList "-m","src.web.server"

# 启动隧道
.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate
```

### 命令行生成 API Key

```bash
python gen_key.py
# 输出: ne_xxxxxxxx...
```

### 配置文件速查

| 文件 | 用途 |
|------|------|
| `config/settings.yaml` | LLM 参数、记忆系统、好感度配置 |
| `config/characters/default.yaml` | 默认角色（小棠） |
| `config/scenes/chapter1.yaml` | 第一章剧情分支 |
| `data/api_keys.json` | API Key 数据库 |
| `data/admin_password.hash` | Admin 密码（SHA256） |

---

## 常见问题

### Q: 客户端一直"连接中"？

模型后台加载中。检查 `has_llm` 状态：

```bash
curl -X POST http://localhost:8765/api/romance/new -H "Content-Type: application/json" -d "{}"
```

`"has_llm": false` → 仍在加载；`"has_llm": true` → 已就绪。

### Q: 聊天只有「*微微一笑* 嗯，我在听呢...」没有 AI 回复？

LLM 未加载。检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `False` 说明 PyTorch 是 CPU 版，需重装 CUDA 版。

### Q: 模型加载需数分钟？

CPU 版 PyTorch。检查 `torch.__version__` 是否含 `+cpu`：

```bash
pip list | findstr torch
# torch  2.x.x+cpu  ← 重装
```

重装：`pip uninstall torch torchvision torchaudio -y && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

### Q: `total_mem` AttributeError？

PyTorch 2.6+ 属性名为 `total_memory`，已修复。

### Q: ngrok 报错 `connectex: No connection could be made`？

中国大陆 DNS 劫持 `connect.ngrok-agent.com` → `127.0.0.1`，无法使用。用 Cloudflare Tunnel。

### Q: Cloudflare Tunnel 502？

检查 `curl http://localhost:8765/api/health`。正常则重启 cloudflared。

### Q: 校园网 IPv6 导致 tunnel 失败？

cloudflared 可能优先 IPv6 但被阻断。重启通常恢复，或用 `--protocol http2`。


## License

MIT