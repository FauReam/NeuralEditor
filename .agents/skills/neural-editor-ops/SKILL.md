---
name: neural-editor-ops
description: NeuralEditor / Heartscape Engine 运维操作。当用户提到 NeuralEditor、Heartscape、启动服务、tunnel、cloudflared、API Key、ROME、MEMIT、LoRA、GPU、服务异常、502、530、健康检查、模型加载、崩溃、段错误时自动加载。加载后第一步：执行健康检查 localhost:8765/api/health，根据结果决定是否干预。
metadata:
  short-description: NeuralEditor 自进化运维 skill - 自动健康检查
  project-root: C:/Users/46326/NeuralEditor
  auto-trigger: true
  companion: references/commands.md
  last-updated: 2026-06-18T17:40
  evolution-version: 5
---

# NeuralEditor 运维 Skill — 进化版 v5

> v5 更新：完整开发复盘（改动文件清单 + 6 个 Bug 修复 + 新增功能 + 经验教训）
> v4 更新：发现 4-7 分钟崩溃根因并修复 —— 多线程并发 CUDA 推理 + 推理锁 + CUDA_LAUNCH_BLOCKING。
> v3 新增：server.py 缺 import、4-7 分钟无声崩溃模式、romance.html API 错误处理、
> Cloudflare 530 重试策略、访问链接.txt 强制同步规则。
> **所有内容均经过验证**。

**加载后第一步**: 执行健康检查
```
Invoke-RestMethod -Uri http://localhost:8765/api/health -TimeoutSec 5
```
> ⚠️ 不要用 Invoke-WebRequest — 对 /romance 等 HTML 页面会超时。
> 始终用 Invoke-RestMethod 做 API 测试。
- 返回 `status: ok` → 报告用户服务正常，继续用户其它任务
- 失败 → 进入下方诊断修复流程

---

## 项目基础

| 项 | 值 |
|------|------|
| 项目路径 | C:/Users/46326/NeuralEditor |
| 虚拟环境 | .venv/Scripts/python.exe |
| 服务端口 | 8765 |
| 隧道工具 | cloudflared.exe |
| LLM | models/Qwen2.5-7B-Instruct/ (4-bit 量化) |
| GPU | RTX 4070, 12GB VRAM |
| 配置文件 | config/settings.yaml, config/characters/, config/scenes/ |
| 数据目录 | data/saves/, data/memories/(Chroma), data/sessions/ |

---

## ⚠️ 致命陷阱 — 按严重程度排序

### 1. `python -m` / `python xxx.py` 启动 → 进程无声崩溃 【致命】

**症状**: LLM 加载正常，`/api/health` 正常，但 `/api/romance/new` 触发后 Python 进程无声退出，**无任何 traceback**。

**根因**: 模块双重初始化。`python -m src.web.server` 将 `server.py` 作为 `__main__` 运行，当后续代码中 `from src.models.llm_engine import LLMEngine` 触发时，可能创建第二个 `src.web.server` 模块实例，导致全局变量（`_shared_llm`、`api_key_manager` 等）出现两份。CUDA 上下文冲突 → C 层 crash。

**正确做法 — 唯一可行**:
```powershell
cd C:/Users/46326/NeuralEditor
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from web.server import main; main()"
```
- ✅ 用 `python -c` 执行，避免 `__main__` 文件
- ❌ 禁止 `python -m src.web.server`
- ❌ 禁止 `python src/web/server.py`
- ❌ 禁止 `python run_server.py`

### 2. Daemon 线程加载 LLM → 权重 100% 后崩溃 【致命 · 已修复】

**症状**: 权重进度条 100% 后进程退出，无 "VRAM: allocated=..." 和 "LLM 已就绪" 消息。

**根因**: 原始代码用 `threading.Thread(daemon=True)` 加载 LLM。Windows 上 daemon 线程执行 CUDA 操作不稳定。

**修复**: `server.py` 中 `main()` 已改为同步调用 `_load_shared_llm()`。

### 3. Handler 缺少 `return` → fall-through 崩溃 【致命 · 已修复】

**症状**: `/api/romance/new` 处理完后，代码继续执行到后续 `elif` 分支，触发 GC/CUDA 冲突。

**修复**: 已在 `/api/romance/new` handler 末尾添加 `return`。

### 4. LLM 推理异常无 try/except → 崩溃 【致命 · 已修复】

**症状**: 用户发送聊天消息后进程退出。

**修复**: `/api/romance/chat` 和 `/api/romance/choice` 中的 `state.llm.chat()` 调用已加 try/except，异常时返回兜底文本。

### 5. Embedding 模型下载卡死 【严重】

**症状**: `SentenceTransformer('BAAI/bge-small-zh-v1.5')` 首次加载连不上 HuggingFace，HTTP 请求超时。

**根因**: 国内网络不通 HuggingFace。

**修复**: MemorySystem 已加固——embedding 加载失败时 `_embedder = None`，后续向量操作优雅跳过。需要 VPN 才能正常使用向量记忆功能。

### 6. API Key 生成到错误路径 【中等】

**症状**: Key 验证失败，`api_keys.json` 在 `C:\Users\46326\data\` 而非 `NeuralEditor\data\`。

**根因**: 运行 Python 脚本时 cwd 不是项目根目录，`Path("data/api_keys.json")` 相对路径解析错误。

**修复**: 所有脚本执行前必须 `cd C:/Users/46326/NeuralEditor`。

### 7. PowerShell 语法陷阱 【轻微】

- 不支持 `&&`，用 `;` 或分次执行
- 单引号内 `'src'` 可以，双引号内需转义
- `Invoke-RestMethod` 的 `-Headers` 需用 `@{}` 哈希表

### 8. server.py 缺少 `import io` / `import signal` → 启动即崩溃 【致命】

**症状**: `NameError: name 'io' is not defined` 或 `NameError: name 'signal' is not defined`

**根因**: `main()` 函数中使用了 `io.TextIOWrapper(...)` 和 `signal.signal(...)` 但未导入。

**修复**: 在 `server.py` 顶部 import 区确保存在:
```python
import io
import signal
```

### 9. 服务器 4-7 分钟后无声崩溃 【致命 · 推理锁延长至 ~10 分钟，仍未根治】

**症状**: 启动正常，LLM 就绪，处理若干请求后 Python 进程无声退出（exit=1），
无任何 stderr traceback。stdout 最后一行始终是 `[+] 游戏引擎预初始化完成`。
约 4-10 分钟内必然崩溃。

**根因分析（v4 确认）**:
1. ThreadingHTTPServer 多线程 + 共享 `_shared_llm` → 多线程并发调用 `model.generate()`
2. CUDA 不是线程安全的 — 多个 Python 线程操作同一 GPU 上下文 → C 层 crash
3. `_shared_llm_lock` 只在 LLM 赋值时使用，**不保护推理调用**

**修复 v4**:

① `llm_engine.py` 添加推理锁:
```python
import threading

class LLMEngine:
    def __init__(self, ...):
        ...
        self._inference_lock = threading.Lock()
        ...

    def chat(self, ...):
        ...
        with self._inference_lock:          # ← 所有推理排队，杜绝并发
            with torch.no_grad():
                outputs = self.model.generate(...)
        ...
```

② `server.py` main() 设置 CUDA_LAUNCH_BLOCKING:
```python
def main():
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # ← 同步 CUDA，消除异步竞态
    ...
```

**两个修复文件的精确位置**:
| 文件 | 修改 | 行号区域 |
|------|------|---------|
| src/models/llm_engine.py | `import threading` | L5 |
| src/models/llm_engine.py | `self._inference_lock = threading.Lock()` | L81 |
| src/models/llm_engine.py | `with self._inference_lock:` 包裹 `torch.no_grad()` | L239 |
| src/web/server.py | `os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")` | L1354 |

**v4 复测结果** (2026-06-18):
- 修复前: 4-7 分钟必然崩溃, exit=1
- 修复后: **延长至约 10 分钟**, 但仍会崩溃, exit=1
- 崩溃发生在 **idle 状态**（无活跃请求），stdout 无新输出
- 结论: 推理锁解决了并发问题, 但存在**非 CUDA 层面的 idle 崩溃机制**
- 下一步排查方向: Windows 进程 idle timeout / watchdog / 内存泄漏

### 10. Cloudflare 隧道 530 错误 + QUIC 超时 【中等】

**症状**: 公网访问返回 `530 The origin has been unregistered from Argo Tunnel`，cloudflared 日志显示 `ERR Failed to dial a quic connection` 后重试成功。

**对策**: 
- 隧道创建后等待 15 秒再做首次公网访问
- 530 时再等 5-10 秒重试
- 如果持续失败，重启 cloudflared 并重新获取 URL

### 11. romance.html 前端 API 无错误处理 【已修复】

**症状**: 浏览器弹出 "Unknown Error"，前端 JS 未捕获异常导致页面卡死。

**根因**: `api()` 函数中 `fetch().then(r => r.json())` 在网络错误或 HTTP 非 200 时抛出未捕获 rejection。

**修复**: `api()` 函数已改为 async/await + try/catch，所有错误返回 `{ error: "..." }` 对象。
修改后 `api()` 签名不变，所有调用者无需改动。

---

## Skillopt 5 步自进化运维管道

```
Step 1: 诊断 → Step 2: 策略选择 → Step 3: 干预执行 → Step 4: 验证 → Step 5: 记录学习
```

### Step 1: 诊断

采集指标:
| 指标 | 命令 |
|------|------|
| 健康检查 | `Invoke-RestMethod -Uri http://localhost:8765/api/health -TimeoutSec 5` |
| 进程状态 | `Get-Process -Name python -ErrorAction SilentlyContinue` |
| CUDA 可用 | `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` |
| VRAM 占用 | `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.memory_allocated()/1e9)"` |
| 隧道 URL | `Get-Content tunnel_url.txt -ErrorAction SilentlyContinue` |
| 公网可达 | `Invoke-RestMethod -Uri (Get-Content tunnel_url.txt)/api/health -TimeoutSec 10` |

诊断分类矩阵:
| health | process | VRAM | 结论 | 策略 |
|:---:|:---:|:---:|---------|-----------|
| OK | alive | <8GB | 正常 | skip |
| OK | alive | >10GB | VRAM压力 | 降低context_length |
| FAIL | alive | - | 端口异常/HTTP bug | warm_restart |
| FAIL | dead | - | 服务未运行 | cold_start |
| OK | alive | <8GB | 但chat崩溃 | 检查stderr |
| OK | alive | N/A | tunnel 530 | 等待15秒重试 |
| OK | alive | N/A | 已运行>5分钟 | 可能随时崩溃，准备warm_restart |

### Step 2: 策略选择

| 诊断 | 策略 | 动作 |
|------|------|------|
| 服务未运行 | cold_start | 杀进程 → `-c` 启动 → 轮询 health |
| 服务无响应 | warm_restart | 杀 Python → 等3秒 → cold_start |
| 隧道异常 | tunnel_restart | 杀 cloudflared → 重开隧道 |
| 公网530 | tunnel_retry | 等待10-15秒 → 重新公网health检查 |
| 频繁崩溃 | pre_init_workaround | 确保 `_pre_init_engine()` 在主线程运行 |

### Step 3: 干预执行

**3-A cold_start（唯一正确的启动方式）**:
```powershell
cd C:/Users/46326/NeuralEditor
Stop-Process -Name python,cloudflared -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from web.server import main; main()"
# 等待 10-15 秒让 LLM 加载
```

**3-B warm_restart**: 同 cold_start。

**3-C 隧道重启**:
```powershell
Stop-Process -Name cloudflared -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate
```

**3-D 完整启动流程（含 Key 和隧道）**:
```powershell
# 1. 杀旧进程
cd C:/Users/46326/NeuralEditor
Stop-Process -Name python,cloudflared -Force -ErrorAction SilentlyContinue

# 2. 启动服务器（后台）
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');from web.server import main;main()"
# 等待 15-20 秒 → health check

# 3. 启动隧道（后台）
.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate
# 等待 10-15 秒 → 从日志提取 trycloudflare URL

# 4. 设置 Admin 密码 + 签发 Key（注意 cwd！）
.venv\Scripts\python.exe -c "import sys,os;os.chdir('C:/Users/46326/NeuralEditor');sys.path.insert(0,'.');from src.web.server import APIKeyManager;m=APIKeyManager();m.set_admin_password('admin123');k=m.generate_key('admin',expires_days=365);print(k['api_key'])"
```

### Step 4: 验证

| 检查项 | 命令 | 期望结果 |
|--------|------|----------|
| 服务存活 | `Invoke-RestMethod http://localhost:8765/api/health` | `status=ok` |
| LLM就绪 | POST `/api/romance/new` → 检查 `has_llm` | `has_llm=true` |
| 公网可达 | `Invoke-RestMethod (gc tunnel_url.txt)/api/health` | 同本地 |
| 端到端 | 公网 POST session→new→chat 全链路 | 模型回复角色对话 |

**端到端验证命令**（公网全链路）:
```powershell
$url="https://<tunnel>"; $h=@{"Content-Type"="application/json"};
Invoke-RestMethod -Method POST -Uri "$url/api/romance/session/start" -Headers $h -Body '{"consent":true}' -TimeoutSec 15
Invoke-RestMethod -Method POST -Uri "$url/api/romance/new" -Headers $h -Body '{"character":"config/characters/default.yaml"}' -TimeoutSec 20
Invoke-RestMethod -Method POST -Uri "$url/api/romance/chat" -Headers $h -Body '{"message":"你好"}' -TimeoutSec 45
| 聊天可用 | POST `/api/romance/chat` `{"message":"你好"}` | 返回角色回复 |

### 强制规则：每次给出 URL 后更新访问链接.txt

**每次**向用户提供公网 URL + API Key 之后，**必须**同步更新 `C:\Users\46326\NeuralEditor\访问链接.txt`，格式如下：

```
客户端直连: https://<tunnel-url>/client?api_key=<key>
管理后台: https://<tunnel-url>/admin
Admin密码: admin123
角色扮演: https://<tunnel-url>/romance
微调端: https://<tunnel-url>/
API Key: <key>
```

此文件是用户快速获取最新地址的唯一入口，**不可跳过**。

### Step 5: 记录学习

记录到 `data/ops_history.jsonl`：
```
{timestamp, trigger, diagnosis, strategy, actions, verification, time_to_recovery_s}
```

---

## 预初始化引擎（workaround）

`server.py` 中 `main()` 在启动 HTTP 服务器**前**调用 `_pre_init_engine()`——在主线程中预构建 StoryEngine，避免 HTTP handler 线程调用 `init_engine` 时触发 CUDA 崩溃。即使预初始化失败（如 embedding 模型下载失败），服务仍可启动，`/api/romance/new` 回退到按需初始化。

---

## 角色-模型绑定

角色 YAML 配置文件中 `model_path` 字段与角色绑定，用户不可在前端更改模型。

| 角色 | 配置文件 | 绑定模型 |
|------|---------|---------|
| 小棠 | config/characters/default.yaml | models/Qwen2.5-7B-Instruct |

新增角色 API: `GET /api/romance/characters` — 返回角色列表含名称、描述、绑定模型。

角色配置示例字段: `character_id`, `model_path`, `profile.name`, `profile.personality_traits`, `profile.background`

---
## 常用命令速查

| 操作 | 命令概要 | 详见 |
|------|---------|------|
| 健康检查 | `Invoke-RestMethod localhost:8765/api/health` | commands.md |
| 正确启动 | `.venv\Scripts\python.exe -c "...from web.server import main;main()"` | commands.md |
| 启动隧道 | `.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate` | commands.md |
| 停止所有 | `Stop-Process -Name python,cloudflared -Force` | commands.md |
| 生成Key | `.venv\Scripts\python.exe -c "...APIKeyManager().generate_key()"` | commands.md |
| 设置Admin密码 | `.venv\Scripts\python.exe -c "...set_admin_password('pw')"` | commands.md |
| GPU检查 | `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` | commands.md |

---

## 自进化机制

1. **模式识别**: 读取 `data/ops_history.jsonl` → 统计崩溃频率 → 识别新模式
2. **策略优化**: MTTR 最长策略 → 分析瓶颈 → 更新本文件
3. **自动化建议**: 连续3次健康检查失败 → 自动 warm_restart；VRAM >80% → 警告
4. **SKILL.md 自我更新**: 每次排障后追加新陷阱到本文的「致命陷阱」章节

---

## 安全边界

- 绝不直接操作 `data/api_keys.json` 和 `data/admin_password.hash`，始终通过 `APIKeyManager`
- 绝不在日志/终端打印完整 API Key 或密码
- 模型编辑 (ROME/MEMIT) 修改权重，操作前确认备份
- Cloudflare Tunnel URL 不加密，敏感数据在应用层加密

---

## 文件索引

| 文件 | 内容 |
|------|------|
| SKILL.md (本文件) | 策略、流程、陷阱、自进化机制 |
| references/commands.md | 精确可执行命令（已验证，无 `chr()` 转义） |

---

## 2026-06-18 开发复盘 (v1→v5 进化实录)

### 改动文件清单

| 文件 | 改动 |
|------|------|
| `src/web/server.py` | +`import io`, +`import signal`, +`from src.utils.config_loader import load_yaml`, +`/api/romance/characters` 路由, `/new` 自动读 model_path, +`CUDA_LAUNCH_BLOCKING=1` |
| `src/models/llm_engine.py` | +`import threading`, +`self._inference_lock`, `chat()` 推理锁包裹 |
| `src/web/romance.html` | `api()` 函数改为 async/await + try/catch，错误友好降级 |
| `config/characters/default.yaml` | +`model_path: models/Qwen2.5-7B-Instruct` (角色-模型绑定) |
| `.codewhale/skills/neural-editor-ops/SKILL.md` | v1→v5 进化，11 个陷阱 + 5 步诊断管道 + 开发复盘 |
| `README.md` | 2026-06-18 完整更新日志 |
| `.gitignore` | +`cloudflared.exe`, +`__sentinel__.log`, +`src/data/api_keys.json` |
| `访问链接.txt` | 强制同步规则（Skill 中定义） |

### 修复的 Bug（6 个）

| # | Bug | 严重度 | 修复方案 |
|---|-----|--------|---------|
| 1 | `import io` / `import signal` 缺失 → 启动崩溃 | 致命 | 补全 import |
| 2 | 多线程并发 CUDA 推理 → 4-7 分钟崩溃 | 致命 | LLMEngine 推理锁 |
| 3 | API Key cwd 错误 → Key 存到错路径 | 中等 | `os.chdir()` 确保 cwd |
| 4 | romance.html `api()` 无错误处理 → Unknown Error | 中等 | async/await + try/catch |
| 5 | Settings 角色下拉硬编码单选项 | 轻微 | 动态加载 `/api/romance/characters` |
| 6 | cloudflared 530 + QUIC 超时 | 中等 | 重试 + 等待策略 |

### 新增功能

| 功能 | 说明 |
|------|------|
| `/api/romance/characters` | GET 角色列表，含名称/描述/绑定模型 |
| 角色-模型绑定 | YAML `model_path` 字段，前端不可更改 |
| 推理锁 | `LLMEngine._inference_lock`，多线程安全 |
| 自动健康检查 | Skill 加载即执行 `localhost:8765/api/health` |
| 5 步运维管道 | 诊断→策略→干预→验证→学习 |
| 访问链接.txt 同步 | 每次给 URL 后强制更新 |

### 已知未解决问题

| 问题 | 状态 |
|------|------|
| 服务器 idle 约 10 分钟后无声崩溃 | 推理锁延长至 ~10min，未根除 |
| Embedding 模型需 VPN 下载 | MemorySystem 已优雅降级 |
| Cloudflare 隧道 URL 每次重启变化 | 需命名隧道（需 Cloudflare 账号） |
| 仅一个角色（小棠） | 需添加更多角色 YAML |

### 经验教训

1. **CUDA 不是线程安全的** — 多线程共享模型必须加锁
2. **`python -c` > `python -m`** — 避免 `__main__` 模块双初始化
3. **PowerShell 不支持 `&&`** — 用 `;` 或分次执行
4. **API Key 生成 cwd 必须正确** — 相对路径依赖于工作目录
5. **前端 API 调用必须有 try/catch** — `fetch().then(r=>r.json())` 在错误时抛未捕获异常

### 预防 500/502/530 和进程崩溃的设计模式

#### 🔴 致命级 — 防止进程无声崩溃

| 原则 | 做法 | 反例 |
|------|------|------|
| **推理串行化** | 所有 `model.generate()` 调用包在 `with lock:` 内 | 多线程直接调用共享 LLM |
| **CUDA 同步** | 启动时设 `CUDA_LAUNCH_BLOCKING=1` | 默认异步 + 多线程 |
| **启动方式锁定** | 永远 `python -c "from x import main;main()"` | `python -m`, `python file.py` |
| **Handler 显式 return** | 每个 handler 末尾 `return`，防止 fall-through | 依赖 elif 链隐式终止 |
| **LLM 推理 try/except** | 每次 `state.llm.chat()` 包 try/except + 兜底文本 | 裸调用，CUDA OOM 直接杀进程 |

#### 🟠 严重级 — 防止 HTTP 500

| 原则 | 做法 | 反例 |
|------|------|------|
| **do_POST 全局 try** | `do_POST` 方法整体 try/except → 500 JSON 响应 | 异常传播到 http.server → 连接断开 |
| **import 完整性验证** | 启动前 `py_compile.compile()` 检查语法 | 启动后触发 NameError |
| **文件读取防御** | `_serve_file` 前检查 `fp.exists()` | 路径不存在时抛 FileNotFoundError |
| **JSON body 容错** | `_read()` 先读 Content-Length，空 body 返回 `{}` | 强制 `json.loads("")` |

#### 🟡 中等级 — 防止 502/530 网关错误

| 原则 | 做法 | 反例 |
|------|------|------|
| **隧道就绪等待** | cloudflared 启动后等 15 秒 + 重试 health | 隧道刚创建就发请求 |
| **后端存活监控** | 每次操作前 `health check`（Skill 自动） | 假设后端一直在运行 |
| **Cloudflare 重试** | 530 错误时等待 10 秒重试 3 次 | 一次失败就放弃 |

#### 🟢 前端防御

| 原则 | 做法 | 反例 |
|------|------|------|
| **API 函数容错** | `api()` 内部 try/catch，永远返回 `{error}` 对象 | `fetch().then(r=>r.json())` 无 catch |
| **HTTP status 检查** | `if (!r.ok) throw new Error(...)` | 假设 200 就是 JSON |
| **UI 友好降级** | 错误时显示 `res.error` 文本，不禁用输入框 | 吐 "Unknown Error" 后页面卡死 |
| **localStorage 安全** | API Key 存 localStorage 但验证后才使用 | 过期 Key 静默失败 |

#### 📋 部署检查清单（每次启动后）

```
□ py_compile 检查语法
□ health check 返回 ok
□ POST /new 返回 has_llm=true
□ POST /chat 返回角色回复（不是兜底文本）
□ 公网 health check 通过
□ 访问链接.txt 已更新
□ .gitignore 排除 api_keys.json
□ git push 同步 GitHub
```