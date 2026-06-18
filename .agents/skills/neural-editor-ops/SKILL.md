---
name: neural-editor-ops
description: NeuralEditor 运维 — 快速启动 + 排障。加载后第一步 health check localhost:8765。
metadata:
  project-root: C:/Users/46326/NeuralEditor
  python: .venv/Scripts/python.exe
  port: 8765
  model: models/Qwen2.5-7B-Instruct
  gpu: RTX 4070 12GB
  vram: ~5.2GB
  last-updated: 2026-06-19
  version: 6
---

# NeuralEditor 快速启动 Skill v6

> 纯操作手册。不讲原理，只给命令。

## 加载后第一步

```
Invoke-RestMethod -Uri http://localhost:8765/api/health -TimeoutSec 5
```
- `status: ok` → 告诉用户"服务器正常运行"
- 失败 → 执行下方完整启动流程

---

## 完整启动流程（已验证，照做即可）

```powershell
cd C:\Users\46326\NeuralEditor

# 1. 杀旧进程
Stop-Process -Name python,cloudflared -Force -ErrorAction SilentlyContinue

# 2. 启动服务器（后台）
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');from web.server import main;main()"
# 等待 20 秒

# 3. 验证服务器
Start-Sleep 20
Invoke-RestMethod -Uri http://localhost:8765/api/health -TimeoutSec 5
# 应返回 {"status":"ok","version":"0.2.0"}

# 4. 检查 8765 端口（必须先确认端口在监听，否则隧道白开）
netstat -ano | Select-String "8765.*LISTENING"
# 没输出 = Python 服务器没起来，回到步骤2

# 5. 启动隧道（后台）
.\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate
# 等待 10 秒，从日志提取 https://xxx.trycloudflare.com

# 6. 设置 Admin 密码 + 签发 Key（注意 cwd！）
.venv\Scripts\python.exe -c "import sys,os;os.chdir('C:/Users/46326/NeuralEditor');sys.path.insert(0,'.');from src.web.server import APIKeyManager;m=APIKeyManager();m.set_admin_password('admin123');k=m.generate_key('admin',expires_days=365);print(k['api_key'])"

# 7. 验证公网
Invoke-RestMethod -Uri https://<tunnel-url>/api/health -TimeoutSec 15
# 530 错误时等待 10 秒重试
```

## 启动后必须做的事

1. **更新 `访问链接.txt`**，格式：
```
客户端直连: https://<tunnel>/client?api_key=<key>
管理后台: https://<tunnel>/admin
Admin密码: admin123
角色扮演: https://<tunnel>/romance
微调端: https://<tunnel>/
API Key: <key>
```
2. 告诉用户：Admin 密码 `admin123`，API Key 为签发结果

---

## 陷阱速查（启动时最常见）

| # | 症状 | 原因 | 解决 |
|---|------|------|------|
| 1 | `NameError: name 'io' is not defined` | server.py 缺 import | 确认文件有 `import io` 和 `import signal` |
| 2 | `NameError: name 'signal' is not defined` | 同上 | 同上 |
| 3 | API Key 验证失败 | 生成时 cwd 不是 NeuralEditor | 必须 `os.chdir('C:/Users/46326/NeuralEditor')` |
| 4 | 公网 530 | cloudflared 未就绪 | 等 10-15 秒重试 |
| 5 | Unknown Error（前端） | romance.html api() 无 catch | 已修复：api() 改为 async/await + try/catch |
| 6 | 服务器 ~10 分钟无声崩溃 | idle 崩溃（未根除） | 用户报 500 时重启服务器 |
| 7 | 公网 502/连接被拒绝 | cloudflared 在跑但 Python 没监听 8765 | 启动隧道前先 `netstat -ano \| Select-String "8765.*LISTENING"` |

---

## 改动过的文件（不要重复改）

| 文件 | 已做的改动 |
|------|-----------|
| `src/web/server.py` | +`import io`, +`import signal`, +`load_yaml`, +`/api/romance/characters`, `/new` 自动读 model_path, +`CUDA_LAUNCH_BLOCKING=1` |
| `src/models/llm_engine.py` | +`import threading`, +`_inference_lock`, `chat()` 推理锁 |
| `src/web/romance.html` | `api()` 改为 async/await + try/catch |
| `config/characters/default.yaml` | +`model_path: models/Qwen2.5-7B-Instruct` |
| `.gitignore` | +`cloudflared.exe`, +`__sentinel__.log`, +`src/data/api_keys.json` |

> ⚠️ 以上改动已生效且推送 GitHub。无需再次修改。

---

## 禁止做的事

- ❌ `python -m src.web.server` — 会导致模块双初始化 + CUDA 崩溃
- ❌ `python src/web/server.py` — 同上
- ❌ 用 `Invoke-WebRequest` 测试 HTML 页面 — 会超时，用 `Invoke-RestMethod`
- ❌ 多行 PowerShell 命令 — 被拦截，拆成单行或用 `;`
- ❌ 在非 NeuralEditor cwd 下生成 API Key — 会存到错路径

---

## 2026-06-18 任务复盘

### 今日完成

| # | 任务 | 结果 |
|---|------|------|
| 1 | 启动本地 LLM 服务器 (Qwen2.5-7B, RTX 4070) | ✅ 5.2GB VRAM |
| 2 | 修复 server.py 缺 import (io, signal) | ✅ |
| 3 | 修复多线程 CUDA 崩溃 (推理锁 + CUDA_LAUNCH_BLOCKING) | ✅ 4-7min→~10min |
| 4 | 部署 Cloudflare 公网隧道 | ✅ trycloudflare |
| 5 | 修复 romance.html Unknown Error (api() try/catch) | ✅ |
| 6 | 角色-模型绑定 (YAML model_path) | ✅ |
| 7 | 新增角色列表 API (/api/romance/characters) | ✅ |
| 8 | 建立运维 skill v1→v6 进化体系 | ✅ |
| 9 | 建立访问链接.txt 自动同步规则 | ✅ |
| 10 | 推送完整代码到 GitHub (FauReam/NeuralEditor) | ✅ |
| 11 | 验证 skill v6 六步启动流程一次性通过 | ✅ |

### 验证有效的启动流程

```
杀进程 → python -c 启动 → 等20s health → 检查8765端口 → cloudflared 隧道 → 签发Key → 公网验证 → 更新访问链接.txt
```
七步总耗时约 60 秒。⚠️ 步骤4检查端口必须在步骤5启动隧道之前，否则隧道白开。

### 下次启动只需照做 skill 中的六步命令，不改任何代码。

### 本次验证 (2026-06-18 最后启动)

- 用户反馈：**"这次访问倒是没问题了"** ✅
- 六步全过，首次公网 530 → 10 秒重试即通
- 本地 LLM 正常回复角色对话
- 结论：skill v6 流程成熟可用，下次直接照做
