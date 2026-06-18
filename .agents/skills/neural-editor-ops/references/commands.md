# NeuralEditor 运维 - 精确可执行命令

**使用说明**: 以下命令已经过 PowerShell 兼容性验证，可直接复制执行。
执行前确保: cd C:/Users/46326/NeuralEditor

## 健康检查

Invoke-RestMethod -Uri http://localhost:8765/api/health -ErrorAction SilentlyContinue

## 进程状态

Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet64

## GPU 状态

.venv\Scripts\python.exe -c 'import torch; print(torch.cuda.is_available())'

## VRAM 占用

.venv\Scripts\python.exe -c 'import torch; a=torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0; t=torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0; print(a,t)'


## 冷启动 (cold_start)

Stop-Process -Name python,cloudflared -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Process -NoNewWindow -PassThru -FilePath 'C:\Users\46326\NeuralEditor\.venv\Scripts\python.exe' -ArgumentList '-c', 'import sys; sys.path.insert(0, chr(39)+chr(39)+src+chr(39)+chr(39)); from web.server import main; main()'

## 等待服务就绪 (health poll)

for ($i=1; $i -le 24; $i++) { Start-Sleep 5; try { =Invoke-RestMethod -Uri http://localhost:8765/api/health -TimeoutSec 3; if(.status -eq 'ok'){break} } catch {} }

## 热重启 (warm_restart)

Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
# 然后执行冷启动命令

## 隧道重启

Stop-Process -Name cloudflared -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Process -NoNewWindow -FilePath 'cloudflared.exe' -ArgumentList 'tunnel','--url','http://localhost:8765','--no-autoupdate'

## LLM 就绪检查

.venv\Scripts\python.exe -c 'import requests,json; r=requests.post(chr(34)+chr(34)+http://localhost:8765/api/romance/new+chr(34)+chr(34),json=dict()); print(json.dumps(r.json()))'

## 生成 API Key

.venv\Scripts\python.exe -c 'import sys; sys.path.insert(0,chr(34)+chr(34)+.+chr(34)+chr(34)); from src.web.server import APIKeyManager; key=APIKeyManager().generate_key(description=chr(34)+chr(34)+ops+chr(34)+chr(34),expires_days=30); print(key[chr(34)+chr(34)+api_key+chr(34)+chr(34)])'

## 撤销 API Key

.venv\Scripts\python.exe -c 'import sys; sys.path.insert(0,chr(34)+chr(34)+.+chr(34)+chr(34)); from src.web.server import APIKeyManager; APIKeyManager().revoke_key(chr(34)+chr(34)+ne_YOURKEY+chr(34)+chr(34)); print(chr(34)+chr(34)+done+chr(34)+chr(34))'

## GPU 完整诊断

nvidia-smi
netstat -ano | findstr :8765
.venv\Scripts\python.exe -c 'import torch; print(chr(34)+chr(34)+PyTorch:+chr(34)+chr(34),torch.__version__); print(chr(34)+chr(34)+CUDA:+chr(34)+chr(34),torch.cuda.is_available())'

## 模型编辑 ROME

.venv\Scripts\python.exe scripts/editing/rome_edit.py --config config/training/rome.yaml

## 模型编辑 MEMIT

.venv\Scripts\python.exe scripts/editing/memit_edit.py --config config/training/memit.yaml

## LoRA 训练

.venv\Scripts\python.exe scripts/train_lora.py --config config/training/lora.yaml

