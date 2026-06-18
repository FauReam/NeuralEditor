# Heartscape Engine — 一键公网隧道启动脚本
# 用法: powershell -ExecutionPolicy Bypass -File start_tunnel.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  Heartscape Engine — 公网隧道启动" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

# ─── 1. 启动 Web 服务器 ───
Write-Host "`n[1/3] 启动 Web 服务器 (端口 8765)..." -ForegroundColor Cyan
$ServerJob = Start-Job -ScriptBlock {
    Set-Location $using:ProjectDir
    & .\.venv\Scripts\python.exe -m src.web.server 2>&1 | Out-File "$using:ProjectDir\server.log"
}
Start-Sleep -Seconds 4

# 等待服务器就绪
$MaxWait = 30
for ($i = 0; $i -lt $MaxWait; $i++) {
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:8765/api/health" -TimeoutSec 2
        Write-Host "  [OK] 服务器就绪 (v$($r.version))" -ForegroundColor Green
        break
    } catch {
        if ($i -eq $MaxWait - 1) {
            Write-Host "  [FAIL] 服务器启动超时" -ForegroundColor Red
            Stop-Job $ServerJob
            exit 1
        }
        Start-Sleep -Seconds 1
    }
}

# ─── 2. 启动 Cloudflare Tunnel ───
Write-Host "`n[2/3] 启动 Cloudflare Tunnel..." -ForegroundColor Cyan
$TunnelJob = Start-Job -ScriptBlock {
    Set-Location $using:ProjectDir
    & .\cloudflared.exe tunnel --url http://localhost:8765 --no-autoupdate 2>&1 | Out-File "$using:ProjectDir\tunnel.log"
}

# 等待隧道 URL 出现
$TunnelUrl = $null
$MaxWait = 60
for ($i = 0; $i -lt $MaxWait; $i++) {
    Start-Sleep -Seconds 2
    if (Test-Path "$ProjectDir\tunnel.log") {
        $log = Get-Content "$ProjectDir\tunnel.log" -Raw
        if ($log -match 'https://[a-zA-Z0-9-]+\.trycloudflare\.com') {
            $TunnelUrl = $Matches[0]
            break
        }
    }
}

if (-not $TunnelUrl) {
    Write-Host "  [FAIL] 隧道启动超时，检查 tunnel.log" -ForegroundColor Red
    Stop-Job $TunnelJob
    Stop-Job $ServerJob
    exit 1
}

Write-Host "  [OK] 隧道已建立" -ForegroundColor Green

# ─── 3. 验证隧道 ───
Write-Host "`n[3/3] 验证隧道连通性..." -ForegroundColor Cyan
try {
    $r = Invoke-RestMethod -Uri "$TunnelUrl/api/health" -TimeoutSec 10
    Write-Host "  [OK] 公网可达 (v$($r.version))" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] 验证失败，但隧道可能仍在预热中" -ForegroundColor Yellow
}

# ─── 输出 ───
Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  公网访问地址" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  基础 URL:    $TunnelUrl" -ForegroundColor Yellow
Write-Host ""
Write-Host "  页面一览:" -ForegroundColor White
Write-Host "    客户端:    $TunnelUrl/client?api_key=ne_xxx" -ForegroundColor Cyan
Write-Host "    恋爱游戏:  $TunnelUrl/romance" -ForegroundColor Cyan
Write-Host "    微调端:    $TunnelUrl/" -ForegroundColor Cyan
Write-Host "    管理后台:  $TunnelUrl/admin" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [提示] 客户端使用方式:" -ForegroundColor Gray
Write-Host "    1. 管理员打开 $TunnelUrl/admin 登录并生成 API Key" -ForegroundColor Gray
Write-Host "    2. 把 Key 发给客户" -ForegroundColor Gray
Write-Host "    3. 客户打开 $TunnelUrl/client?api_key=ne_xxx 即可使用" -ForegroundColor Gray
Write-Host ""
Write-Host "  按 Ctrl+C 停止所有服务" -ForegroundColor DarkYellow
Write-Host "========================================" -ForegroundColor Magenta

# 保存 URL 供外部读取
$TunnelUrl | Out-File "$ProjectDir\tunnel_url.txt" -Encoding UTF8

# ─── 保持运行 ───
try {
    while ($true) {
        Start-Sleep -Seconds 10
        if ($ServerJob.State -ne "Running") {
            Write-Host "[WARN] 服务器进程已退出" -ForegroundColor Red
            break
        }
        if ($TunnelJob.State -ne "Running") {
            Write-Host "[WARN] 隧道进程已退出" -ForegroundColor Red
            break
        }
    }
} finally {
    Write-Host "`n正在停止服务..." -ForegroundColor Yellow
    Stop-Job $ServerJob -ErrorAction SilentlyContinue
    Stop-Job $TunnelJob -ErrorAction SilentlyContinue
    Remove-Job $ServerJob -ErrorAction SilentlyContinue
    Remove-Job $TunnelJob -ErrorAction SilentlyContinue
    Write-Host "已停止。" -ForegroundColor Gray
}
