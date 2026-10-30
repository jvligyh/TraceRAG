param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$OpenBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$serviceUrl = "http://127.0.0.1:$Port"

function Get-TraceRAGState {
    try {
        $schema = Invoke-RestMethod -Uri "$serviceUrl/openapi.json" -TimeoutSec 3
        if ($schema.info.title -ne 'TraceRAG') { return 'other' }
        $health = Invoke-RestMethod -Uri "$serviceUrl/health" -TimeoutSec 3
        if ($health.status -ne 'ok') { return 'not-ready' }
        $paths = @($schema.paths.PSObject.Properties.Name)
        $queryProperties = @($schema.components.schemas.QueryRequest.properties.PSObject.Properties.Name)
        if (($paths -contains '/documents/{document_id}/file') -and
            ($queryProperties -contains 'document_ids') -and
            ($queryProperties -contains 'history') -and
            ($health.features -contains 'ocr')) { return 'ready' }
        return 'outdated'
    } catch {
        return 'other'
    }
}

$listener = [System.Net.Sockets.TcpClient]::new()
try {
    $connection = $listener.ConnectAsync('127.0.0.1', $Port)
    $portOccupied = $connection.Wait(1500) -and $listener.Connected
} catch {
    $portOccupied = $false
} finally {
    $listener.Dispose()
}

if ($portOccupied) {
    $runningState = Get-TraceRAGState
    if ($runningState -eq 'ready') {
        Write-Host "TraceRAG 已经在运行：$serviceUrl"
        if ($OpenBrowser) { Start-Process $serviceUrl }
        exit 0
    }
    if ($runningState -eq 'outdated') {
        throw '旧版 TraceRAG 仍在运行，尚未载入图片OCR等新功能。请在原服务窗口按 Ctrl+C，然后重新双击启动文件。'
    }
    if ($runningState -eq 'not-ready') {
        throw '已有 TraceRAG 进程，但服务尚未准备好。请查看原服务窗口，稍后重新打开；不必重复启动模型。'
    }
    throw "端口 $Port 被其他程序占用。请关闭占用程序，或运行 .\start.ps1 -Port 8001 -OpenBrowser。"
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '没有找到项目虚拟环境。请按 docs/quickstart.md 创建 .venv 并安装依赖，然后重新启动。'
}
& $pythonPath -c 'import fastapi, uvicorn, multipart, tracerag'
if ($LASTEXITCODE -ne 0) {
    throw '项目依赖尚未装好。请在项目目录执行 .\.venv\Scripts\python.exe -m pip install -c constraints.txt -e ".[local-translation]"。'
}

Write-Host "TraceRAG 正在启动：$serviceUrl"
Write-Host '浏览器打开后，上传文档并选择要查询的资料。首次查询需要加载模型。'
Write-Host '使用期间保留这个窗口；退出服务请在此窗口按 Ctrl+C。'

if ($OpenBrowser) {
    # A short-lived hidden helper opens the browser after readiness. The server
    # stays in this console so shutdown and startup errors remain visible.
    $browserScript = @'
$ErrorActionPreference = 'SilentlyContinue'
$url = '__SERVICE_URL__'
for ($attempt = 0; $attempt -lt 90; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2
        $schema = Invoke-RestMethod -Uri "$url/openapi.json" -TimeoutSec 2
        if ($health.status -eq 'ok' -and $schema.info.title -eq 'TraceRAG') {
            Start-Process $url
            exit 0
        }
    } catch {}
    Start-Sleep -Seconds 1
}
'@
    $browserScript = $browserScript.Replace('__SERVICE_URL__', $serviceUrl)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($browserScript))
    $helperShell = if (Get-Command pwsh.exe -ErrorAction SilentlyContinue) { 'pwsh.exe' } else { 'powershell.exe' }
    Start-Process -FilePath $helperShell -ArgumentList '-NoProfile', '-EncodedCommand', $encoded -WindowStyle Hidden
}

& $pythonPath -m tracerag.cli serve --port $Port
if ($LASTEXITCODE -ne 0) {
    throw "TraceRAG 启动失败（退出码 $LASTEXITCODE）。请查看上面的错误信息；不要关闭这个窗口后忽略错误。"
}
