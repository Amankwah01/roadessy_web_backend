$env:TEMP = "F:\temp"
$env:TMP = "F:\temp"

# The API only enqueues survey processing; the worker process
# claims the queued tasks and runs the heavy pipeline. The
# worker runs hidden with its output in logs\worker.log, so
# starting the system is one command and one window (the API).
$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Start-Process `
    -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m","app.worker" `
    -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "worker-stdout.log") `
    -RedirectStandardError (Join-Path $logDir "worker.log")

.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000
