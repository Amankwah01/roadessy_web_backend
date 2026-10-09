# Stops the Roadessy stack started by start_server.ps1:
# the hidden pipeline worker and the API server (both the
# uvicorn reloader and its server child). A pipeline interrupted
# this way is requeued by its stale heartbeat and re-run by the
# next worker, so force-stopping is safe.
$stopped = 0

Get-CimInstance Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%app.worker%'" |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped += 1
    }

Get-CimInstance Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%app.main:app%'" |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped += 1
    }

Write-Host "Stopped $stopped process(es)."
