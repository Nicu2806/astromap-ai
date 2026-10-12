$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    throw 'Python environment missing. Configure the workspace Python environment first.'
}

Start-Process -FilePath $python -ArgumentList '-m','uvicorn','astromap.api:app','--host','127.0.0.1','--port','8001' -WorkingDirectory $root
Start-Process -FilePath 'cargo' -ArgumentList 'run','--manifest-path','apps/api-rust/Cargo.toml' -WorkingDirectory $root
Start-Process -FilePath 'flutter' -ArgumentList 'run','-d','web-server','--web-port','8088' -WorkingDirectory (Join-Path $root 'apps\client-flutter')

Write-Host 'AstroMap-AI services started:'
Write-Host '  Flutter  http://127.0.0.1:8088'
Write-Host '  Rust API http://127.0.0.1:8080/health/ready'
Write-Host '  Python   http://127.0.0.1:8001/docs'
