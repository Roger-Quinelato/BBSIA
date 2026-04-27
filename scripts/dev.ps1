param(
  [Parameter(Mandatory = $false)]
  [ValidateSet("test", "lint", "format", "typecheck", "run", "reprocess", "install", "install-dev")]
  [string]$Task = "test"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\\.venv\\Scripts\\python.exe"
$pip = Join-Path $PSScriptRoot "..\\.venv\\Scripts\\pip.exe"
$uvicorn = Join-Path $PSScriptRoot "..\\.venv\\Scripts\\uvicorn.exe"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Set-Location $repoRoot

switch ($Task) {
  "install" {
    & $pip install -r requirements.txt
    break
  }
  "install-dev" {
    & $pip install -r requirements.txt
    & $pip install ruff mypy
    break
  }
  "test" {
    $env:PYTHONPATH = "."
    & $python -m pytest
    break
  }
  "lint" {
    & $python -m ruff check .
    break
  }
  "format" {
    & $python -m ruff format .
    break
  }
  "typecheck" {
    & $python -m mypy api.py rag_engine.py reprocess_worker.py
    break
  }
  "run" {
    & $uvicorn api:app --host 0.0.0.0 --port 8000
    break
  }
  "reprocess" {
    Invoke-RestMethod -Method Post -Uri "http://localhost:8000/reprocessar"
    break
  }
}

