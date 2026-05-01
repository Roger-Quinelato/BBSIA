param(
  [Parameter(Mandatory = $false)]
  [ValidateSet("test", "lint", "format", "typecheck", "run", "reprocess", "solucoes-embedding", "install", "install-dev")]
  [string]$Task = "test"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$pip = Join-Path $repoRoot ".venv\\Scripts\\pip.exe"
$uvicorn = Join-Path $repoRoot ".venv\\Scripts\\uvicorn.exe"

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
    & $python -m mypy bbsia
    break
  }
  "run" {
    & $uvicorn bbsia.app.bootstrap.main:app --host 0.0.0.0 --port 8000
    break
  }
  "reprocess" {
    Invoke-RestMethod -Method Post -Uri "http://localhost:8000/reprocessar"
    break
  }
  "solucoes-embedding" {
    & $python -m bbsia.cli.gerar_embeddings_solucoes
    break
  }
}
