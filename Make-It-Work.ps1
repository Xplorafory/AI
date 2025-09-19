param(
  [string]$ComposeFile = "infra\docker-compose.pgvector.yml",
  [switch]$Ingest = $true,
  [switch]$Query = $true
)

function Wait-Docker { for ($i=0;$i -lt 60;$i++){ try{ docker info | Out-Null; return } catch { Start-Sleep 2 } } throw "Docker not ready" }

# Start Docker if needed
try { docker info | Out-Null } catch {
  Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  Wait-Docker
}

# Ensure env exists and contains PG vars
$envPath = Join-Path "infra" "infra.env"
if (-not (Test-Path $envPath)) { throw "Missing infra\infra.env" }
$envText = Get-Content $envPath -Raw
if ($envText -notmatch "PGUSER=" -or $envText -notmatch "PGDATABASE=" -or $envText -notmatch "PGPASSWORD=") {
  throw "infra\infra.env must define PGUSER, PGDATABASE, PGPASSWORD"
}

# Up containers
Write-Host "docker compose up -d"
docker compose -f $ComposeFile up -d

# Wait health
for ($i=0;$i -lt 60;$i++){
  $state = (docker inspect -f '{{json .State.Health.Status}}' xploraforys_pg) 2>$null
  if ($state -match "healthy"){ break }
  Start-Sleep 2
}

# Run migrate (uses env_file in compose)
Write-Host "Running migrate..."
docker compose -f $ComposeFile run --rm migrate

# Health check
if (Test-Path .\health_check.py) {
  python .\health_check.py
}

# Ingest / Query (optional)
if ($Ingest -and (Test-Path .\src\ingest.py)) { python .\src\ingest.py }
if ($Query -and (Test-Path .\src\rag_query.py)) { python .\src\rag_query.py }

Write-Host "All done."