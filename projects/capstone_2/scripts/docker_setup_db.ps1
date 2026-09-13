[CmdletBinding()]
param([switch]$Reset)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Container = "opsmate-mysql"
$RootPw    = "rootpw"
$Database  = "opsmate"

if ($Reset) {
    Write-Host "Removing existing container and data volume..." -ForegroundColor Yellow
    docker compose down -v
}

Write-Host "Starting MySQL..." -ForegroundColor Cyan
docker compose up -d
if (-not $?) { throw "docker compose up failed. Is Docker Desktop running?" }

Write-Host "Waiting for the container to report healthy..." -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes(3)
$status = ""
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $status = (docker inspect --format "{{.State.Health.Status}}" $Container).Trim()
    Write-Host "  health: $status"
    if ($status -eq "healthy") { break }
}
if ($status -ne "healthy") {
    throw "MySQL did not become healthy within 3 minutes. Check: docker logs $Container"
}

Write-Host "Applying db/schema.sql..." -ForegroundColor Cyan
Get-Content "db/schema.sql" -Raw | docker exec -i $Container mysql -uroot -p$RootPw
if (-not $?) { throw "Schema application failed." }

$tables = docker exec $Container mysql -uroot -p$RootPw -N -B -e `
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='$Database';"
Write-Host ""
Write-Host "Ready. Database '$Database' has $($tables.Trim()) tables." -ForegroundColor Green
Write-Host "Next: python scripts/populate_database.py" -ForegroundColor Green
