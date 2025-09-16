# Setup script for Docker deployment
# Usage: .\setup_docker.ps1 [dev|prod] [api|train|inference|jupyter]

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "prod")]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("api", "train", "inference", "jupyter")]
    [string]$Mode = "api"
)

Write-Host "🐳 Setting up 3D LDM Docker environment..." -ForegroundColor Cyan

# Check if Docker is installed and running
try {
    docker --version | Out-Null
    docker info | Out-Null
} catch {
    Write-Error "Docker is not installed or not running. Please install Docker Desktop and ensure it's running."
    exit 1
}

# Check for NVIDIA Docker support (for GPU training)
if ($Mode -eq "train") {
    try {
        docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
        Write-Host "✅ NVIDIA Docker support detected" -ForegroundColor Green
    } catch {
        Write-Warning "NVIDIA Docker support not detected. GPU training may not work."
    }
}

# Build the Docker image
$ImageTag = "3d-ldm:$Environment"
Write-Host "📦 Building Docker image: $ImageTag" -ForegroundColor Yellow

$BuildArgs = @()
if ($Environment -eq "prod") {
    $BuildArgs += "--target", "production"
}

$BuildCommand = @("docker", "build") + $BuildArgs + @("-t", $ImageTag, "-f", "mlops/docker/Dockerfile", ".")
& $BuildCommand[0] $BuildCommand[1..($BuildCommand.Length-1)]

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed!"
    exit 1
}

Write-Host "✅ Docker image built successfully!" -ForegroundColor Green

# Create Docker network if it doesn't exist
$NetworkName = "3d-ldm-network"
$ExistingNetwork = docker network ls --filter name=$NetworkName --format "{{.Name}}"
if (-not $ExistingNetwork) {
    Write-Host "🌐 Creating Docker network: $NetworkName" -ForegroundColor Yellow
    docker network create $NetworkName
}

# Create volumes for persistent data
$Volumes = @("3d-ldm-models", "3d-ldm-data", "3d-ldm-outputs", "3d-ldm-logs")
foreach ($Volume in $Volumes) {
    $ExistingVolume = docker volume ls --filter name=$Volume --format "{{.Name}}"
    if (-not $ExistingVolume) {
        Write-Host "💾 Creating Docker volume: $Volume" -ForegroundColor Yellow
        docker volume create $Volume
    }
}

# Run container based on mode
$ContainerName = "3d-ldm-$Mode"
$RunArgs = @(
    "docker", "run", "-d",
    "--name", $ContainerName,
    "--network", $NetworkName,
    "-v", "3d-ldm-models:/app/models",
    "-v", "3d-ldm-data:/app/data", 
    "-v", "3d-ldm-outputs:/app/outputs",
    "-v", "3d-ldm-logs:/app/logs",
    "-e", "MODE=$Mode"
)

# Add GPU support for training
if ($Mode -eq "train") {
    $RunArgs += "--gpus", "all"
}

# Add port mapping for API mode
if ($Mode -eq "api") {
    $RunArgs += "-p", "8000:8000"
}

# Add port mapping for Jupyter mode
if ($Mode -eq "jupyter") {
    $RunArgs += "-p", "8888:8888"
}

# Stop and remove existing container if it exists
$ExistingContainer = docker ps -a --filter name=$ContainerName --format "{{.Names}}"
if ($ExistingContainer) {
    Write-Host "🛑 Stopping existing container: $ContainerName" -ForegroundColor Yellow
    docker stop $ContainerName
    docker rm $ContainerName
}

$RunArgs += $ImageTag

Write-Host "🚀 Starting container: $ContainerName" -ForegroundColor Yellow
& $RunArgs[0] $RunArgs[1..($RunArgs.Length-1)]

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start container!"
    exit 1
}

# Display container information
Start-Sleep -Seconds 3
$ContainerInfo = docker ps --filter name=$ContainerName --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
Write-Host "`n📊 Container Status:" -ForegroundColor Cyan
Write-Host $ContainerInfo

# Display relevant URLs and commands
Write-Host "`n🎯 Quick Commands:" -ForegroundColor Cyan
Write-Host "View logs:     docker logs -f $ContainerName" -ForegroundColor White
Write-Host "Stop container: docker stop $ContainerName" -ForegroundColor White
Write-Host "Remove container: docker rm $ContainerName" -ForegroundColor White

if ($Mode -eq "api") {
    Write-Host "`n🌐 API Endpoints:" -ForegroundColor Cyan
    Write-Host "Health Check:  http://localhost:8000/health" -ForegroundColor White
    Write-Host "API Docs:      http://localhost:8000/docs" -ForegroundColor White
    Write-Host "Generate:      POST http://localhost:8000/generate" -ForegroundColor White
    Write-Host "Metrics:       http://localhost:8000/metrics" -ForegroundColor White
}

if ($Mode -eq "jupyter") {
    Write-Host "`n📓 Jupyter Notebook:" -ForegroundColor Cyan
    Write-Host "URL:           http://localhost:8888" -ForegroundColor White
    Write-Host "Get token:     docker logs $ContainerName | Select-String 'token='" -ForegroundColor White
}

Write-Host "`n✅ Docker setup complete!" -ForegroundColor Green