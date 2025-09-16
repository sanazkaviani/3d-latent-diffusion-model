# Azure Infrastructure Deployment Script
# Usage: .\deploy_azure.ps1 -Environment [dev|staging|prod] -ResourceGroup [name] -Location [location]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev", "staging", "prod")]
    [string]$Environment,
    
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup,
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "East US",
    
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$false)]
    [switch]$EnableGpu = $true,
    
    [Parameter(Mandatory=$false)]
    [int]$AksNodeCount = 3
)

Write-Host "🚀 Deploying 3D LDM Infrastructure to Azure..." -ForegroundColor Cyan

# Check if Azure CLI is installed
try {
    az --version | Out-Null
} catch {
    Write-Error "Azure CLI is not installed. Please install Azure CLI first."
    exit 1
}

# Login to Azure if not already logged in
$Account = az account show --query "user.name" -o tsv 2>$null
if (-not $Account) {
    Write-Host "🔐 Logging in to Azure..." -ForegroundColor Yellow
    az login
}

# Set subscription if provided
if ($SubscriptionId) {
    Write-Host "📋 Setting subscription: $SubscriptionId" -ForegroundColor Yellow
    az account set --subscription $SubscriptionId
}

$CurrentSubscription = az account show --query "name" -o tsv
Write-Host "✅ Using subscription: $CurrentSubscription" -ForegroundColor Green

# Create resource group if it doesn't exist
$ExistingRG = az group show --name $ResourceGroup --query "name" -o tsv 2>$null
if (-not $ExistingRG) {
    Write-Host "📁 Creating resource group: $ResourceGroup" -ForegroundColor Yellow
    az group create --name $ResourceGroup --location $Location
} else {
    Write-Host "✅ Resource group exists: $ResourceGroup" -ForegroundColor Green
}

# Deploy ARM template
Write-Host "🏗️ Deploying ARM template..." -ForegroundColor Yellow
$DeploymentName = "3dldm-deployment-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

$TemplateParams = @{
    resourcePrefix = "3dldm"
    location = $Location
    environment = $Environment
    aksNodeCount = $AksNodeCount
    enableGpu = $EnableGpu.IsPresent
}

# Create parameters file
$ParamsJson = @{
    "`$schema" = "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#"
    contentVersion = "1.0.0.0"
    parameters = @{}
}

foreach ($key in $TemplateParams.Keys) {
    $ParamsJson.parameters[$key] = @{ value = $TemplateParams[$key] }
}

$ParamsFile = "mlops/azure/deployment-params-$Environment.json"
$ParamsJson | ConvertTo-Json -Depth 10 | Out-File -FilePath $ParamsFile -Encoding UTF8

try {
    $DeploymentResult = az deployment group create `
        --resource-group $ResourceGroup `
        --name $DeploymentName `
        --template-file "mlops/azure/infrastructure.json" `
        --parameters "@$ParamsFile" `
        --output json | ConvertFrom-Json
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Infrastructure deployment completed successfully!" -ForegroundColor Green
        
        # Extract output values
        $Outputs = $DeploymentResult.properties.outputs
        
        Write-Host "`n📊 Deployment Outputs:" -ForegroundColor Cyan
        foreach ($key in $Outputs.PSObject.Properties.Name) {
            Write-Host "  $key`: $($Outputs.$key.value)" -ForegroundColor White
        }
        
        # Save outputs to file for later use
        $OutputsFile = "mlops/azure/deployment-outputs-$Environment.json"
        $Outputs | ConvertTo-Json -Depth 3 | Out-File -FilePath $OutputsFile -Encoding UTF8
        Write-Host "`n💾 Outputs saved to: $OutputsFile" -ForegroundColor Yellow
        
    } else {
        Write-Error "❌ Infrastructure deployment failed!"
        exit 1
    }
} catch {
    Write-Error "❌ Deployment error: $_"
    exit 1
}

# Configure AKS credentials
$AksName = $Outputs.aksClusterName.value
Write-Host "`n🔐 Configuring AKS credentials..." -ForegroundColor Yellow
az aks get-credentials --resource-group $ResourceGroup --name $AksName --overwrite-existing

# Verify AKS connection
Write-Host "🔍 Verifying AKS connection..." -ForegroundColor Yellow
kubectl cluster-info

# Configure ACR integration with AKS
$AcrName = $Outputs.acrLoginServer.value -replace '\.azurecr\.io$', ''
Write-Host "🔗 Configuring ACR integration..." -ForegroundColor Yellow
az aks update --resource-group $ResourceGroup --name $AksName --attach-acr $AcrName

# Install GPU driver daemonset if GPU is enabled
if ($EnableGpu) {
    Write-Host "🎮 Installing NVIDIA GPU drivers..." -ForegroundColor Yellow
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
}

# Install monitoring components
Write-Host "📊 Installing monitoring components..." -ForegroundColor Yellow

# Install Prometheus and Grafana using Helm
try {
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack `
        --namespace monitoring --create-namespace `
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi
    
    Write-Host "✅ Monitoring components installed!" -ForegroundColor Green
} catch {
    Write-Warning "⚠️ Failed to install monitoring components. You can install them manually later."
}

# Create namespace for the application
Write-Host "📁 Creating application namespace..." -ForegroundColor Yellow
kubectl create namespace 3d-ldm --dry-run=client -o yaml | kubectl apply -f -

# Set up Azure ML workspace connection
$MlWorkspaceName = $Outputs.mlWorkspaceName.value
Write-Host "🤖 Configuring Azure ML workspace..." -ForegroundColor Yellow

# Install Azure ML CLI extension
az extension add --name ml --upgrade --yes

# Create a compute cluster in Azure ML
try {
    az ml compute create `
        --resource-group $ResourceGroup `
        --workspace-name $MlWorkspaceName `
        --name "gpu-cluster" `
        --type amlcompute `
        --size "Standard_NC6s_v3" `
        --min-instances 0 `
        --max-instances 4
    
    Write-Host "✅ Azure ML compute cluster created!" -ForegroundColor Green
} catch {
    Write-Warning "⚠️ Failed to create Azure ML compute cluster. You can create it manually in the Azure portal."
}

Write-Host "`n🎉 Azure infrastructure deployment completed!" -ForegroundColor Green
Write-Host "`n🔧 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Build and push Docker image to ACR: docker build -t $($Outputs.acrLoginServer.value)/3d-ldm:latest ." -ForegroundColor White
Write-Host "2. Deploy to Kubernetes: .\mlops\kubernetes\setup_kubernetes.sh apply" -ForegroundColor White
Write-Host "3. Configure monitoring dashboards in Grafana" -ForegroundColor White
Write-Host "4. Set up Azure ML experiments and pipelines" -ForegroundColor White

Write-Host "`n📋 Important Information:" -ForegroundColor Cyan
Write-Host "ACR Login Server: $($Outputs.acrLoginServer.value)" -ForegroundColor White
Write-Host "AKS Cluster Name: $($Outputs.aksClusterName.value)" -ForegroundColor White
Write-Host "ML Workspace: $($Outputs.mlWorkspaceName.value)" -ForegroundColor White
Write-Host "Storage Account: $($Outputs.storageAccountName.value)" -ForegroundColor White