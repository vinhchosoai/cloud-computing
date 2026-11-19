param(
    [string]$ResourceGroup = "rg-ai-inference",
    [string]$Location = "eastasia",
    [string]$AcrName = "acraiinference$((Get-Random -Minimum 1000 -Maximum 9999))",
    [string]$EnvName = "aca-env-ai-inference",
    [string]$AppName = "ai-inference-api",
    [string]$ImageTag = "v1.0.0",
    [string]$ApiKey = "dev-secret-key-change-me",
    [string]$BenchmarkKey = "bench-secret-key-change-me",
    [int]$MinReplicas = 0,
    [int]$MaxReplicas = 5
)

$ErrorActionPreference = "Stop"
$az = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"

Write-Host "[1/4] Creating Azure Container Registry: $AcrName ..."
& $az acr create --resource-group $ResourceGroup --name $AcrName --sku Basic --admin-enabled true --location $Location -o table

Write-Host "[2/4] Building & pushing Docker image remotely via ACR Build ..."
& $az acr build --registry $AcrName --image "ai-inference-api:$ImageTag" .

$acrServer = & $az acr show --name $AcrName --query loginServer -o tsv
$acrUser = & $az acr credential show --name $AcrName --query username -o tsv
$acrPass = & $az acr credential show --name $AcrName --query "passwords[0].value" -o tsv

Write-Host "[3/4] Creating Container Apps Environment: $EnvName ..."
& $az containerapp env create --name $EnvName --resource-group $ResourceGroup --location $Location -o table

Write-Host "[4/4] Deploying Container App: $AppName (1 vCPU, 2 GiB, min=$MinReplicas, max=$MaxReplicas) ..."
& $az containerapp create `
    --name $AppName `
    --resource-group $ResourceGroup `
    --environment $EnvName `
    --image "$acrServer/ai-inference-api:$ImageTag" `
    --registry-server $acrServer `
    --registry-username $acrUser `
    --registry-password $acrPass `
    --target-port 8000 `
    --ingress external `
    --cpu 1.0 `
    --memory 2.0Gi `
    --min-replicas $MinReplicas `
    --max-replicas $MaxReplicas `
    --scale-rule-name http-concurrency `
    --scale-rule-type http `
    --scale-rule-http-concurrency 10 `
    --secrets "api-keys=$ApiKey,$BenchmarkKey" "bench-key=$BenchmarkKey" `
    --env-vars `
        "API_KEYS=secretref:api-keys" `
        "BENCHMARK_API_KEY=secretref:bench-key" `
        "RATE_LIMIT=60/minute" `
        "BENCHMARK_RATE_LIMIT=10000/minute" `
        "MODEL_VERSION=$ImageTag" `
    -o table

$fqdn = & $az containerapp show --name $AppName --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv
Write-Host "ACA deployed! URL: https://$fqdn"

