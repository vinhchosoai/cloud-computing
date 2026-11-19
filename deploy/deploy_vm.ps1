param(
  [string]$ResourceGroup = "rg-ai-inference",
  [string]$Location = "eastasia",
  [string]$VmName = "vm-ai-inference",
  [string]$VmSize = "Standard_B2als_v2",
  [string]$AdminUser = "azureuser"
)

$ErrorActionPreference = "Stop"
$az = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"

$myIp = (Invoke-RestMethod -Uri "https://api.ipify.org") + "/32"
Write-Host "Detected your public IP for SSH restriction: $myIp"

$cloudInit = @"
#cloud-config
package_update: true
packages:
  - docker.io
  - docker-compose-v2
runcmd:
  - systemctl enable --now docker
  - usermod -aG docker $AdminUser
"@
$cloudInitPath = "$env:TEMP\cloud-init-ai.yaml"
$cloudInit | Set-Content -Path $cloudInitPath -Encoding UTF8

Write-Host "Creating VM $VmName ($VmSize) in $ResourceGroup ($Location)..."
& $az vm create `
  --resource-group $ResourceGroup `
  --name $VmName `
  --location $Location `
  --image Ubuntu2404 `
  --size $VmSize `
  --admin-username $AdminUser `
  --generate-ssh-keys `
  --custom-data $cloudInitPath `
  --public-ip-sku Standard `
  -o table

$nsgName = "${VmName}NSG"
Write-Host "Restricting NSG ${nsgName}: SSH from $myIp only, opening 8000 (API) & 3000 (Grafana)..."
& $az network nsg rule update --resource-group $ResourceGroup --nsg-name $nsgName --name default-allow-ssh --source-address-prefixes $myIp -o none
& $az network nsg rule create --resource-group $ResourceGroup --nsg-name $nsgName --name allow-api-8000 --priority 1010 --destination-port-ranges 8000 --protocol Tcp --access Allow -o none
& $az network nsg rule create --resource-group $ResourceGroup --nsg-name $nsgName --name allow-grafana-3000 --priority 1020 --destination-port-ranges 3000 --protocol Tcp --access Allow -o none

$pubIp = & $az vm show -d -g $ResourceGroup -n $VmName --query publicIps -o tsv
Write-Host "VM deployed! Public IP: $pubIp"
Write-Host "Copy repo & start stack: scp -r . ${AdminUser}@${pubIp}:~/ai-inference && ssh ${AdminUser}@${pubIp} 'cd ~/ai-inference && docker compose up -d --build'"

