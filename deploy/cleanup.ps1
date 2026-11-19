param(
    [string]$ResourceGroup = "rg-ai-inference",
    [switch]$Force
)

$az = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"

if (-not $Force) {
    $confirm = Read-Host "This will DELETE resource group '$ResourceGroup' and ALL resources inside it. Type 'yes' to confirm"
    if ($confirm -ne "yes") {
        Write-Host "Aborted."
        exit 0
    }
}

Write-Host "Deleting resource group $ResourceGroup ..."
& $az group delete --name $ResourceGroup --yes --no-wait
Write-Host "Deletion initiated asynchronously. Verify on Azure Portal that no resources remain."

