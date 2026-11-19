param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("aca", "vm", "local")]
    [string]$EnvName,
    [Parameter(Mandatory = $true)]
    [string]$TargetUrl,
    [string]$ApiKey = "bench-secret-key-change-me",
    [int[]]$BatchSizes = @(1, 8, 32),
    [int[]]$VUs = @(1, 10, 50),
    [int]$Repeats = 3,
    [string]$Duration = "2m"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "results/raw" | Out-Null

foreach ($batch in $BatchSizes) {
    foreach ($vu in $VUs) {
        for ($rep = 1; $rep -le $Repeats; $rep++) {
            $outFile = "results/raw/${EnvName}_b${batch}_vu${vu}_r${rep}.json"
            Write-Host "Running [$EnvName] batch=$batch vu=$vu repeat=$rep -> $outFile"
            k6 run `
                -e TARGET_URL="$TargetUrl" `
                -e API_KEY="$ApiKey" `
                -e BATCH_SIZE="$batch" `
                -e VUS="$vu" `
                -e DURATION="$Duration" `
                --summary-export="$outFile" `
                loadtest/predict.js
            Start-Sleep -Seconds 10
        }
    }
}
Write-Host "Matrix completed! Raw files saved in results/raw/"

