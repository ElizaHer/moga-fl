$ErrorActionPreference = "Stop"

$strategies = @("hybrid_opt", "sync", "async", "bridge_free", "bandwidth_first", "energy_first")
$modes = @("good", "bad", "jitter")
$alphas = @(0.1, 0.5, 1.0)
$algorithms = @("fedavg")
$seeds = @(42, 52, 62, 72, 82)

$runTag = (Get-Date).ToString("yyyyMMdd_HHmmss")
$outRoot = "outputs/hybrid_runs/$runTag"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$manifest = Join-Path $outRoot "manifest.csv"
"run_tag,strategy,mode,alpha,algorithm,seed,csv_path" | Out-File -FilePath $manifest -Encoding utf8

function Get-LatestCsv([string]$dir) {
    if (!(Test-Path $dir)) { return $null }
    $files = Get-ChildItem -Path $dir -Filter *.csv | Sort-Object LastWriteTime -Descending
    if ($files.Count -eq 0) { return $null }
    return $files[0].FullName
}

foreach ($strategy in $strategies) {
    foreach ($mode in $modes) {
        foreach ($alpha in $alphas) {
            foreach ($algo in $algorithms) {
                foreach ($seed in $seeds) {
                    Write-Host "Running $strategy | mode=$mode | alpha=$alpha | algo=$algo | seed=$seed"
                    python .\src\flower\hybrid_opt_demo.py `
                        --strategy $strategy `
                        --wireless-model simulated `
                        --simulated-mode $mode `
                        --alpha $alpha `
                        --algorithm $algo `
                        --seed $seed

                    $srcDir = "outputs/hybrid_metrics/$strategy"
                    $latest = Get-LatestCsv $srcDir
                    if ($null -ne $latest) {
                        $dstDir = Join-Path $outRoot $strategy
                        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
                        $dstPath = Join-Path $dstDir ([IO.Path]::GetFileName($latest))
                        Move-Item -Force -Path $latest -Destination $dstPath
                        "$runTag,$strategy,$mode,$alpha,$algo,$seed,$dstPath" | Add-Content -Path $manifest
                    }
                }
            }
        }
    }
}
