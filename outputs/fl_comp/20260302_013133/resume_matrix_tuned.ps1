param(
    [string]$RunTag = "20260302_013133"
)

$ErrorActionPreference = "Stop"

$root = "outputs/fl_comp/$RunTag"
$matrixDir = Join-Path $root "B_matrix_tuned"
$configDir = Join-Path $root "configs"

$alpha = 0.5
$algo = "fedprox"
$mu = 0.02
$seed = 42
$numRounds = 60

$cfgInvFalse = Join-Path $configDir "hybrid_opt_tuned_inv_false.yaml"
$cfgInvTrue = Join-Path $configDir "hybrid_opt_tuned_inv_true.yaml"

$jobs = @(
    @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='hybrid_wsn_invFalse_tuned'; metrics='hybrid_opt'},
    @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='wsn'; sim=''; tag='hybrid_wsn_invTrue_tuned';  metrics='hybrid_opt'},
    @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='bandwidth_first_wsn_tuned'; metrics='bandwidth_first'},
    @{strategy='energy_first';    cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='energy_first_wsn_tuned';    metrics='energy_first'},

    @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invFalse_tuned'; metrics='hybrid_opt'},
    @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invTrue_tuned';  metrics='hybrid_opt'},
    @{strategy='sync'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='sync_jitter_tuned'; metrics='sync'},
    @{strategy='async'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='async_jitter_tuned'; metrics='async'},
    @{strategy='bridge_free'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bridge_free_jitter_tuned'; metrics='bridge_free'},
    @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bandwidth_first_jitter_tuned'; metrics='bandwidth_first'},
    @{strategy='energy_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='energy_first_jitter_tuned'; metrics='energy_first'}
)

function Test-ValidCsv([string]$csvPath, [int]$expectedRounds) {
    if (!(Test-Path $csvPath)) { return $false }
    try {
        $rows = Import-Csv $csvPath
        if ($rows.Count -lt 1) { return $false }
        $lastRound = ($rows | Select-Object -Last 1).round
        if ([int]$lastRound -lt $expectedRounds) { return $false }
        return $true
    } catch {
        return $false
    }
}

function Get-LatestCsvInOut([string]$outDir) {
    if (!(Test-Path $outDir)) { return $null }
    return Get-ChildItem -Path $outDir -Filter *.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

foreach ($j in $jobs) {
    $outDir = Join-Path $matrixDir $j.tag
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $existing = Get-LatestCsvInOut -outDir $outDir
    if ($null -ne $existing -and (Test-ValidCsv -csvPath $existing.FullName -expectedRounds $numRounds)) {
        Write-Host "[SKIP] $($j.tag) already valid: $($existing.Name)"
        continue
    }

    $ok = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "[RUN] $($j.tag) outer-attempt $attempt/3"

        Get-Process | Where-Object { $_.ProcessName -eq 'python' } | Stop-Process -Force -ErrorAction SilentlyContinue

        $metricDir = Join-Path "outputs/hybrid_metrics" $j.metrics
        if (Test-Path $metricDir) {
            Get-ChildItem -Path $metricDir -Filter *.csv -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }

        & .\run_one.ps1 -Strategy $j.strategy -ConfigPath $j.cfg -WirelessModel $j.wireless -SimulatedMode $j.sim -NumRounds $numRounds -Alpha $alpha -Algorithm $algo -FedproxMu $mu -Seed $seed -OutDir $outDir -Tag $j.tag
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] run_one failed for $($j.tag)"
            Start-Sleep -Seconds 3
            continue
        }

        $csv = Get-LatestCsvInOut -outDir $outDir
        if ($null -eq $csv) {
            Write-Host "[WARN] no csv for $($j.tag)"
            Start-Sleep -Seconds 3
            continue
        }

        if (-not (Test-ValidCsv -csvPath $csv.FullName -expectedRounds $numRounds)) {
            Write-Host "[WARN] invalid csv for $($j.tag): $($csv.Name)"
            Start-Sleep -Seconds 3
            continue
        }

        $ok = $true
        break
    }

    if (-not $ok) {
        throw "job failed after retries: $($j.tag)"
    }
}

$manifest = Join-Path $matrixDir "matrix_manifest.csv"
"strategy,wireless,sim_mode,alpha,algorithm,seed,fedprox_mu,csv_path" | Out-File -FilePath $manifest -Encoding utf8
foreach ($j in $jobs) {
    $outDir = Join-Path $matrixDir $j.tag
    $csv = Get-LatestCsvInOut -outDir $outDir
    if ($null -eq $csv) { throw "missing csv for $($j.tag)" }
    if (-not (Test-ValidCsv -csvPath $csv.FullName -expectedRounds $numRounds)) {
        throw "invalid final csv for $($j.tag): $($csv.FullName)"
    }
    "$($j.strategy),$($j.wireless),$($j.sim),$alpha,$algo,$seed,$mu,$($csv.FullName)" | Add-Content -Path $manifest
}

.\.venv\Scripts\python.exe analyze_results.py --run-dir $matrixDir --out-dir (Join-Path $root "analysis/B_matrix_tuned")
if ($LASTEXITCODE -ne 0) { throw "analyze_results failed" }

.\.venv\Scripts\python.exe generate_comparison_plots.py --metrics-root $matrixDir --out-dir (Join-Path $root "analysis/plots")
if ($LASTEXITCODE -ne 0) { throw "generate_comparison_plots failed" }

Write-Host "RESUME_DONE runTag=$RunTag"
