param(
    [string]$RunTag = "20260302_013133"
)

$ErrorActionPreference = "Stop"

$root = "outputs/fl_comp/$RunTag"
$matrixDir = Join-Path $root "B_matrix_tuned"
$configDir = Join-Path $root "configs"
$analysisDir = Join-Path $root "analysis/B_matrix_tuned"
$plotsDir = Join-Path $root "analysis/plots"

New-Item -ItemType Directory -Force -Path $matrixDir | Out-Null
New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
New-Item -ItemType Directory -Force -Path $plotsDir | Out-Null

$manifest = Join-Path $matrixDir "matrix_manifest.csv"
"strategy,wireless,sim_mode,alpha,algorithm,seed,fedprox_mu,csv_path" | Out-File -FilePath $manifest -Encoding utf8

$alpha = 0.5
$algo = "fedprox"
$mu = 0.02
$seed = 42
$numRounds = 60

$cfgInvFalse = Join-Path $configDir "hybrid_opt_tuned_inv_false.yaml"
$cfgInvTrue = Join-Path $configDir "hybrid_opt_tuned_inv_true.yaml"

$jobs = @(
    @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='hybrid_wsn_invFalse_tuned'; metrics=''},
    @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='wsn'; sim=''; tag='hybrid_wsn_invTrue_tuned';  metrics=''},
    @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='bandwidth_first_wsn_tuned'; metrics=''},
    @{strategy='energy_first';    cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='energy_first_wsn_tuned';    metrics=''},

    @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invFalse_tuned'; metrics=''},
    @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invTrue_tuned';  metrics=''},
    @{strategy='sync'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='sync_jitter_tuned'; metrics=''},
    @{strategy='async'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='async_jitter_tuned'; metrics=''},
    @{strategy='bridge_free'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bridge_free_jitter_tuned'; metrics=''},
    @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bandwidth_first_jitter_tuned'; metrics=''},
    @{strategy='energy_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='energy_first_jitter_tuned'; metrics=''}
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

foreach ($j in $jobs) {
    $outDir = Join-Path $matrixDir $j.tag
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $ok = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "[$($j.tag)] outer-attempt $attempt/3"

        $args = @{
            Strategy = $j.strategy
            ConfigPath = $j.cfg
            WirelessModel = $j.wireless
            SimulatedMode = $j.sim
            NumRounds = $numRounds
            Alpha = $alpha
            Algorithm = $algo
            FedproxMu = $mu
            Seed = $seed
            OutDir = $outDir
            Tag = $j.tag
        }

        if (-not [string]::IsNullOrWhiteSpace($j.metrics)) {
            $args["MetricsStrategyName"] = $j.metrics
        }

        & .\run_one.ps1 @args
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[$($j.tag)] run_one failed with code=$LASTEXITCODE"
            Start-Sleep -Seconds 3
            continue
        }

        $csv = Get-ChildItem -Path $outDir -Filter *.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -eq $csv) {
            Write-Host "[$($j.tag)] no csv produced"
            Start-Sleep -Seconds 3
            continue
        }

        if (-not (Test-ValidCsv -csvPath $csv.FullName -expectedRounds $numRounds)) {
            Write-Host "[$($j.tag)] invalid csv: $($csv.FullName)"
            Start-Sleep -Seconds 3
            continue
        }

        "$($j.strategy),$($j.wireless),$($j.sim),$alpha,$algo,$seed,$mu,$($csv.FullName)" | Add-Content -Path $manifest
        $ok = $true
        break
    }

    if (-not $ok) {
        throw "job failed after retries: $($j.tag)"
    }
}

.\.venv\Scripts\python.exe analyze_results.py --run-dir $matrixDir --out-dir $analysisDir
if ($LASTEXITCODE -ne 0) { throw "analyze_results failed" }

.\.venv\Scripts\python.exe generate_comparison_plots.py --metrics-root $matrixDir --out-dir $plotsDir
if ($LASTEXITCODE -ne 0) { throw "generate_comparison_plots failed" }

Write-Host "DONE runTag=$RunTag"
