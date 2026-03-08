param(
    [string]$RunTag = "20260302_013133"
)

$ErrorActionPreference = "Stop"

$root = "outputs/fl_comp/$RunTag"
$matrixDir = Join-Path $root "B_matrix_tuned"
$configDir = Join-Path $root "configs"
$analysisDir = Join-Path $root "analysis/B_matrix_tuned"
$plotsDir = Join-Path $root "analysis/plots"

$alpha = 0.5
$algo = "fedprox"
$mu = 0.02
$seed = 42
$numRounds = 60

$cfgInvFalse = Join-Path $configDir "hybrid_opt_tuned_inv_false.yaml"
$cfgInvTrue = Join-Path $configDir "hybrid_opt_tuned_inv_true.yaml"

$jobs = @(
    @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='hybrid_wsn_invFalse_tuned'},
    @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='wsn'; sim=''; tag='hybrid_wsn_invTrue_tuned'},
    @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='bandwidth_first_wsn_tuned'},
    @{strategy='energy_first';    cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='energy_first_wsn_tuned'},

    @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invFalse_tuned'},
    @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invTrue_tuned'},
    @{strategy='sync'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='sync_jitter_tuned'},
    @{strategy='async'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='async_jitter_tuned'},
    @{strategy='bridge_free'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bridge_free_jitter_tuned'},
    @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bandwidth_first_jitter_tuned'},
    @{strategy='energy_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='energy_first_jitter_tuned'}
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

function Get-LatestCsv([string]$dir) {
    if (!(Test-Path $dir)) { return $null }
    $files = Get-ChildItem -Path $dir -Filter *.csv | Sort-Object LastWriteTime -Descending
    if ($files.Count -eq 0) { return $null }
    return $files[0]
}

$metricDir = "outputs/hybrid_metrics/hybrid_opt"

foreach ($j in $jobs) {
    $outDir = Join-Path $matrixDir $j.tag
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $existing = Get-LatestCsv $outDir
    if ($null -ne $existing -and (Test-ValidCsv -csvPath $existing.FullName -expectedRounds $numRounds)) {
        Write-Host "[SKIP] $($j.tag) already valid: $($existing.Name)"
        continue
    }

    $ok = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "[RUN] $($j.tag) attempt $attempt/3"

        Get-Process | Where-Object { $_.ProcessName -eq 'python' } | Stop-Process -Force -ErrorAction SilentlyContinue

        if (Test-Path $metricDir) {
            Get-ChildItem -Path $metricDir -Filter *.csv -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }

        $log = Join-Path $outDir ("run_" + $j.tag + "_attempt" + $attempt + ".log")
        $err = Join-Path $outDir ("run_" + $j.tag + "_attempt" + $attempt + ".err")
        if (Test-Path $log) { Remove-Item -Force $log -ErrorAction SilentlyContinue }
        if (Test-Path $err) { Remove-Item -Force $err -ErrorAction SilentlyContinue }

        $args = @(
            "-u", ".\\src\\flower\\hybrid_opt_demo.py",
            "--strategy", $j.strategy,
            "--config", $j.cfg,
            "--num-rounds", "$numRounds",
            "--alpha", "$alpha",
            "--algorithm", $algo,
            "--fedprox-mu", "$mu",
            "--seed", "$seed",
            "--wireless-model", $j.wireless
        )
        if ($j.wireless -eq "simulated" -and -not [string]::IsNullOrWhiteSpace($j.sim)) {
            $args += @("--simulated-mode", $j.sim)
        }

        $p = Start-Process -FilePath ".\\.venv\\Scripts\\python.exe" -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
        Wait-Process -Id $p.Id

        if ($p.ExitCode -ne 0) {
            Write-Host "[WARN] non-zero exit code for $($j.tag): $($p.ExitCode)"
            Start-Sleep -Seconds 3
            continue
        }

        $srcCsv = Get-LatestCsv $metricDir
        if ($null -eq $srcCsv) {
            Write-Host "[WARN] no metrics csv in $metricDir"
            Start-Sleep -Seconds 3
            continue
        }

        $dst = Join-Path $outDir ("$($j.tag)_" + $srcCsv.Name)
        Move-Item -Force -Path $srcCsv.FullName -Destination $dst

        if (-not (Test-ValidCsv -csvPath $dst -expectedRounds $numRounds)) {
            Write-Host "[WARN] invalid csv for $($j.tag): $dst"
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
    $csv = Get-LatestCsv $outDir
    if ($null -eq $csv) { throw "missing csv for $($j.tag)" }
    if (-not (Test-ValidCsv -csvPath $csv.FullName -expectedRounds $numRounds)) { throw "invalid final csv for $($j.tag)" }
    "$($j.strategy),$($j.wireless),$($j.sim),$alpha,$algo,$seed,$mu,$($csv.FullName)" | Add-Content -Path $manifest
}

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
New-Item -ItemType Directory -Force -Path $plotsDir | Out-Null

.\.venv\Scripts\python.exe analyze_results.py --run-dir $matrixDir --out-dir $analysisDir
if ($LASTEXITCODE -ne 0) { throw "analyze_results failed" }

.\.venv\Scripts\python.exe generate_comparison_plots.py --metrics-root $matrixDir --out-dir $plotsDir
if ($LASTEXITCODE -ne 0) { throw "generate_comparison_plots failed" }

Write-Host "RESUME_DIRECT_DONE runTag=$RunTag"
