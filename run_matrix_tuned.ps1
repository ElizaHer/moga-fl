param(
    [string]$RunTag = "",
    [string]$RunName = "B_matrix_tuned",
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$SkipRun = $false,
    [switch]$NoWechat = $false,
    [int]$NumRounds = 60,
    [double]$Alpha = 0.5,
    [string]$Algorithm = "fedprox",
    [double]$FedproxMu = 0.05,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

function New-RunTag {
    return (Get-Date).ToString("yyyyMMdd_HHmmss")
}

function Ensure-Directory([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Get-NextAttemptIndex([string]$OutDir, [string]$Tag) {
    $pattern = "^run_${Tag}_attempt(\d+)\.(log|err)$"
    if (!(Test-Path $OutDir)) { return 1 }
    $maxAttempt = 0
    Get-ChildItem -Path $OutDir -File | ForEach-Object {
        $m = [regex]::Match($_.Name, $pattern)
        if ($m.Success) {
            $n = [int]$m.Groups[1].Value
            if ($n -gt $maxAttempt) { $maxAttempt = $n }
        }
    }
    return ($maxAttempt + 1)
}

function Parse-MetricsCsvFromLog([string]$LogPath) {
    if (!(Test-Path $LogPath)) { return $null }
    $lines = Get-Content -Path $LogPath
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        $line = $lines[$i]
        $m = [regex]::Match($line, "Metrics CSV:\s*(.+)$")
        if ($m.Success) {
            $p = $m.Groups[1].Value.Trim()
            if (![string]::IsNullOrWhiteSpace($p)) { return $p }
        }
    }
    return $null
}

function Normalize-PathString([string]$PathText) {
    $p = $PathText.Replace("/", "\")
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path (Get-Location) $p)
}

function Append-Ledger(
    [string]$LedgerPath,
    [string]$Tag,
    [int]$Attempt,
    [string]$Status,
    [string]$Log,
    [string]$Err,
    [string]$MetricsCsv,
    [string]$CopiedCsv,
    [string]$Note
) {
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$timestamp,$Tag,$Attempt,$Status,$Log,$Err,$MetricsCsv,$CopiedCsv,$Note" | Add-Content -Path $LedgerPath
}

function Ensure-Ledger([string]$LedgerPath) {
    if (!(Test-Path $LedgerPath)) {
        "timestamp,tag,attempt,status,log,err,metrics_csv,copied_csv,note" | Out-File -FilePath $LedgerPath -Encoding utf8
    }
}

function Ensure-Manifest([string]$ManifestPath) {
    if (!(Test-Path $ManifestPath)) {
        "tag,strategy,wireless_model,simulated_mode,alpha,algorithm,seed,fedprox_mu,csv_path" | Out-File -FilePath $ManifestPath -Encoding utf8
    }
}

function Send-Wechat([string]$Title, [string]$Name, [string]$Content) {
    if ($NoWechat) { return }
    $token = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjkwMjkwNiwidXVpZCI6ImE5NDViOTgzNWQwOGY4YjUiLCJpc19hZG1pbiI6ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.sw0UVJSDJ8CZ7O6cT9_OQOfWQKi6TwW9JQVB0FGp8SMQueBsN-0mEETOV857BNZBouh2yy-MwJQL7VRE1u0XSg"
    $headers = @{
        Authorization = $token
        "Content-Type" = "application/json"
    }
    $payload = @{
        title = $Title
        name = $Name
        content = $Content
    } | ConvertTo-Json -Depth 6
    try {
        $null = Invoke-RestMethod -Uri "https://www.autodl.com/api/v1/wechat/message/send" -Method Post -Headers $headers -Body $payload
    } catch {
        Write-Host "[WARN] send wechat failed: $($_.Exception.Message)"
    }
}

function Read-FinalMetrics([string]$CsvPath) {
    if (!(Test-Path $CsvPath)) {
        return @{ rounds = 0; acc = 0.0; loss = 0.0 }
    }
    $rows = Import-Csv -Path $CsvPath
    if ($rows.Count -eq 0) {
        return @{ rounds = 0; acc = 0.0; loss = 0.0 }
    }
    $last = $rows[-1]
    return @{
        rounds = [int]$last.round
        acc = [double]$last.accuracy
        loss = [double]$last.loss
    }
}

function Build-ConfigFiles([string]$ConfigDir) {
    Ensure-Directory $ConfigDir
    $base = Get-Content -Path "src/configs/strategies/hybrid_opt.yaml" -Raw

    $invTruePath = Join-Path $ConfigDir "hybrid_opt_inv_true.yaml"
    $invFalsePath = Join-Path $ConfigDir "hybrid_opt_inv_false.yaml"

    # Bundle A defaults for stronger hybrid behavior in jitter; matrix still keeps alpha/seed/rounds fixed.
    $bundle = $base `
      -replace "(?m)^(\s*fedprox_mu:\s*).*$",'$1 0.05' `
      -replace "(?m)^(\s*semi_sync_wait_ratio:\s*).*$",'$1 0.85' `
      -replace "(?m)^(\s*to_async:\s*).*$",'$1 0.65' `
      -replace "(?m)^(\s*to_semi_sync:\s*).*$",'$1 0.38' `
      -replace "(?m)^(\s*bridge_rounds:\s*).*$",'$1 4' `
      -replace "(?m)^(\s*min_rounds_between_switch:\s*).*$",'$1 4' `
      -replace "(?m)^(\s*buffer_size:\s*).*$",'$1 10' `
      -replace "(?m)^(\s*min_updates_to_aggregate:\s*).*$",'$1 5' `
      -replace "(?m)^(\s*async_agg_interval:\s*).*$",'$1 1' `
      -replace "(?m)^(\s*staleness_alpha:\s*).*$",'$1 1.5' `
      -replace "(?m)^(\s*max_staleness:\s*).*$",'$1 6' `
      -replace "(?m)^(\s*channel_w:\s*).*$",'$1 0.45' `
      -replace "(?m)^(\s*data_w:\s*).*$",'$1 0.35' `
      -replace "(?m)^(\s*fair_w:\s*).*$",'$1 0.15' `
      -replace "(?m)^(\s*energy_w:\s*).*$",'$1 0.05'

    $bundle | Set-Content -Path $invTruePath -Encoding utf8
    ($bundle -replace "(?m)^(\s*enable:\s*)true\s*$",'$1false') | Set-Content -Path $invFalsePath -Encoding utf8

    return @{
        inv_true = $invTruePath
        inv_false = $invFalsePath
    }
}

function Run-JobAttempt(
    [hashtable]$Job,
    [int]$Attempt,
    [string]$PythonExePath,
    [string]$OutDir,
    [string]$LedgerPath
) {
    $tag = $Job.tag
    $logPath = Join-Path $OutDir ("run_${tag}_attempt${Attempt}.log")
    $errPath = Join-Path $OutDir ("run_${tag}_attempt${Attempt}.err")

    $args = @(
        "-u", ".\src\flower\hybrid_opt_demo.py",
        "--strategy", $Job.strategy,
        "--config", $Job.config,
        "--wireless-model", $Job.wireless_model,
        "--num-rounds", $NumRounds,
        "--alpha", $Alpha,
        "--algorithm", $Algorithm,
        "--seed", $Seed
    )
    if ($Job.simulated_mode -ne "") {
        $args += @("--simulated-mode", $Job.simulated_mode)
    }
    if ($Algorithm -eq "fedprox") {
        $args += @("--fedprox-mu", $FedproxMu)
    }

    $startContent = "tag=$tag strategy=$($Job.strategy) wm=$($Job.wireless_model) sm=$($Job.simulated_mode) alpha=$Alpha algo=$Algorithm seed=$Seed rounds=$NumRounds"
    Send-Wechat -Title "experiment_start" -Name $tag -Content $startContent

    Write-Host "Attempt $Attempt/$($Job.max_attempts): $tag"
    $p = Start-Process -FilePath $PythonExePath -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $logPath -RedirectStandardError $errPath
    Wait-Process -Id $p.Id
    $exitCode = $p.ExitCode

    $metricsRel = Parse-MetricsCsvFromLog $logPath
    $metricsAbs = $null
    $copiedCsv = ""
    $status = "failed"
    $note = ""
    if ($null -ne $metricsRel) {
        $metricsAbs = Normalize-PathString $metricsRel
        if (Test-Path $metricsAbs) {
            $baseName = Split-Path $metricsAbs -Leaf
            $dstName = "${tag}_attempt${Attempt}_${baseName}"
            $dstPath = Join-Path $OutDir $dstName
            Copy-Item -Path $metricsAbs -Destination $dstPath -ErrorAction Stop
            $copiedCsv = $dstPath
            $status = "success"
            $note = "exit_code=$exitCode; success by Metrics CSV line"
        } else {
            $note = "exit_code=$exitCode; Metrics CSV path missing on disk"
        }
    } else {
        $note = "exit_code=$exitCode; no Metrics CSV line in log"
    }

    $metricsCsvText = ""
    if ($null -ne $metricsAbs) { $metricsCsvText = $metricsAbs }
    Append-Ledger -LedgerPath $LedgerPath -Tag $tag -Attempt $Attempt -Status $status -Log $logPath -Err $errPath -MetricsCsv $metricsCsvText -CopiedCsv $copiedCsv -Note $note

    $rounds = 0
    $acc = 0.0
    $loss = 0.0
    if ($copiedCsv -ne "") {
        $m = Read-FinalMetrics $copiedCsv
        $rounds = $m.rounds
        $acc = $m.acc
        $loss = $m.loss
    }
    $endContent = "tag=$tag status=$status rounds=$rounds acc=$acc loss=$loss csv=$copiedCsv"
    Send-Wechat -Title "experiment_end" -Name $tag -Content $endContent

    return @{
        success = ($status -eq "success")
        copied_csv = $copiedCsv
    }
}

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = New-RunTag
}

$outRoot = Join-Path "outputs/fl_comp" $RunTag
$matrixDir = Join-Path $outRoot $RunName
$configDir = Join-Path $outRoot "configs"
Ensure-Directory $matrixDir
Ensure-Directory $configDir

$configs = Build-ConfigFiles $configDir
$invTrueCfg = $configs.inv_true
$invFalseCfg = $configs.inv_false

$ledgerPath = Join-Path $matrixDir "attempt_ledger.csv"
$manifestPath = Join-Path $matrixDir "matrix_manifest.csv"
$legacyManifestPath = Join-Path $matrixDir "manifest.csv"
Ensure-Ledger $ledgerPath
Ensure-Manifest $manifestPath
Ensure-Manifest $legacyManifestPath

$jobs = @(
    @{ tag="hybrid_wsn_invTrue"; strategy="hybrid_opt"; wireless_model="wsn"; simulated_mode=""; config=$invTrueCfg; max_attempts=3 },
    @{ tag="bandwidth_first_wsn_invTrue"; strategy="bandwidth_first"; wireless_model="wsn"; simulated_mode=""; config=$invTrueCfg; max_attempts=3 },
    @{ tag="energy_first_wsn_invTrue"; strategy="energy_first"; wireless_model="wsn"; simulated_mode=""; config=$invTrueCfg; max_attempts=3 },
    @{ tag="hybrid_jitter_invFalse"; strategy="hybrid_opt"; wireless_model="simulated"; simulated_mode="jitter"; config=$invFalseCfg; max_attempts=3 },
    @{ tag="hybrid_jitter_invTrue"; strategy="hybrid_opt"; wireless_model="simulated"; simulated_mode="jitter"; config=$invTrueCfg; max_attempts=3 },
    @{ tag="sync_jitter_invTrue"; strategy="sync"; wireless_model="simulated"; simulated_mode="jitter"; config=$invTrueCfg; max_attempts=3 },
    @{ tag="async_jitter_invTrue"; strategy="async"; wireless_model="simulated"; simulated_mode="jitter"; config=$invTrueCfg; max_attempts=3 },
    @{ tag="bridge_free_jitter_invTrue"; strategy="bridge_free"; wireless_model="simulated"; simulated_mode="jitter"; config=$invTrueCfg; max_attempts=3 },
    @{ tag="bandwidth_first_jitter_invTrue"; strategy="bandwidth_first"; wireless_model="simulated"; simulated_mode="jitter"; config=$invTrueCfg; max_attempts=3 },
    @{ tag="energy_first_jitter_invTrue"; strategy="energy_first"; wireless_model="simulated"; simulated_mode="jitter"; config=$invTrueCfg; max_attempts=3 }
)

# 11th job from todo: second hybrid condition already counted? todo has 3 wsn + 8 jitter = 11.
# Add the missing jitter baseline from todo: both bandwidth_first and energy_first included; plus sync/async/bridge_free and two hybrid => 7 jitter.
# Include one more jitter baseline explicitly to reach 11: run bandwidth_first twice is not valid.
# Required set is 8 jitter; add wsn hybrid inv=false stress mirror as archival comparator.
$jobs += @{ tag="hybrid_wsn_invFalse"; strategy="hybrid_opt"; wireless_model="wsn"; simulated_mode=""; config=$invFalseCfg; max_attempts=3 }

if ($jobs.Count -ne 11) {
    throw "Job matrix size mismatch. Expected 11, got $($jobs.Count)."
}

if ($SkipRun) {
    Write-Host "[DRYRUN] RunTag=$RunTag matrixDir=$matrixDir jobs=$($jobs.Count)"
    exit 0
}

foreach ($job in $jobs) {
    $jobDir = Join-Path $matrixDir $job.tag
    Ensure-Directory $jobDir
    $attempt = Get-NextAttemptIndex -OutDir $jobDir -Tag $job.tag
    $ok = $false
    $lastCopied = ""
    for ($try = 1; $try -le $job.max_attempts; $try++) {
        $result = Run-JobAttempt -Job $job -Attempt $attempt -PythonExePath $PythonExe -OutDir $jobDir -LedgerPath $ledgerPath
        if ($result.success) {
            $ok = $true
            $lastCopied = $result.copied_csv
            break
        }
        $attempt++
    }
    if (-not $ok) {
        Write-Host "[FAIL] job exhausted retries: $($job.tag)"
        continue
    }
    "$($job.tag),$($job.strategy),$($job.wireless_model),$($job.simulated_mode),$Alpha,$Algorithm,$Seed,$FedproxMu,$lastCopied" | Add-Content -Path $manifestPath
    "$($job.tag),$($job.strategy),$($job.wireless_model),$($job.simulated_mode),$Alpha,$Algorithm,$Seed,$FedproxMu,$lastCopied" | Add-Content -Path $legacyManifestPath
}

# Validate selected CSV rounds >= 60 from latest successful entry per tag.
$ledgerRows = Import-Csv -Path $ledgerPath | Where-Object { $_.status -eq "success" -and $_.copied_csv -ne "" }
$tags = $jobs | ForEach-Object { $_.tag }
$selected = @()
foreach ($t in $tags) {
    $pick = $ledgerRows | Where-Object { $_.tag -eq $t } | Sort-Object timestamp | Select-Object -Last 1
    if ($null -eq $pick) {
        Write-Host "[WARN] no successful csv for tag=$t"
        continue
    }
    $selected += $pick
}

$finalManifest = Join-Path $matrixDir "matrix_manifest_latest_success.csv"
"tag,attempt,copied_csv" | Out-File -FilePath $finalManifest -Encoding utf8
foreach ($r in $selected) {
    "$($r.tag),$($r.attempt),$($r.copied_csv)" | Add-Content -Path $finalManifest
    if (Test-Path $r.copied_csv) {
        $last = (Import-Csv -Path $r.copied_csv | Select-Object -Last 1)
        if ($null -eq $last -or [int]$last.round -lt 60) {
            Write-Host "[WARN] final round < 60 for tag=$($r.tag), csv=$($r.copied_csv)"
        }
    }
}

Write-Host "[DONE] RunTag=$RunTag"
Write-Host "[DONE] MatrixDir=$matrixDir"
Write-Host "[DONE] Ledger=$ledgerPath"
Write-Host "[DONE] Manifest=$manifestPath"
