param(
    [string]$Strategy,
    [string]$MetricsStrategyName = "",
    [string]$ConfigPath,
    [string]$WirelessModel,
    [string]$SimulatedMode,
    [int]$NumRounds,
    [double]$Alpha,
    [string]$Algorithm,
    [double]$FedproxMu = 0.0,
    [int]$Seed,
    [string]$OutDir,
    [string]$Tag
)

$ErrorActionPreference = "Stop"

function Get-LatestCsv([string]$dir) {
    if (!(Test-Path $dir)) { return $null }
    $files = Get-ChildItem -Path $dir -Filter *.csv | Sort-Object LastWriteTime -Descending
    if ($files.Count -eq 0) { return $null }
    return $files[0].FullName
}

function Has-Error([string]$errPath) {
    if (!(Test-Path $errPath)) { return $false }
    $content = Get-Content -Path $errPath -Raw
    if ($content -match "Traceback") { return $true }
    if ($content -match "Simulation crashed") { return $true }
    if ($content -match "ERROR\\s*:") { return $true }
    return $false
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$args = @("-u", ".\\src\\flower\\hybrid_opt_demo.py", "--strategy", $Strategy, "--num-rounds", $NumRounds, "--alpha", $Alpha, "--algorithm", $Algorithm, "--seed", $Seed)
if ($Algorithm -eq "fedprox" -and $FedproxMu -gt 0) { $args += @("--fedprox-mu", $FedproxMu) }
if ($ConfigPath -and (Test-Path $ConfigPath)) { $args += @("--config", $ConfigPath) }
if ($WirelessModel) { $args += @("--wireless-model", $WirelessModel) }
if ($WirelessModel -eq "simulated" -and $SimulatedMode) { $args += @("--simulated-mode", $SimulatedMode) }

for ($attempt = 1; $attempt -le 2; $attempt++) {
    Write-Host "Attempt ${attempt}: $Strategy | $WirelessModel | $SimulatedMode | alpha=$Alpha | algo=$Algorithm | seed=$Seed"
    $log = Join-Path $OutDir ("run_" + $Tag + "_attempt" + $attempt + ".log")
    $err = Join-Path $OutDir ("run_" + $Tag + "_attempt" + $attempt + ".err")
    if (Test-Path $log) { Remove-Item -Force $log -ErrorAction SilentlyContinue }
    if (Test-Path $err) { Remove-Item -Force $err -ErrorAction SilentlyContinue }
    $p = Start-Process -FilePath ".\\.venv\\Scripts\\python.exe" -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $lastWrite = Get-Date
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 10
        if (Test-Path $log) {
            $lw = (Get-Item $log).LastWriteTime
            if ($lw -gt $lastWrite) { $lastWrite = $lw }
        }
        if (Test-Path $err) {
            $ew = (Get-Item $err).LastWriteTime
            if ($ew -gt $lastWrite) { $lastWrite = $ew }
        }
        if ((Get-Date) - $lastWrite -gt (New-TimeSpan -Minutes 2)) {
            Write-Host "No output in 2 minutes, stopping and retrying..."
            try { $p.Kill() } catch {}
            try { Wait-Process -Id $p.Id -Timeout 30 } catch {}
            break
        }
    }
    $metricsName = $MetricsStrategyName
    if ([string]::IsNullOrWhiteSpace($metricsName)) { $metricsName = $Strategy }
    if ($p.HasExited -and $p.ExitCode -eq 0) {
        $errFailed = Has-Error $err
        $srcDir = "outputs/hybrid_metrics/$metricsName"
        $latest = Get-LatestCsv $srcDir
        if (($null -ne $latest) -and (-not $errFailed)) {
            $dst = Join-Path $OutDir ("${Tag}_" + (Split-Path $latest -Leaf))
            Move-Item -Force $latest $dst
            exit 0
        }
    }
    if ($p.HasExited) {
        $errFailed = Has-Error $err
        $srcDir = "outputs/hybrid_metrics/$metricsName"
        $latest = Get-LatestCsv $srcDir
        if (($null -ne $latest) -and (-not $errFailed)) {
            $dst = Join-Path $OutDir ("${Tag}_" + (Split-Path $latest -Leaf))
            Move-Item -Force $latest $dst
            exit 0
        }
    }
}

exit 1
