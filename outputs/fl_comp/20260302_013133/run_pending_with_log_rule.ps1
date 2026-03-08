param([string]$RunTag='20260302_013133')
$ErrorActionPreference='Stop'

$root = "outputs/fl_comp/$RunTag"
$matrixDir = Join-Path $root 'B_matrix_tuned'
$configDir = Join-Path $root 'configs'
$analysisDir = Join-Path $root 'analysis/B_matrix_tuned'
$plotsDir = Join-Path $root 'analysis/plots'

$alpha=0.5; $algo='fedprox'; $mu=0.02; $seed=42; $numRounds=60
$cfgInvFalse = Join-Path $configDir 'hybrid_opt_tuned_inv_false.yaml'
$cfgInvTrue  = Join-Path $configDir 'hybrid_opt_tuned_inv_true.yaml'

$jobs=@(
 @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='hybrid_wsn_invFalse_tuned'},
 @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='wsn'; sim=''; tag='hybrid_wsn_invTrue_tuned'},
 @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='bandwidth_first_wsn_tuned'},
 @{strategy='energy_first'; cfg=$cfgInvFalse; wireless='wsn'; sim=''; tag='energy_first_wsn_tuned'},
 @{strategy='hybrid_opt'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invFalse_tuned'},
 @{strategy='hybrid_opt'; cfg=$cfgInvTrue;  wireless='simulated'; sim='jitter'; tag='hybrid_jitter_invTrue_tuned'},
 @{strategy='sync'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='sync_jitter_tuned'},
 @{strategy='async'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='async_jitter_tuned'},
 @{strategy='bridge_free'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bridge_free_jitter_tuned'},
 @{strategy='bandwidth_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='bandwidth_first_jitter_tuned'},
 @{strategy='energy_first'; cfg=$cfgInvFalse; wireless='simulated'; sim='jitter'; tag='energy_first_jitter_tuned'}
)

function Test-ValidCsv([string]$csvPath,[int]$expectedRounds){
 if(!(Test-Path $csvPath)){return $false}
 try{ $d=Import-Csv $csvPath; if($d.Count -lt 1){return $false}; $r=[int](($d|Select-Object -Last 1).round); return ($r -ge $expectedRounds)}catch{return $false}
}
function LatestCsv([string]$dir){ if(!(Test-Path $dir)){return $null}; $f=Get-ChildItem $dir -Filter *.csv | Sort-Object LastWriteTime -Descending; if($f.Count -eq 0){return $null}; return $f[0] }
function LogMetricsCsv([string]$logPath){
 if(!(Test-Path $logPath)){return $null}
 $m=Select-String -Path $logPath -Pattern 'Metrics CSV:\s*(.+\.csv)' | Select-Object -Last 1
 if($null -eq $m){ return $null }
 return $m.Matches[0].Groups[1].Value
}

$metricDir='outputs/hybrid_metrics/hybrid_opt'

foreach($j in $jobs){
 $outDir=Join-Path $matrixDir $j.tag
 New-Item -ItemType Directory -Force -Path $outDir | Out-Null
 $existing=LatestCsv $outDir
 if($null -ne $existing -and (Test-ValidCsv $existing.FullName $numRounds)){
   Write-Host "[SKIP] $($j.tag) valid csv exists: $($existing.Name)"
   continue
 }

 $ok=$false
 for($attempt=1;$attempt -le 3;$attempt++){
   Write-Host "[RUN] $($j.tag) attempt $attempt/3"
   Get-Process | Where-Object { $_.ProcessName -eq 'python' } | Stop-Process -Force -ErrorAction SilentlyContinue

   if(Test-Path $metricDir){ Get-ChildItem $metricDir -Filter *.csv -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue }

   $log=Join-Path $outDir ("run_"+$j.tag+"_attempt"+$attempt+".log")
   $err=Join-Path $outDir ("run_"+$j.tag+"_attempt"+$attempt+".err")
   if(Test-Path $log){Remove-Item -Force $log -ErrorAction SilentlyContinue}
   if(Test-Path $err){Remove-Item -Force $err -ErrorAction SilentlyContinue}

   $args=@('-u','.\\src\\flower\\hybrid_opt_demo.py','--strategy',$j.strategy,'--config',$j.cfg,'--num-rounds',"$numRounds",'--alpha',"$alpha",'--algorithm',$algo,'--fedprox-mu',"$mu",'--seed',"$seed",'--wireless-model',$j.wireless)
   if($j.wireless -eq 'simulated' -and -not [string]::IsNullOrWhiteSpace($j.sim)){ $args += @('--simulated-mode',$j.sim) }

   $p=Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
   Wait-Process -Id $p.Id

   if($p.ExitCode -ne 0){ Write-Host "[WARN] non-zero exit code: $($p.ExitCode)"; continue }

   $metricCsvRel = LogMetricsCsv $log
   if([string]::IsNullOrWhiteSpace($metricCsvRel)){ Write-Host '[WARN] no Metrics CSV line at log end'; continue }

   $metricCsv = $metricCsvRel -replace '\\','/'
   if(!(Test-Path $metricCsv)){ Write-Host "[WARN] metrics csv missing: $metricCsvRel"; continue }

   $dst=Join-Path $outDir ("$($j.tag)_" + [IO.Path]::GetFileName($metricCsv))
   Copy-Item -Force -Path $metricCsv -Destination $dst

   if(-not (Test-ValidCsv $dst $numRounds)){ Write-Host "[WARN] copied csv invalid: $dst"; continue }

   $ok=$true
   break
 }
 if(-not $ok){ throw "job failed by log-rule after retries: $($j.tag)" }
}

$manifest = Join-Path $matrixDir 'matrix_manifest.csv'
"strategy,wireless,sim_mode,alpha,algorithm,seed,fedprox_mu,csv_path" | Out-File -FilePath $manifest -Encoding utf8
foreach($j in $jobs){
 $dir=Join-Path $matrixDir $j.tag
 $csv=LatestCsv $dir
 if($null -eq $csv){ throw "missing csv for $($j.tag)" }
 if(-not (Test-ValidCsv $csv.FullName $numRounds)){ throw "invalid csv for $($j.tag)" }
 "$($j.strategy),$($j.wireless),$($j.sim),$alpha,$algo,$seed,$mu,$($csv.FullName)" | Add-Content -Path $manifest
}

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
New-Item -ItemType Directory -Force -Path $plotsDir | Out-Null
.\.venv\Scripts\python.exe analyze_results.py --run-dir $matrixDir --out-dir $analysisDir
.\.venv\Scripts\python.exe generate_comparison_plots.py --metrics-root $matrixDir --out-dir $plotsDir
Write-Host "LOG_RULE_DONE $RunTag"
