param([string]$RunTag='20260302_013133')
$ErrorActionPreference='Stop'

$root = "outputs/fl_comp/$RunTag"
$matrixDir = Join-Path $root 'B_matrix_tuned'
$configDir = Join-Path $root 'configs'
$analysisDir = Join-Path $root 'analysis/B_matrix_tuned'
$plotsDir = Join-Path $root 'analysis/plots'
$ledger = Join-Path $matrixDir 'attempt_ledger.csv'

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
function NextAttemptNo([string]$outDir,[string]$tag){
 $max=0
 if(Test-Path $outDir){
   $logs=Get-ChildItem $outDir -Filter "run_${tag}_attempt*.log" -ErrorAction SilentlyContinue
   foreach($f in $logs){
     if($f.Name -match "attempt(\d+)\.log$"){
       $n=[int]$Matches[1]; if($n -gt $max){$max=$n}
     }
   }
 }
 return ($max+1)
}
function AppendLedger([string]$timestamp,[string]$tag,[int]$attempt,[string]$status,[string]$log,[string]$err,[string]$metrics,[string]$copied,[string]$note){
 "$timestamp,$tag,$attempt,$status,$log,$err,$metrics,$copied,$note" | Add-Content -Path $ledger
}

if(!(Test-Path $ledger)){
 "timestamp,tag,attempt,status,log,err,metrics_csv,copied_csv,note" | Out-File -FilePath $ledger -Encoding utf8
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
 for($retry=1;$retry -le 3;$retry++){
   $attempt=NextAttemptNo $outDir $j.tag
   $log=Join-Path $outDir ("run_"+$j.tag+"_attempt"+$attempt+".log")
   $err=Join-Path $outDir ("run_"+$j.tag+"_attempt"+$attempt+".err")

   Write-Host "[RUN] $($j.tag) retry $retry/3 -> attempt $attempt"

   Get-Process | Where-Object { $_.ProcessName -eq 'python' } | Stop-Process -Force -ErrorAction SilentlyContinue

   $args=@('-u','.\\src\\flower\\hybrid_opt_demo.py','--strategy',$j.strategy,'--config',$j.cfg,'--num-rounds',"$numRounds",'--alpha',"$alpha",'--algorithm',$algo,'--fedprox-mu',"$mu",'--seed',"$seed",'--wireless-model',$j.wireless)
   if($j.wireless -eq 'simulated' -and -not [string]::IsNullOrWhiteSpace($j.sim)){ $args += @('--simulated-mode',$j.sim) }

   $p=Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
   Wait-Process -Id $p.Id

   $ts=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
   if($p.ExitCode -ne 0){
     AppendLedger $ts $j.tag $attempt 'failed' $log $err '' '' "exit_code=$($p.ExitCode)"
     continue
   }

   $metricsRel = LogMetricsCsv $log
   if([string]::IsNullOrWhiteSpace($metricsRel)){
     AppendLedger $ts $j.tag $attempt 'failed' $log $err '' '' 'missing_metrics_csv_line'
     continue
   }

   $metricsPath = $metricsRel -replace '\\','/'
   if(!(Test-Path $metricsPath)){
     AppendLedger $ts $j.tag $attempt 'failed' $log $err $metricsRel '' 'metrics_csv_not_found'
     continue
   }

   $baseName=[IO.Path]::GetFileName($metricsPath)
   $dst=Join-Path $outDir ("$($j.tag)_attempt$attempt" + "_" + $baseName)
   Copy-Item -Path $metricsPath -Destination $dst

   if(-not (Test-ValidCsv $dst $numRounds)){
     AppendLedger $ts $j.tag $attempt 'failed' $log $err $metricsRel $dst 'copied_csv_invalid_rounds'
     continue
   }

   AppendLedger $ts $j.tag $attempt 'success' $log $err $metricsRel $dst 'ok'
   $ok=$true
   break
 }
 if(-not $ok){ throw "job failed after retries: $($j.tag)" }
}

$manifest = Join-Path $matrixDir 'matrix_manifest.csv'
"strategy,wireless,sim_mode,alpha,algorithm,seed,fedprox_mu,csv_path" | Out-File -FilePath $manifest -Encoding utf8
foreach($j in $jobs){
 $pickedPath = $null
 if(Test-Path $ledger){
   $rows = Import-Csv $ledger | Where-Object { $_.tag -eq $j.tag -and $_.status -eq 'success' -and -not [string]::IsNullOrWhiteSpace($_.copied_csv) }
   if($rows){
     $latest = $rows | Sort-Object { [datetime]::ParseExact($_.timestamp, 'yyyy-MM-dd HH:mm:ss', $null) } -Descending | Select-Object -First 1
     if($null -ne $latest -and (Test-Path $latest.copied_csv)){ $pickedPath = $latest.copied_csv }
   }
 }
 if([string]::IsNullOrWhiteSpace($pickedPath)){
   $dir=Join-Path $matrixDir $j.tag
   $latestCsv = Get-ChildItem $dir -Filter *.csv -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
   if($null -eq $latestCsv){ throw "no csv for $($j.tag)" }
   $pickedPath = $latestCsv.FullName
 }
 if(-not (Test-ValidCsv $pickedPath $numRounds)){ throw "latest picked csv invalid for $($j.tag): $pickedPath" }
 "$($j.strategy),$($j.wireless),$($j.sim),$alpha,$algo,$seed,$mu,$pickedPath" | Add-Content -Path $manifest
}

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
New-Item -ItemType Directory -Force -Path $plotsDir | Out-Null
.\.venv\Scripts\python.exe analyze_results.py --run-dir $matrixDir --out-dir $analysisDir
.\.venv\Scripts\python.exe generate_comparison_plots.py --metrics-root $matrixDir --out-dir $plotsDir
Write-Host "PRESERVE_MODE_DONE $RunTag"
