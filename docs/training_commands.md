$runTag = "20260228_011223"; $outDir = "outputs/fl_comp/$runTag/A_algo_select"; & .\run_one.ps1 -Strategy "hybrid_opt" -ConfigPath "outputs/fl_comp/$runTag/configs/hybrid_opt_inv_false.yaml" -WirelessModel "wsn" -SimulatedMode "" -NumRounds 100 -Alpha 0.5 -Algorithm "fedavg" -Seed 42 -OutDir $outDir -Tag "wsn_fedavg_invFalse_seed42"; if ($LASTEXITCODE -ne 0) { throw "run failed" }

$runTag = "20260228_011223"; $outDir = "outputs/fl_comp/$runTag/A_algo_select"; & .\run_one.ps1 -Strategy "hybrid_opt" -ConfigPath "outputs/fl_comp/$runTag/configs/hybrid_opt_inv_false.yaml" -WirelessModel "wsn" -SimulatedMode "" -NumRounds 100 -Alpha 0.5 -Algorithm "fedprox" -Seed 42 -OutDir $outDir -Tag "wsn_fedprox_invFalse_seed42"; if ($LASTEXITCODE -ne 0) { throw "run failed" }

$runTag = "20260228_033709"; $muDir = "outputs/fl_comp/$runTag/A_mu_sweep"; New-Item -ItemType Directory -Force -Path $muDir | Out-Null; $mus = @(0.001, 0.01, 0.05); foreach ($mu in $mus) { $tag = "wsn_fedprox_mu${mu}_invFalse" -replace '\.','p'; & .\run_one.ps1 -Strategy "hybrid_opt" -ConfigPath "outputs/fl_comp/$runTag/configs/hybrid_opt_inv_false.yaml" -WirelessModel "wsn" -SimulatedMode "" -NumRounds 60 -Alpha 0.5 -Algorithm "fedprox" -FedproxMu $mu -Seed 42 -OutDir (Join-Path $muDir $tag) -Tag $tag; if ($LASTEXITCODE -ne 0) { throw "run failed: $mu" } }

$runTag = "20260228_033709"; $outRoot = "outputs/fl_comp/$runTag"; $matrixDir = "$outRoot/B_matrix_mu0p01"; $algo = "fedprox"; $mu = 0.01; $seed = 42; $alpha = 0.5; $jobs = @(
    @{strategy='hybrid_opt'; cfg="$outRoot/configs/hybrid_opt_inv_false.yaml"; wireless='wsn'; sim='"''"'; tag='hybrid_wsn_invFalse'},
    @{strategy='hybrid_opt'; cfg="$outRoot/configs/hybrid_opt_inv_true.yaml"; wireless='wsn'; sim='"''"'; tag='hybrid_wsn_invTrue'},
    @{strategy='sync'; cfg=""; wireless='wsn'; sim='"''"'; tag='sync_wsn'},
    @{strategy='async'; cfg=""; wireless='wsn'; sim='"''"'; tag='async_wsn'},
    @{strategy='bridge_free'; cfg=""; wireless='wsn'; sim='"''"'; tag='bridgefree_wsn'},
    @{strategy='bandwidth_first'; cfg=""; wireless='wsn'; sim='"''"'; tag='bwfirst_wsn'},
    @{strategy='energy_first'; cfg=""; wireless='wsn'; sim='"''"'; tag='energyfirst_wsn'}
);
foreach ($j in $jobs) {
    $outDir = Join-Path $matrixDir $j.tag; New-Item -ItemType Directory -Force -Path $outDir | Out-Null;
    & .\run_one.ps1 -Strategy $j.strategy -ConfigPath $j.cfg -WirelessModel $j.wireless -SimulatedMode $j.sim -NumRounds 60 -Alpha $alpha -Algorithm $algo -FedproxMu $mu -Seed $seed -OutDir $outDir -Tag $j.tag;
    if ($LASTEXITCODE -ne 0) { throw "run failed: $($j.tag)" }
    $csv = Get-ChildItem -Path $outDir -Filter *.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1;
    if ($null -ne $csv) { "$($j.strategy),$($j.wireless),$($j.sim),$alpha,$algo,$seed,$mu,$($csv.FullName)" | Add-Content -Path (Join-Path $matrixDir "matrix_manifest.csv") }
}

$runTag = "20260228_033709"; $outRoot = "outputs/fl_comp/$runTag"; $matrixDir = "$outRoot/B_matrix_mu0p01"; $algo = "fedprox"; $mu = 0.01; $seed = 42; $alpha = 0.5; $cfg = "$outRoot/configs/hybrid_opt_inv_false.yaml"; $jobs = @('sync','async','bridge_free','bandwidth_first','energy_first'); foreach ($s in $jobs) { $tag = "${s}_wsn"; $outDir = Join-Path $matrixDir $tag; New-Item -ItemType Directory -Force -Path $outDir | Out-Null; & .\run_one.ps1 -Strategy $s -ConfigPath $cfg -WirelessModel "wsn" -SimulatedMode "" -NumRounds 60 -Alpha $alpha -Algorithm $algo -FedproxMu $mu -Seed $seed -OutDir $outDir -Tag $tag; if ($LASTEXITCODE -ne 0) { throw "run failed: $tag" } $csv = Get-ChildItem -Path $outDir -Filter *.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($null -ne $csv) { "$s,wsn,,$alpha,$algo,$seed,$mu,$($csv.FullName)" | Add-Content -Path (Join-Path $matrixDir "matrix_manifest.csv") } }

$runTag = "20260228_033709"; $matrixDir = "outputs/fl_comp/$runTag/B_matrix_mu0p01"; $algo = "fedprox"; $mu = 0.01; $seed = 42; $alpha = 0.5; $cfg = "outputs/fl_comp/$runTag/configs/hybrid_opt_inv_false.yaml"; $pending = @('bandwidth_first','energy_first'); foreach ($s in $pending) { $tag = "${s}_wsn"; $outDir = Join-Path $matrixDir $tag; New-Item -ItemType Directory -Force -Path $outDir | Out-Null; & .\run_one.ps1 -Strategy $s -MetricsStrategyName "hybrid_opt" -ConfigPath $cfg -WirelessModel "wsn" -SimulatedMode "" -NumRounds 60 -Alpha $alpha -Algorithm $algo -FedproxMu $mu -Seed $seed -OutDir $outDir -Tag $tag; if ($LASTEXITCODE -ne 0) { throw "run failed: $tag" } $csv = Get-ChildItem -Path $outDir -Filter *.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($null -ne $csv) { "$s,wsn,,$alpha,$algo,$seed,$mu,$($csv.FullName)" | Add-Content -Path (Join-Path $matrixDir "matrix_manifest.csv") } }
