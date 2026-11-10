# Bean validation on the server: local maps vs SoilGrids, NITRO sensitivity (same approach as maize).
# Three cultivars (IB0006 ICTA-Ostua, 990005 short, 990006 long), so there is no screening step: every run
# simulates all three on the villages covering 90% of the rainfed bean area (MapSPAM 2020).
# Run from the repository root, in the PowerShell window where the conda environment is active:
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\scripts\run_bean_validation.ps1 *> validation_bean.log
# Run it again after a stop: villages and cultivars already in the caches are not simulated again.

# ---- paths (edit) -------------------------------------------------------------------------------
$DATA   = 'E:/Worspace2024/aaguilar/spatial_data/vector/latam/hnd_validation_yield_data.csv'
$LOCAL  = 'E:/Worspace2024/aaguilar/spatial_data/raster/village_sp_files'             # local soils + weather
$SG     = 'E:/Worspace2024/aaguilar/spatial_data/raster/village_sp_files_soilgrids'   # SoilGrids soils
$BIN    = 'C:/DSSAT48/DSCSM048.EXE'
$DSSAT  = 'C:/DSSAT48'
$NCORES = 20
$OUT    = 'validation_bean'
$CULTIVARS = 'IB0006:icta-ostua,990005:short,990006:long'
$PY     = 'python'                                    # or the full path to the environment's python.exe

$ErrorActionPreference = 'Continue'   # python warnings on stderr must not stop the run; exit codes are checked
$COMMON = @('--crop', 'bean', '--data', $DATA, '--bin-path', $BIN, '--dssat-path', $DSSAT,
            '--cultivars', $CULTIVARS)
$LOCAL_SOIL = @('--villages-dir', $LOCAL)
$SG_SOIL = @('--villages-dir', $SG, '--weather-dir', $LOCAL)   # SoilGrids soil, local weather and dem

Write-Host "python: $((Get-Command $PY).Source)"
& $PY -c "import pandas, xarray"
if ($LASTEXITCODE -ne 0) { throw "$PY cannot import pandas/xarray: activate the conda environment or set `$PY" }
foreach ($d in @($LOCAL, $SG)) { if (-not (Test-Path $d)) { throw "folder not found: $d" } }
if (-not (Test-Path $DATA)) { throw "file not found: $DATA" }
if (-not (Test-Path 'data/bean_area_village_spam2020.csv')) {
    Write-Host 'data/bean_area_village_spam2020.csv missing: it will be computed from MapSPAM (needs internet)'
}

function Run-Py([string[]]$arguments) {
    Write-Host "`n>>> python $($arguments -join ' ')"
    & $PY @arguments
    if ($LASTEXITCODE -ne 0) { throw "python failed (exit $LASTEXITCODE): $($arguments -join ' ')" }
}

# PowerShell names are case-insensitive: a parameter called $out would hide $OUT
# Second pass: villages that failed (e.g. workers colliding at start-up) are tried once more on fewer cores.
function Validate([string[]]$soil, [string]$nitro, [string]$run, [string]$work) {
    $base = @('scripts/validate_maize_cultivars.py') + $COMMON + $soil + @('--nitro', $nitro, '--out', "$OUT/$run")
    Run-Py ($base + @('--ncores', $NCORES, '--work-dir', $work))
    Run-Py ($base + @('--ncores', 4, '--work-dir', "${work}_r", '--retry-failed'))
}

function Compare-SoilSources([string]$nitro) {
    foreach ($code in ($CULTIVARS -split ',' | ForEach-Object { ($_ -split ':')[0] })) {
        Run-Py @('scripts/compare_soil_sources.py', '--a', "$OUT/local_nitro$nitro", '--b', "$OUT/soilgrids_nitro$nitro",
                 '--labels', 'local,soilgrids', '--cultivar', $code, '--out', "$OUT/comparison_nitro$nitro/$code")
    }
}

# ---- 1. NITRO = Y: local maps and SoilGrids ---------------------------------------------------------
Validate $LOCAL_SOIL 'Y' 'local_nitroY' 'runs_bn_l'
Validate $SG_SOIL 'Y' 'soilgrids_nitroY' 'runs_bn_s'
Compare-SoilSources 'Y'

# ---- 2. sensitivity: no N stress (NITRO = N) ---------------------------------------------------------
Validate $LOCAL_SOIL 'N' 'local_nitroN' 'runs_bn_ln'
Validate $SG_SOIL 'N' 'soilgrids_nitroN' 'runs_bn_sn'
Compare-SoilSources 'N'

# ---- summary: delta RMSE (SoilGrids - local) per cultivar and NITRO -------------------------------
Run-Py @('scripts/maize_validation_steps.py', 'summary', $OUT)
