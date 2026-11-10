# Maize validation on the server: cultivar screening, local maps vs SoilGrids, NITRO sensitivity.
# Run from the repository root, in the PowerShell window where the conda environment is active
# (a new 'powershell' process does not see the environment's python):
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\run_maize_validation.ps1 *> validation_maize.log
# Run it again after a stop: villages and cultivars already in the caches are not simulated again.

# ---- paths (edit) -------------------------------------------------------------------------------
$DATA   = 'E:/path/to/validation/data.csv'            # observed yields
$LOCAL  = 'E:/path/to/village_sp_files'               # local soil maps (+ the weather used by both runs)
$SG     = 'E:/path/to/village_sp_files_soilgrids'     # SoilGrids soils
$BIN    = 'C:/DSSAT48/DSCSM048.EXE'                   # DSSAT executable
$DSSAT  = 'C:/DSSAT48'                                # DSSAT folder
$NCORES = 16
$OUT    = 'validation_maize'
$TOP    = 3                                           # cultivars carried to the final runs
$PY     = 'python'                                    # or the full path to the environment's python.exe

$ErrorActionPreference = 'Continue'   # python warnings on stderr must not stop the run; exit codes are checked
$COMMON = @('--data', $DATA, '--bin-path', $BIN, '--dssat-path', $DSSAT, '--ncores', $NCORES)
$LOCAL_SOIL = @('--villages-dir', $LOCAL)
$SG_SOIL = @('--villages-dir', $SG, '--weather-dir', $LOCAL)   # SoilGrids soil, local weather and dem

Write-Host "python: $((Get-Command $PY).Source)"
& $PY -c "import pandas, xarray"
if ($LASTEXITCODE -ne 0) { throw "$PY cannot import pandas/xarray: activate the conda environment or set `$PY" }
foreach ($d in @($LOCAL, $SG)) { if (-not (Test-Path $d)) { throw "folder not found: $d" } }
if (-not (Test-Path $DATA)) { throw "file not found: $DATA" }

function Run-Py([string[]]$arguments) {
    Write-Host "`n>>> python $($arguments -join ' ')"
    & $PY @arguments
    if ($LASTEXITCODE -ne 0) { throw "python failed (exit $LASTEXITCODE): $($arguments -join ' ')" }
}

# PowerShell names are case-insensitive: a parameter called $out would hide $OUT
function Validate([string[]]$soil, [string]$nitro, [string]$run, [string]$work, [string[]]$extra) {
    Run-Py (@('scripts/validate_maize_cultivars.py') + $COMMON + $soil +
            @('--nitro', $nitro, '--out', "$OUT/$run", '--work-dir', $work) + $extra)
}

function Compare-SoilSources([string]$nitro, [string]$cultivars) {
    foreach ($code in ($cultivars -split ',' | ForEach-Object { ($_ -split ':')[0] })) {
        Run-Py @('scripts/compare_soil_sources.py', '--a', "$OUT/local_nitro$nitro", '--b', "$OUT/soilgrids_nitro$nitro",
                 '--labels', 'local,soilgrids', '--cultivar', $code, '--out', "$OUT/comparison_nitro$nitro/$code")
    }
}

# ---- 1. cultivar screening: 10 cultivars, 15 villages per unit drawn by maize area, NITRO = Y -----
$PPS = @('--village-selection', 'pps', '--max-villages-per-unit', '15')
Validate $LOCAL_SOIL 'Y' 'screening_local' 'runs_scr_l' $PPS
Validate $SG_SOIL 'Y' 'screening_soilgrids' 'runs_scr_s' $PPS

# best cultivars: lowest mean RMSE on the department census over both soil sources
$CULTIVARS = (& $PY scripts/maize_validation_steps.py select $OUT $TOP | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $CULTIVARS) { throw 'cultivar selection failed' }
Write-Host "cultivars for the final runs: $CULTIVARS"

# ---- 2. final runs: top cultivars, villages covering 90% of the maize area, NITRO = Y -------------
Validate $LOCAL_SOIL 'Y' 'local_nitroY' 'runs_fin_l' @('--cultivars', $CULTIVARS)
Validate $SG_SOIL 'Y' 'soilgrids_nitroY' 'runs_fin_s' @('--cultivars', $CULTIVARS)

# ---- 3. local maps vs SoilGrids, one comparison per cultivar ------------------------------------
Compare-SoilSources 'Y' $CULTIVARS

# ---- 4. sensitivity: the same cultivars without N stress (NITRO = N) -----------------------------
Validate $LOCAL_SOIL 'N' 'local_nitroN' 'runs_sen_l' @('--cultivars', $CULTIVARS)
Validate $SG_SOIL 'N' 'soilgrids_nitroN' 'runs_sen_s' @('--cultivars', $CULTIVARS)
Compare-SoilSources 'N' $CULTIVARS

# ---- summary: delta RMSE (SoilGrids - local) per cultivar and NITRO -------------------------------
Run-Py @('scripts/maize_validation_steps.py', 'summary', $OUT)
