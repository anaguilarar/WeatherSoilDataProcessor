#!/usr/bin/env bash
# Maize validation on the server: cultivar screening, local maps vs SoilGrids, NITRO sensitivity.
# Run from the repository root, e.g.  nohup bash scripts/run_maize_validation.sh > validation_maize.log 2>&1 &
set -euo pipefail

# ---- paths (edit) -------------------------------------------------------------------------------
DATA=/path/to/validation/data.csv                       # observed yields
LOCAL=/path/to/village_sp_files                         # local soil maps (+ the weather used by both runs)
SG=/path/to/village_sp_files_soilgrids                  # SoilGrids soils
BIN=/path/to/dssat/dscsm048                             # DSSAT executable
DSSAT=/path/to/dssat                                    # DSSAT folder
NCORES=16
OUT=validation_maize
TOP=3                                                   # cultivars carried to the final runs

COMMON=(--data "$DATA" --bin-path "$BIN" --dssat-path "$DSSAT" --ncores "$NCORES")
LOCAL_SOIL=(--villages-dir "$LOCAL")
SG_SOIL=(--villages-dir "$SG" --weather-dir "$LOCAL")    # SoilGrids soil, local weather and dem

# ---- 1. cultivar screening: 10 cultivars, 15 villages per unit drawn by maize area, NITRO = Y -----
python scripts/validate_maize_cultivars.py "${COMMON[@]}" "${LOCAL_SOIL[@]}" --nitro Y \
    --village-selection pps --max-villages-per-unit 15 --out $OUT/screening_local --work-dir runs_scr_l
python scripts/validate_maize_cultivars.py "${COMMON[@]}" "${SG_SOIL[@]}" --nitro Y \
    --village-selection pps --max-villages-per-unit 15 --out $OUT/screening_soilgrids --work-dir runs_scr_s

# best cultivars: lowest mean RMSE on the department census over both soil sources
CULTIVARS=$(python scripts/maize_validation_steps.py select $OUT $TOP)
echo "cultivars for the final runs: $CULTIVARS"

# ---- 2. final runs: top cultivars, villages covering 90% of the maize area, NITRO = Y -------------
python scripts/validate_maize_cultivars.py "${COMMON[@]}" "${LOCAL_SOIL[@]}" --nitro Y \
    --cultivars "$CULTIVARS" --out $OUT/local_nitroY --work-dir runs_fin_l
python scripts/validate_maize_cultivars.py "${COMMON[@]}" "${SG_SOIL[@]}" --nitro Y \
    --cultivars "$CULTIVARS" --out $OUT/soilgrids_nitroY --work-dir runs_fin_s

# ---- 3. local maps vs SoilGrids, one comparison per cultivar ------------------------------------
for code in $(echo "$CULTIVARS" | tr ',' '\n' | cut -d: -f1); do
    python scripts/compare_soil_sources.py --a $OUT/local_nitroY --b $OUT/soilgrids_nitroY \
        --labels local,soilgrids --cultivar "$code" --out $OUT/comparison_nitroY/$code
done

# ---- 4. sensitivity: the same cultivars without N stress (NITRO = N) -----------------------------
python scripts/validate_maize_cultivars.py "${COMMON[@]}" "${LOCAL_SOIL[@]}" --nitro N \
    --cultivars "$CULTIVARS" --out $OUT/local_nitroN --work-dir runs_sen_l
python scripts/validate_maize_cultivars.py "${COMMON[@]}" "${SG_SOIL[@]}" --nitro N \
    --cultivars "$CULTIVARS" --out $OUT/soilgrids_nitroN --work-dir runs_sen_s
for code in $(echo "$CULTIVARS" | tr ',' '\n' | cut -d: -f1); do
    python scripts/compare_soil_sources.py --a $OUT/local_nitroN --b $OUT/soilgrids_nitroN \
        --labels local,soilgrids --cultivar "$code" --out $OUT/comparison_nitroN/$code
done

# ---- summary: delta RMSE (SoilGrids - local) per cultivar and NITRO -------------------------------
python scripts/maize_validation_steps.py summary $OUT
