"""P and K yield response (QUEFTS) of a village from its DSSAT run, per texture class and planting date.

DSSAT gives, for every year, the attainable yield, the biomass, the season length and the season
temperature; PyQUEFTS (unchanged) turns them into the nutrient-limited yield for a grid of fertiliser
rates. Reads a run folder of modeladocultivo_.py (runs/<GEOCODIGO>):

  yield_data_<texture>.csv      mean yield per planting date; only the dates above `percentile` are used
  <texture>/Summary_<TRNO>.OUT  one row per year: HWAH, CWAM, HWAM, PDAT, MDAT, TMAXA, TMINA
  <texture>/experimental_file_config.yaml  MANAGEMENT.fertilizer_npk: the N, P, K (kg/ha of element) of each
                                application of the DSSAT run, null when nothing was applied
  _texture.tif                  texture class of every soil pixel (TEXTURE_CLASSES codes)
  soil_.nc                      soil cube in SoilGrids conventions plus p (Mehlich-3, mg/kg) and k (cmol(+)/kg)

Method
  attainable yield   HWAH (kg/ha dry matter); vegetative biomass CWAM - HWAM, half leaf and half stem
                     (QUEFTS uses the same nutrient concentrations for both, the split does not change yield)
  season length      MDAT - PDAT (days); season temperature (TMAXA + TMINA) / 2
  soil supply        pyquefts.nut_supply2 (Sattari et al. 2014) from the 0-20 cm pH, organic C, exchangeable K
                     and Olsen P (Mehlich-3 converted, McDowell et al. 2023) of every pixel of the texture class
  fertiliser         the season total of fertilizer_npk (the management of the DSSAT run) plus the extra rate;
                     DSSAT already simulated that N (NITRO Y) but not P and K (PHOSP N, POTAS N)
  yield              QUEFTS runs for pixel x year x N x P x K; yields are averaged over the pixels and then
                     summarised over the years

Output: <run>/quefts/quefts_response_<crop>_<texture>_trno<NN>.csv, one row per extra N x P x K rate
(kg/ha of element, plus P2O5 and K2O) applied on top of the DSSAT management (n/p/k_base_kg_ha), with the
mean and the percentiles of the yield over the years, and the gain over the DSSAT management alone (extra P = K = 0
at the same N), computed year by year (kg/ha and % of the yield with the management) and then summarised.
The extra N is 0 by default: the attainable yield of DSSAT already depends on the N of the run, so other N rates
would not be consistent with it (and for bean QUEFTS ignores N fertiliser, N-fixing crop).
quefts_index_<crop>.json lists the files (texture, TRNO, planting day) and the rates, for the web plots.

PyQUEFTS is imported when the response is computed, so this module can be imported without it.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml

QUEFTS_CROP = {'bean': 'Field bean', 'maize': 'Maize'}
TOPSOIL_WEIGHTS = {'0-5': 5, '5-15': 10, '15-30': 5}   # cm of each layer inside 0-20 cm
LEAF_SHARE = 0.5
P_TO_P2O5, K_TO_K2O = 2.2914, 1.2046
PERCENTILES = (10, 25, 50, 75, 90)
GAIN_PERCENTILES = (25, 50, 75)
DEFAULT_RATES = {'n': '0', 'p': '0:120:20', 'k': '0:120:20'}


def parse_rates(rates):
    """'0:120:20' (first:last:step), '0,40,80' or a list of numbers as an array of rates."""
    if not isinstance(rates, str):
        return np.atleast_1d(np.asarray(rates, dtype=float))
    if ':' in rates:
        first, last, step = (float(v) for v in rates.split(':'))
        return np.arange(first, last + step / 2, step)
    return np.array([float(v) for v in rates.split(',')])


def read_summary(fn):
    """DSSAT Summary.OUT as a table."""
    with open(fn) as f:
        lines = [line for line in f if line.strip()]
    start = next(i for i, line in enumerate(lines) if line.startswith('@'))
    columns = lines[start].split()[1:]
    rows = [line.split() for line in lines[start + 1:]]
    assert all(len(r) == len(columns) for r in rows), f'{fn}: columns do not match the header'
    return pd.DataFrame(rows, columns=columns).apply(pd.to_numeric, errors='coerce')


def season_runs(run_dir, texture, trno):
    """Attainable yield, biomass, season length and temperature of every year of a planting date."""
    d = read_summary(os.path.join(run_dir, texture, f'Summary_{trno}.OUT'))
    doy = lambda c: pd.to_datetime(d[c].astype('Int64').astype(str), format='%Y%j', errors='coerce')
    out = pd.DataFrame({
        'year': doy('PDAT').dt.year,
        'store_att': d['HWAH'].astype(float),
        'veg_att': (d['CWAM'] - d['HWAM']).astype(float),
        'season_length': (doy('MDAT') - doy('PDAT')).dt.days,
        'temp': (d['TMAXA'] + d['TMINA']) / 2,
    })
    return out[(out.season_length > 0) & (out.veg_att >= 0) & (out.temp > -90)].reset_index(drop=True)


def texture_pixels(run_dir):
    """{texture name: 0-20 cm ph, soc (g/kg), kex (mmol/kg), polsen (mg/kg) of its pixels}."""
    from pyquefts.core.conversions import mehlich3_to_olsen
    from spatialdata.soil_data import TEXTURE_CLASSES

    with xr.open_dataset(os.path.join(run_dir, 'soil_.nc')) as ds:
        weights = xr.DataArray([TOPSOIL_WEIGHTS.get(str(d), 0) for d in ds.depth.values], dims='depth')
        nodata = float(ds.attrs.get('nodata', -99.99))

        def topsoil(var):
            values = ds[var].where(ds[var] > nodata + 1)
            return ((values * weights).sum('depth', skipna=False) / weights.sum()).values

        soil = {
            'ph': topsoil('phh2o') / 10,            # pH x 10
            'soc': topsoil('soc') / 10,             # dg/kg -> g/kg
            'kex': topsoil('k') * 10,               # cmol(+)/kg -> mmol/kg
            'polsen': mehlich3_to_olsen(topsoil('p')),
        }
    with rasterio.open(os.path.join(run_dir, '_texture.tif')) as src:
        texture = src.read(1)
    assert texture.shape == soil['ph'].shape, '_texture.tif and soil_.nc are not on the same grid'
    valid = np.all([np.isfinite(v) for v in soil.values()], axis=0) & np.isfinite(texture) & (texture > 0)
    return {TEXTURE_CLASSES[int(code)].replace(' ', ''): {k: v[valid & (texture == code)] for k, v in soil.items()}
            for code in np.unique(texture[valid])}


def base_fertilizer(run_dir, texture):
    """Season total N, P, K (kg/ha of element) applied in the DSSAT run of the texture."""
    with open(os.path.join(run_dir, texture, 'experimental_file_config.yaml')) as f:
        management = yaml.safe_load(f).get('MANAGEMENT') or {}
    if str(management.get('phosphorus_simulation', 'N')).upper() == 'Y':
        print(f'warning: {texture} was run with the DSSAT P model (PHOSP Y), its attainable yield is already '
              'P-limited, so the QUEFTS P response counts the P limit twice')
    npk = management.get('fertilizer_npk') or []
    total = np.sum(np.array(npk, dtype=float).reshape(-1, 3), axis=0)
    return dict(zip('npk', np.where(total < 0, 0, total)))      # -99 (missing) counts as 0, as in DSSAT


def best_planting_dates(run_dir, texture, percentile):
    """Planting dates whose mean yield over the years is above the percentile of all dates."""
    y = pd.read_csv(os.path.join(run_dir, f'yield_data_{texture}.csv'))
    return y[y['y'] > np.percentile(y['y'], percentile)].astype({'TRNO': int})


def response(crop, pixels, runs, n_rates, p_rates, k_rates):
    """Yield (kg/ha) for year x N x P x K (season totals), mean of the pixels of the texture class."""
    import pyquefts

    px = lambda v: np.asarray(v, dtype=float).reshape(-1, 1, 1, 1, 1)
    yr = lambda v: np.asarray(v, dtype=float).reshape(1, -1, 1, 1, 1)
    supply = pyquefts.nut_supply2(yr(runs.temp), px(pixels['ph']), px(pixels['soc']), px(pixels['kex']),
                                  px(pixels['polsen']))
    soil = pyquefts.SoilParameters(N_base_supply=supply.N_base_supply, P_base_supply=supply.P_base_supply,
                                   K_base_supply=supply.K_base_supply)
    fert = pyquefts.Fertilizer(n_rates.reshape(1, 1, -1, 1, 1), p_rates.reshape(1, 1, 1, -1, 1),
                               k_rates.reshape(1, 1, 1, 1, -1))
    biom = pyquefts.Biomass(yr(runs.veg_att * LEAF_SHARE), yr(runs.veg_att * (1 - LEAF_SHARE)),
                            yr(runs.store_att), yr(runs.season_length))
    result = pyquefts.run_quefts(soil, crop, fert, biom, keep_intermediate=False)
    return np.nanmean(np.asarray(result['store_lim'], dtype=float), axis=0)


def response_table(crop_name, texture, trno, month_day, runs, yields, n_rates, p_rates, k_rates, base):
    """One row per extra N x P x K rate with the yield and the gain over the management, summarised over years."""
    grid = pd.MultiIndex.from_product([n_rates, p_rates, k_rates], names=['n_kg_ha', 'p_kg_ha', 'k_kg_ha'])
    table = grid.to_frame(index=False)
    for nutrient in 'kpn':
        table.insert(0, f'{nutrient}_base_kg_ha', base[nutrient])
    table.insert(0, 'planting_day', month_day)
    table.insert(0, 'TRNO', trno)
    table.insert(0, 'texture', texture)
    table.insert(0, 'crop', crop_name)
    table['p2o5_kg_ha'] = (table.p_kg_ha * P_TO_P2O5).round(1)
    table['k2o_kg_ha'] = (table.k_kg_ha * K_TO_K2O).round(1)
    table['n_years'] = len(runs)
    table['yield_attainable_mean'] = runs.store_att.mean()
    table['yield_attainable_p50'] = runs.store_att.median()
    table['yield_mean'] = yields.mean(axis=0)
    for q in PERCENTILES:
        table[f'yield_p{q}'] = np.percentile(yields, q, axis=0)
    # gain of every year over the management alone (extra P = K = 0) at the same N
    by_n = yields.reshape(len(runs), len(n_rates), -1)
    reference = by_n[:, :, :1]
    gain = (by_n - reference).reshape(len(runs), -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        gain_pct = np.where(reference > 0, 100 * (by_n - reference) / reference, np.nan).reshape(len(runs), -1)
    table['yield_gain_mean'] = gain.mean(axis=0)
    for q in GAIN_PERCENTILES:
        table[f'yield_gain_p{q}'] = np.percentile(gain, q, axis=0)
    for q in GAIN_PERCENTILES:
        table[f'yield_gain_pct_p{q}'] = np.nanpercentile(gain_pct, q, axis=0)
    return table.round(1)


def quefts_response(run_dir, crop_name, p_rates=DEFAULT_RATES['p'], k_rates=DEFAULT_RATES['k'],
                    n_rates=DEFAULT_RATES['n'], percentile=75, out_dir=None, verbose=True):
    """Write the QUEFTS response CSV files and the index of a village run folder; returns the output folder.

    crop_name is 'bean' or 'maize'; the rates are extra kg/ha of element ('first:last:step', 'a,b,c' or a list).
    """
    import pyquefts

    crop_name = crop_name.lower()
    assert crop_name in QUEFTS_CROP, f'QUEFTS response is only set up for {sorted(QUEFTS_CROP)}, not {crop_name}'
    crop = pyquefts.get_crop(QUEFTS_CROP[crop_name])
    n_rates, p_rates, k_rates = parse_rates(n_rates), parse_rates(p_rates), parse_rates(k_rates)
    out_dir = out_dir or os.path.join(run_dir, 'quefts')
    os.makedirs(out_dir, exist_ok=True)

    files, base = [], {}
    for texture, pixels in texture_pixels(run_dir).items():
        if not glob.glob(os.path.join(run_dir, texture, 'Summary_*.OUT')) or \
                not os.path.exists(os.path.join(run_dir, f'yield_data_{texture}.csv')):
            if verbose:
                print(f'{texture}: no DSSAT output, skipped')
            continue
        dates = best_planting_dates(run_dir, texture, percentile)
        base[texture] = b = base_fertilizer(run_dir, texture)
        for trno, month_day in zip(dates.TRNO, dates.month_day):
            runs = season_runs(run_dir, texture, trno)
            yields = response(crop, pixels, runs, n_rates + b['n'], p_rates + b['p'],
                              k_rates + b['k']).reshape(len(runs), -1)
            table = response_table(crop_name, texture, trno, month_day, runs, yields, n_rates, p_rates, k_rates, b)
            fn = f'quefts_response_{crop_name}_{texture}_trno{trno:02d}.csv'
            table.to_csv(os.path.join(out_dir, fn), index=False)
            files.append({'file': fn, 'texture': texture, 'TRNO': int(trno), 'planting_day': month_day})
        if verbose:
            print(f'{texture}: {len(pixels["ph"])} pixels, base N-P-K {b["n"]:g}-{b["p"]:g}-{b["k"]:g} kg/ha, '
                  f'{len(dates)} planting dates ({", ".join(dates.month_day)}) -> {out_dir}')
    with open(os.path.join(out_dir, f'quefts_index_{crop_name}.json'), 'w') as f:
        json.dump({'crop': crop_name, 'n_kg_ha': n_rates.tolist(), 'p_kg_ha': p_rates.tolist(),
                   'k_kg_ha': k_rates.tolist(), 'base_npk_kg_ha': {t: {k: float(v) for k, v in b.items()}
                                                                    for t, b in base.items()},
                   'files': files}, f, indent=1)
    return out_dir
