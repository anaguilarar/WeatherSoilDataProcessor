"""Validate DSSAT maize cultivars against observed yields (farm, municipio and department level).

Observed yields come from the validation table (government statistics and papers). For each observation
the script finds the matching villages, simulates the cultivars for primera and postrera, and compares
simulated and observed yields (RMSE, bias, R2) with a cluster bootstrap over units.

Steps
  1. Read the maize yield rows, convert them to t/ha, parse year, season and planting date.
  2. Locate each row:
       farm        coordinates inside a village polygon, or a village name that matches one aldea
       municipio   municipio name matched in the village table
       department  department name(s)
     The villages of a municipio/department are chosen and weighted with the rainfed maize harvested
     area of MapSPAM 2020 (--village-selection):
       coverage  villages with the most maize area until --maize-coverage of the unit's area is reached,
                 weighted by their maize area (default)
       pps       --max-villages-per-unit draws with probability proportional to maize area (with
                 replacement), plain mean of the draws; cheap, for cultivar screening
       random    --max-villages-per-unit random villages, plain mean
     Irrigated rows, rows that cannot be located, and census "annual" rows of a unit and year that also
     has primera/postrera rows are written to excluded_observations.csv.
  3. Simulate every needed village once per cultivar and season (cached per village and cultivar in
     OUT/cache/sim_<GEOCODIGO>_<cultivar>.csv, so new cultivars or villages only simulate what is missing).
  4. Prediction per observation: simulated years that match the observed year (if inside the weather
     period; otherwise all years), the planting window nearest to the reported planting date (if any),
     texture shares of the village as weights, and the weighted mean over the villages of the unit.
  5. One value per unit and season (and source type): observed and predicted yields are averaged over the
     rows of the same department/municipio/site and season (trial treatments, years), so a unit is not
     counted several times. --aggregate-by-year keeps the years apart.
  6. Metrics per level, source type and cultivar with 95% bootstrap intervals (units resampled with
     replacement, the same resample for every cultivar), the share of replicates in which each cultivar
     has the lowest RMSE (p_best) and the RMSE difference to the best cultivar.

The runs use no fertilizer. With --nitro Y (default) DSSAT simulates the soil N balance, so predictions
are water- and N-limited yields with soil N only; --nitro N turns the N balance off (water-limited yields,
sensitivity run). Use one --out folder per --nitro value: the cache refuses to mix them. With --fertilizer-rows prefer-unfertilized (default), a trial
unit and season that has unfertilized rows (0 kg N/ha, '-F') is compared with those rows only; units without
an unfertilized treatment keep all their rows.
HWAH is dry grain; use --grain-moisture to compare with yields reported at field moisture.

Workflow (local maps vs SoilGrids), one --out folder per soil source
  1. screening    all cultivars, local soils, pps sample of villages
       python scripts/validate_maize_cultivars.py --out validation_maize/screening --village-selection pps
  2. final        best cultivar(s), local soils and SoilGrids, maize-area coverage, same weather
       python scripts/validate_maize_cultivars.py --out validation_maize/local --cultivars IB0057:tocoron3
       python scripts/validate_maize_cultivars.py --out validation_maize/soilgrids --cultivars IB0057:tocoron3 \
           --villages-dir <village_sp_files_soilgrids> --weather-dir data/village_sp_files
  3. comparison
       python scripts/compare_soil_sources.py --a validation_maize/local --b validation_maize/soilgrids
  4. sensitivity  the same two runs with --nitro N (other --out folders, same cultivar), and step 3 on them

Usage (from the repository root)
  python scripts/validate_maize_cultivars.py --villages-dir data/village_sp_files --ncores 8
  python scripts/validate_maize_cultivars.py --skip-simulation      # metrics and plots from the cache
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULT_DATA = 'D:/OneDrive - CGIAR/projects/suelos_honduras/validation/data.csv'
SPAM_URL = 'https://digital-atlas.s3.amazonaws.com/cdh/data/mapspam2020-v2r2/spam2020-v2r2.zarr'

# Settings per crop (--crop). seasons: first planting date (month-day) and number of weekly planting windows;
# annual: seasons averaged for census rows of the whole year; unknown_season: season of rows that state none.
CROPS = {
    'maize': {
        'config': 'model_configurations/crop_model_configuration_maize.yaml',
        # MZCER048.CUL: generic long/medium/short season and tropical cultivars of the shortlist
        'cultivars': ('990001:long,990002:medium,990003:short,IB0057:tocoron-3,IB0065:suwan-1,'
                      'IB0063:pio-x304c,IB0060:h610,IB1072:dekalb-485,GF0001:garst-808,GF0301:gf-normcycle'),
        'spam_crop': 'maiz',
        'area_file': 'data/maize_area_village_spam2020.csv',
        'seasons': {
            'primera': {'start': '05-01', 'windows': 8},    # 1 May - 19 Jun
            'postrera': {'start': '08-15', 'windows': 5},   # 15 Aug - 12 Sep
        },
        'annual': ['primera', 'postrera'],
        'unknown_season': 'primera',
        'harvest_after_days': 210,   # late enough for the long-season cultivars
        'model': 'DSSAT CERES-Maize',
    },
    'bean': {
        'config': 'model_configurations/crop_model_configuration_bean.yaml',
        # BNGRO048.CUL: ICTA-Ostua (Mesoamerican small red, type II) and the generic Mesoamerican habits,
        # determinate habit 1 (shorter cycle) and indeterminate habit 2-3 (longer cycle)
        'cultivars': 'IB0006:icta-ostua,990005:short,990006:long',
        'spam_crop': 'bean',
        'area_file': 'data/bean_area_village_spam2020.csv',
        'seasons': {
            'primera': {'start': '05-15', 'windows': 7},    # 15 May - 26 Jun
            'postrera': {'start': '09-01', 'windows': 7},   # 1 Sep - 13 Oct, main bean season
            'apante': {'start': '11-15', 'windows': 6},     # 15 Nov - 20 Dec, residual moisture (north, east)
        },
        'annual': ['primera', 'postrera'],
        'unknown_season': 'postrera',
        'harvest_after_days': 150,
        'model': 'DSSAT CROPGRO-Drybean',
    },
}
FIRST_SIM_YEAR = 1990

# first two digits of GEOCODIGO (the shapefile names lost their accents)
DEPARTMENTS = {
    '01': 'Atlantida', '02': 'Colon', '03': 'Comayagua', '04': 'Copan', '05': 'Cortes', '06': 'Choluteca',
    '07': 'El Paraiso', '08': 'Francisco Morazan', '09': 'Gracias a Dios', '10': 'Intibuca',
    '11': 'Islas de la Bahia', '12': 'La Paz', '13': 'Lempira', '14': 'Ocotepeque', '15': 'Olancho',
    '16': 'Santa Barbara', '17': 'Valle', '18': 'Yoro',
}
LEVELS = ['farm', 'municipio', 'department']
NOT_A_PLACE = ('mean', 'pooled', 'farms in', 'multiple', 'not stated', 'trial site', 'hybrid', 'site mean')


# ----------------------------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------------------------
def normalize(text) -> str:
    """Lower case, no accents, single spaces. The replacement character of the shapefile becomes '.'."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKD', text.replace('\ufffd', '\x00'))
    text = ''.join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r'[^a-z0-9\x00]+', ' ', text).strip()
    return re.sub(r'\s+', ' ', text).replace('\x00', '.')


def to_t_ha(value, unit):
    text = str(value).strip()
    try:
        val = float(text.replace(',', '.'))
    except ValueError:
        return np.nan
    unit = str(unit).lower()
    if 'kg' in unit or 'not printed' in unit:
        if re.fullmatch(r'\d{1,3}[.,]\d{3}', text):   # '5.036' / '1,988' kg/ha: thousands separator
            t_ha = val
        elif 'not printed' in unit:
            t_ha = val if val < 100 else val / 1000
        else:
            t_ha = val / 1000
        return t_ha / 0.7 if 'mz' in unit else t_ha   # kg per manzana (0.7 ha)
    return val


def parse_years(text):
    """'1988' -> [1988]; '1996-1997' / '2005-2007 (mean)' -> range; '2023/24' or '1999/2000' -> first year."""
    text = str(text)
    if re.search(r'\d{4}\s*/\s*\d{2,4}', text):
        return [int(re.search(r'\d{4}', text).group())]
    years = [int(y) for y in re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', text)]
    if len(years) >= 2 and '-' in text:
        return list(range(min(years), max(years) + 1))
    return years[:1]


def parse_season(season, management):
    text = f'{season} {management}'.lower()
    if any(k in text for k in ('irrigat', 'riego', 'apante', 'verano', 'dry season')):
        return 'irrigated'
    has_primera, has_postrera = 'primera' in text, 'postrera' in text
    if has_primera and has_postrera:
        return 'annual'
    return 'postrera' if has_postrera else 'primera'


def parse_season_bean(season, management, planting_doy, unknown='postrera'):
    """Bean season: the first season named in the season text (source footnotes defining season codes are
    ignored); verano and irrigated rows -> 'irrigated'; without a season, the planting month, else `unknown`."""
    text = re.sub(r'codes? (?:letters )?defined.*', '', str(season).lower())
    text = re.sub(r'\(code letters[^)]*\)', '', text)
    if 'annual' in text or re.search(r'primera\s*\+\s*postrera', text):
        return 'annual'
    mgmt = str(management).lower()
    if re.search(r'irrigat|riego', text) or (re.search(r'irrigat|riego', mgmt) and 'without irrigation' not in mgmt):
        return 'irrigated'
    hits = {k: text.find(k) for k in ('primera', 'postrera', 'apante', 'verano') if k in text}
    if hits:
        first = min(hits, key=hits.get)
        return 'irrigated' if first == 'verano' else first
    if np.isfinite(planting_doy):
        if 120 <= planting_doy < 213:
            return 'primera'
        if 213 <= planting_doy < 305:
            return 'postrera'
        return 'apante' if planting_doy >= 305 or planting_doy < 32 else 'irrigated'   # Feb-Apr: verano
    return unknown


def parse_planting_doy(value):
    try:
        return pd.to_datetime(value, format='%Y-%m-%d').dayofyear
    except (ValueError, TypeError):
        pass
    try:  # e.g. '2 May (typical)'
        return pd.to_datetime(re.sub(r'\(.*\)', '', str(value)).strip() + ' 2001', format='%d %B %Y').dayofyear
    except (ValueError, TypeError):
        return np.nan


def parse_fertilization(treatment, management):
    """('none' | 'fertilized' | 'not reported', N rate in kg/ha or NaN) from the treatment and management text.

    'no fertilizer/sowing date printed' means not reported, and 'Testigo' is a check variety, not an
    unfertilized plot.
    """
    text = f'{treatment} | {management}'.lower()
    rate = re.search(r'(\d+(?:\.\d+)?)\s*kg\s*n\s*/\s*ha', text)
    rate = float(rate.group(1)) if rate else np.nan
    if rate == 0 or re.search(r'without fertili[sz]|\(-f\)|no fertili[sz]er(?!/)', text):
        return 'none', 0.0
    if np.isfinite(rate) or re.search(r'fertili[sz]|urea|18-46-0|npk|\(\+f\)', text):
        return 'fertilized', rate
    return 'not reported', rate


def load_observations(path, crop='maize', unknown_season=None):
    df = pd.read_csv(path)
    df = df.loc[(df['crop'].str.lower() == crop) & (df['variable'] == 'yield')].copy()
    df = df.rename(columns={'cultivar': 'obs_cultivar'})
    df.insert(0, 'obs_id', df.index)
    df['obs_t_ha'] = [to_t_ha(v, u) for v, u in zip(df['value_reported'], df['unit_reported'])]
    df['years'] = df['year'].map(parse_years)
    df['planting_doy'] = df['planting_date'].map(parse_planting_doy)
    if crop == 'maize':
        df['sim_season'] = [parse_season(s, m) for s, m in zip(df['season'], df['management'])]
    else:
        unknown = unknown_season or CROPS[crop]['unknown_season']
        df['sim_season'] = [parse_season_bean(s, m, d, unknown)
                            for s, m, d in zip(df['season'], df['management'], df['planting_doy'])]
    fert = [parse_fertilization(t, m) for t, m in zip(df['treatment_or_system'], df['management'])]
    df['fertilization'] = [f for f, _ in fert]
    df['n_rate_kg_ha'] = [n for _, n in fert]
    df['source_type'] = np.where(df['data_type'].str.contains('census|survey|report', case=False, na=False),
                                 'census/survey', 'field trial')
    return df


def fertilized_at_unfertilized_unit(obs):
    """Fertilized/not reported rows of a unit, season and source that also has unfertilized rows."""
    key = obs[AGG_KEY].astype(object).fillna('').astype(str).agg('|'.join, axis=1)
    with_none = set(key[obs['fertilization'] == 'none'])
    return (obs['fertilization'] != 'none') & key.isin(with_none)


def redundant_annual(obs):
    """Census 'annual' rows of a unit, year and source that also has season rows (primera, postrera, ...)."""
    key = obs['level'].astype(str) + '|' + obs['unit'].astype(str) + '|' + obs['year'].astype(str) \
        + '|' + obs['source_type']
    with_season = set(key[~obs['sim_season'].isin(['annual', 'irrigated'])])
    return (obs['sim_season'] == 'annual') & key.isin(with_season)


# ----------------------------------------------------------------------------------------------
# locations and maize area
# ----------------------------------------------------------------------------------------------
def load_villages(path):
    import geopandas as gpd
    gdf = gpd.read_file(path, encoding='utf-8')
    gdf = gdf.dropna(subset=['GEOCODIGO']).copy()   # Lago de Yojoa has no code
    gdf['GEOCODIGO'] = gdf['GEOCODIGO'].astype(str).str.zfill(6)
    gdf['dep_norm'] = gdf['GEOCODIGO'].str[:2].map(DEPARTMENTS).map(normalize)
    gdf['mun_norm'] = gdf['MUNI'].map(normalize)
    gdf['ald_norm'] = gdf['ALDEA'].map(normalize)
    return gdf.to_crs('EPSG:4326')


def maize_area_by_village(villages, path, url=SPAM_URL, crop='maize'):
    """Rainfed harvested area (ha) of the crop (MapSPAM 2020) in each village, cached in `path`.

    The SPAM cells (5 arc-min) are much larger than the villages: each cell's area is spread uniformly over
    the cell and every village gets density x (village ∩ cell) area. The column keeps the name maize_ha for
    every crop.
    """
    if os.path.exists(path):
        return pd.read_csv(path, dtype={'GEOCODIGO': str}).set_index('GEOCODIGO')['maize_ha']
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import box
    spam_crop = CROPS[crop]['spam_crop']
    print(f'reading MapSPAM rainfed {crop} ({spam_crop}) harvested area from {url}')
    minx, miny, maxx, maxy = villages.total_bounds
    da = xr.open_zarr(url)['harvested_area'].sel(crop=spam_crop, technology='rainfed')
    da = da.sel(x=slice(minx - 0.1, maxx + 0.1), y=slice(maxy + 0.1, miny - 0.1)).load()
    dx, dy = abs(float(da.x[1] - da.x[0])), abs(float(da.y[1] - da.y[0]))
    cells = da.to_dataframe(name='ha').reset_index()[['x', 'y', 'ha']]
    cells = cells.loc[cells['ha'] > 0]
    cells = gpd.GeoDataFrame(cells[['ha']], crs='EPSG:4326', geometry=[
        box(x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2) for x, y in zip(cells['x'], cells['y'])]).to_crs(6933)
    cells['density'] = cells['ha'] / (cells.area / 1e4)
    vil = villages[['GEOCODIGO', 'geometry']].to_crs(6933)
    vil['geometry'] = vil.make_valid()
    vil['village_ha'] = vil.area / 1e4
    inter = gpd.overlay(vil[['GEOCODIGO', 'geometry']], cells[['density', 'geometry']], how='intersection',
                        keep_geom_type=True)
    inter['maize_ha'] = inter['density'] * inter.area / 1e4
    out = vil.groupby('GEOCODIGO')['village_ha'].sum().to_frame().join(
        inter.groupby('GEOCODIGO')['maize_ha'].sum()).fillna({'maize_ha': 0}).reset_index()
    out['maize_share'] = out['maize_ha'] / out['village_ha']
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False)
    print(f'{crop} area per village saved in {path} ({out["maize_ha"].sum():,.0f} ha rainfed {crop})')
    return out.set_index('GEOCODIGO')['maize_ha']


def village_selector(maize_ha, method, n, coverage, seed):
    """Returns select(unit, codes) -> (geocodes, weights); the same villages for every row of a unit."""
    rng = np.random.default_rng(seed)
    chosen = {}

    def select(unit, codes):
        if unit in chosen:
            return chosen[unit]
        codes = sorted(set(codes))
        area = maize_ha.reindex(codes).fillna(0).to_numpy()
        if method == 'random' or area.sum() <= 0:
            pick = codes if len(codes) <= n else sorted(rng.choice(codes, size=n, replace=False).tolist())
            chosen[unit] = (pick, [1.0] * len(pick))
        elif method == 'pps':   # Hansen-Hurwitz: draws with p ~ area, plain mean of the draws
            draws, counts = np.unique(rng.choice(codes, size=n, replace=True, p=area / area.sum()),
                                      return_counts=True)
            chosen[unit] = (draws.tolist(), counts.astype(float).tolist())
        else:                   # coverage: largest maize villages up to `coverage` of the unit's area
            order = np.argsort(-area)
            share = np.cumsum(area[order]) / area.sum()
            keep = order[:int(np.searchsorted(share, coverage) + 1)]
            keep = keep[area[keep] > 0]
            chosen[unit] = ([codes[i] for i in keep], area[keep].tolist())
        return chosen[unit]

    return select


def name_matches(name_norm, column):
    """Rows whose normalized name equals name_norm ('.' in the table is one lost accented letter)."""
    if not name_norm:
        return pd.Series(False, index=column.index)
    return column.map(lambda s: bool(s) and re.fullmatch(re.escape(s).replace(r'\.', '.'), name_norm) is not None)


def split_names(text):
    """'multiple (El Paraiso, Cortes)' / 'A; B' / 'Candelaria / Guarita' -> names."""
    text = re.sub(r'^\s*multiple\s*', '', str(text), flags=re.I)
    return [p for p in (normalize(t) for t in re.split(r'[;,/()]| and ', text)) if p]


def place_candidates(text):
    """First name of a free-text site: 'Galeras (farm of Marcos)' -> 'galeras', 'Zamorano, lote...' -> 'zamorano'."""
    if not isinstance(text, str) or text.lower().startswith(NOT_A_PLACE):
        return []
    head = normalize(re.split(r'[(,;]', text)[0])
    return [head, f'el {head}', f'la {head}'] if head else []


def locate(obs, villages, select):
    """Add level, match_method, unit, geocodes and village weights (lists) to each observation."""
    import geopandas as gpd
    pts = obs.dropna(subset=['lat', 'lon'])
    pts = gpd.GeoDataFrame(pts[['obs_id']], geometry=gpd.points_from_xy(pts['lon'], pts['lat']), crs='EPSG:4326')
    joined = gpd.sjoin_nearest(pts.to_crs(3857), villages[['GEOCODIGO', 'geometry']].to_crs(3857),
                               how='left', max_distance=2000).drop_duplicates('obs_id')
    by_point = dict(zip(joined['obs_id'], joined['GEOCODIGO']))

    rows = []
    for _, r in obs.iterrows():
        deps = [d for d in split_names(r['department']) if d in set(villages['dep_norm'])]
        in_dep = villages['dep_norm'].isin(deps) if deps else pd.Series(True, index=villages.index)
        mun_names = split_names(r['municipio'])
        in_mun = pd.Series(False, index=villages.index)
        if mun_names:
            in_mun = name_matches(mun_names[0], villages['mun_norm']) & in_dep
        level, method, unit, codes, weights = None, None, None, [], []
        if isinstance(by_point.get(r['obs_id']), str):
            level, method, codes = 'farm', 'coordinates', [by_point[r['obs_id']]]
        else:
            scope = in_mun if in_mun.any() else in_dep
            for name in place_candidates(r['aldea_or_site']):
                hit = villages.loc[scope & name_matches(name, villages['ald_norm']), 'GEOCODIGO'].unique()
                if len(hit) == 1:
                    level, method, codes = 'farm', f'village name ({name})', [hit[0]]
                    break
        if level is None and in_mun.any():
            level, method = 'municipio', f'municipio name ({mun_names[0]})'
            unit = f"{'+'.join(sorted(deps))}|{mun_names[0]}"
            codes, weights = select(unit, villages.loc[in_mun, 'GEOCODIGO'].unique())
        if level is None and deps:
            level, method = 'department', 'department name(s)'
            unit = '+'.join(sorted(deps))
            codes, weights = select(unit, villages.loc[in_dep, 'GEOCODIGO'].unique())
        if level == 'farm':
            unit, weights = codes[0], [1.0]
        rows.append({'obs_id': r['obs_id'], 'level': level, 'match_method': method, 'unit': unit,
                     'geocodes': codes, 'village_weights': weights})
    return obs.merge(pd.DataFrame(rows), on='obs_id')


# ----------------------------------------------------------------------------------------------
# simulation (worker)
# ----------------------------------------------------------------------------------------------
def texture_weights(tmp_path):
    import rioxarray
    from spatialdata.soil_data import TEXTURE_CLASSES
    values = rioxarray.open_rasterio(os.path.join(tmp_path, 'texture.tif')).values.ravel()
    values = values[np.isfinite(values) & (values > 0) & (values < 100)].astype(int)
    codes, counts = np.unique(values, return_counts=True)
    return {TEXTURE_CLASSES[c].replace(' ', ''): n / counts.sum() for c, n in zip(codes, counts)}


def clear_previous_outputs(process_paths):
    """DSSAT only copies Summary.OUT when Summary_<n>.OUT is missing: remove old results before each run."""
    for path in process_paths:
        for fn in os.listdir(path):
            full = os.path.join(path, fn)
            if fn.startswith('Summary_') and fn.endswith('.OUT'):
                os.remove(full)
            elif fn.startswith('_') and os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)


def match_soil_to_weather(soil_nc, weather_nc):
    """Reproject the soil datacube onto the weather grid (nearest neighbour), in place.

    The texture groups are computed on the soil grid and applied to the weather pixels, so both cubes need
    the same grid. Needed when soil and weather come from different village packages (e.g. SoilGrids soil
    in Goode Homolosine with the local weather in EPSG:4326).
    """
    import rioxarray  # noqa: F401  (registers .rio)
    import xarray as xr
    from rasterio.enums import Resampling
    with xr.open_dataset(weather_nc) as w:
        template = w[list(w.data_vars)[0]].isel(date=0).load()
        template = template.rio.write_crs(w.attrs.get('crs') or template.rio.crs)
    with xr.open_dataset(soil_nc) as ds:
        soil = ds.load()
    soil = soil.rio.write_crs(soil.attrs.get('crs') or soil.rio.crs)
    if soil.rio.crs == template.rio.crs and soil.sizes['x'] == template.sizes['x']             and soil.sizes['y'] == template.sizes['y'] and np.allclose(soil['x'], template['x'])             and np.allclose(soil['y'], template['y']):
        return
    out = xr.Dataset({k: soil[k].transpose('depth', 'y', 'x').rio.write_nodata(np.nan)
                     .rio.reproject_match(template, resampling=Resampling.nearest)
                      for k in soil.data_vars}, attrs=soil.attrs)
    out = out.assign_coords(x=template['x'].values, y=template['y'].values)   # exact same coordinates
    for k in out.data_vars:
        out[k] = out[k].where(np.isfinite(out[k]))
        out[k].attrs.pop('_FillValue', None)
        out[k].encoding = {}
    out.attrs.update(crs=str(template.rio.crs), width=template.sizes['x'], height=template.sizes['y'],
                     transform=np.array(template.rio.transform())[:6])
    out.to_netcdf(soil_nc + '.tmp.nc')
    os.replace(soil_nc + '.tmp.nc', soil_nc)


def simulate_village(geocode, config, villages_dir, weather_dir, cultivars, nitro='Y', keep_runs=False,
                     crop='maize'):
    """Simulate one village; soil from villages_dir, weather and dem from weather_dir."""
    from omegaconf import OmegaConf
    from crop_modeling.spatial_process import SpatialCM
    from crop_modeling.utils.process import model_selection
    from crop_modeling.utils.output_transforms import update_data_using_path, ColumnNames
    from spatialdata.soil_data import TEXTURE_CLASSES
    from spatialdata.utils import read_compressed_xarray

    config = OmegaConf.create(OmegaConf.to_container(config))
    working_path = config.GENERAL_INFO.working_path
    shutil.rmtree(os.path.join(working_path, geocode), ignore_errors=True)
    cm_sp = SpatialCM(configuration_dict=config, load_env_data=False)
    roi = cm_sp.geo_features.loc[cm_sp.geo_features['GEOCODIGO'].astype(str).str.zfill(6) == geocode]
    if roi.empty:
        raise ValueError(f'{geocode} not in the village shapefile')
    cm_sp.model = model_selection('dssat', working_path)
    cm_sp.set_up_folders(site=geocode)
    for dataset in ['weather', 'soil', 'dem']:
        fn = os.path.basename(cm_sp.__dict__[f'_{dataset}_tmppath']).replace('.nc', '.zip')
        src = os.path.join(villages_dir if dataset == 'soil' else weather_dir, geocode, fn)
        if not os.path.exists(src):
            if dataset == 'dem':
                continue
            raise FileNotFoundError(src)
        shutil.copy2(src, cm_sp._tmp_path)
        read_compressed_xarray(os.path.join(cm_sp._tmp_path, fn), cm_sp._tmp_path)
    if os.path.normpath(weather_dir) != os.path.normpath(villages_dir):
        match_soil_to_weather(cm_sp._soil_tmppath, cm_sp._weather_tmppath)
    cm_sp.create_roi_sp_data(roi=roi, group_codes=TEXTURE_CLASSES, export_spatial_data=False)
    cm_sp.model.find_envworking_paths(cm_sp._tmp_path, 'WTH')
    weights = texture_weights(cm_sp._tmp_path)
    cols = ColumnNames('dssat').growth_colnames

    results = []
    harvest_after = CROPS[crop]['harvest_after_days']
    for season, spec in CROPS[crop]['seasons'].items():
        planting = pd.Timestamp(f"{FIRST_SIM_YEAR}-{spec['start']}")
        management = dict(OmegaConf.to_container(config.MANAGEMENT))
        management.update(planting_date=planting.strftime('%Y-%m-%d'),
                          harvesting_date=(planting + pd.Timedelta(days=harvest_after)).strftime('%Y-%m-%d'),
                          plantingWindow=spec['windows'], nitrogen_simulation=nitro,
                          fertilizer_schedule={'days_after_planting': None, 'npk': None})
        for cultivar in cultivars:
            clear_previous_outputs(cm_sp.model._process_paths)
            cm_sp.model.set_up_crop(crop=cm_sp.crop, cultivar=cultivar)
            cm_sp.model.set_up_management(crop=cm_sp.crop, cultivar=cultivar, verbose=False, **management)
            done = cm_sp.model.run(cm_sp.model.crop_code, crop=cm_sp.crop, planting_window=spec['windows'],
                                   bin_path=config.GENERAL_INFO.bin_path, dssat_path=config.GENERAL_INFO.dssat_path,
                                   remove_tmp_folder=True, verbose=False)
            outputs = update_data_using_path(cm_sp._tmp_path, model='dssat')
            for texture, ok in done.items():
                if not ok:
                    continue
                sim = outputs[texture].output_data()
                results.append(pd.DataFrame({
                    'geocode': geocode, 'season': season, 'cultivar': cultivar, 'texture': texture,
                    'weight': weights.get(texture, np.nan),
                    'pdate': pd.to_datetime(sim[cols['date']]).values,
                    'mdate': pd.to_datetime(sim[cols['mdate']], errors='coerce').values,
                    'hdate': pd.to_datetime(sim[cols['hdate']], errors='coerce').values,
                    'hwah_kg_ha': pd.to_numeric(sim[cols['yield']], errors='coerce').values}))
    if not keep_runs:
        shutil.rmtree(os.path.join(working_path, geocode), ignore_errors=True)
    if not results:
        raise RuntimeError('no texture completed')
    return pd.concat(results, ignore_index=True)


def sim_file(cache, geocode, cultivar):
    return os.path.join(cache, f'sim_{geocode}_{cultivar}.csv')


def check_cache_settings(cache, args):
    """The cache keeps the simulation settings; a folder holds one soil source, weather source and NITRO."""
    folder = lambda path: os.path.basename(os.path.normpath(path))   # portable between local and server
    settings = {'crop': args.crop, 'nitro': args.nitro, 'soil_folder': folder(args.villages_dir),
                'weather_folder': folder(args.weather_dir)}
    path = os.path.join(cache, 'settings.json')
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        diff = {k: (saved.get(k), v) for k, v in settings.items() if saved.get(k, v) != v}
        if diff:
            raise SystemExit(f'{cache} was simulated with other settings {diff} (saved, now): use another --out')
    elif args.nitro != 'Y' and any(fn.startswith('sim_') for fn in os.listdir(cache)):
        raise SystemExit(f'{cache} has simulations made before --nitro existed (NITRO = Y): use another --out')
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)


def migrate_cache(cache):
    """Split the old per-village cache files (sim_<GEOCODIGO>.csv, all cultivars) into one file per cultivar."""
    for fn in os.listdir(cache):
        if re.fullmatch(r'sim_\d{6}\.csv', fn):
            old = pd.read_csv(os.path.join(cache, fn), dtype={'geocode': str, 'cultivar': str})
            for code, sub in old.groupby('cultivar'):
                if not os.path.exists(sim_file(cache, fn[4:10], code)):
                    sub.to_csv(sim_file(cache, fn[4:10], code), index=False)
            os.replace(os.path.join(cache, fn), os.path.join(cache, fn + '.migrated'))


def worker(jobs_file, args):
    from omegaconf import OmegaConf
    os.chdir(REPO)
    config = OmegaConf.load(args.config)
    config.GENERAL_INFO.working_path = args.work_dir
    config.SPATIAL_INFO.geospatial_path = args.villages_shp
    if args.bin_path:
        config.GENERAL_INFO.bin_path = args.bin_path
    if args.dssat_path:
        config.GENERAL_INFO.dssat_path = args.dssat_path
    cache = os.path.join(args.out, 'cache')
    with open(jobs_file) as f:
        jobs = [line.split() for line in f if line.strip()]
    for i, (geocode, codes) in enumerate(jobs, 1):
        t0 = time.perf_counter()
        try:
            sim = simulate_village(geocode, config, args.villages_dir, args.weather_dir, codes.split(','),
                                   nitro=args.nitro, keep_runs=args.keep_runs, crop=args.crop)
            for code, sub in sim.groupby('cultivar'):
                sub.to_csv(sim_file(cache, geocode, code) + '.tmp', index=False)
                os.replace(sim_file(cache, geocode, code) + '.tmp', sim_file(cache, geocode, code))
            if os.path.exists(os.path.join(cache, f'failed_{geocode}.txt')):
                os.remove(os.path.join(cache, f'failed_{geocode}.txt'))
            print(f'[{i}/{len(jobs)}] {geocode} ok, {time.perf_counter() - t0:.0f} s', flush=True)
        except Exception as exc:  # keep going with the other villages
            with open(os.path.join(cache, f'failed_{geocode}.txt'), 'w') as f:
                f.write(repr(exc))
            print(f'[{i}/{len(jobs)}] {geocode} FAILED: {exc!r}', flush=True)


def run_simulations(geocodes, cultivars, args):
    cache = os.path.join(args.out, 'cache')
    jobs = {}
    for g in sorted(set(geocodes)):
        missing = [c for c in cultivars if not os.path.exists(sim_file(cache, g, c))]
        if missing and (args.retry_failed or not os.path.exists(os.path.join(cache, f'failed_{g}.txt'))):
            jobs[g] = missing
    print(f'villages needed: {len(set(geocodes))}, complete in cache: {len(set(geocodes)) - len(jobs)}, '
          f'to simulate: {len(jobs)} ({sum(map(len, jobs.values()))} village x cultivar runs)')
    if not jobs:
        return
    pending = list(jobs)
    nworkers = max(1, min(args.ncores, len(pending)))
    procs = []
    for k in range(nworkers):
        jobs_file = os.path.join(cache, f'jobs_{k}.txt')
        with open(jobs_file, 'w') as f:
            f.write('\n'.join(f'{g} {",".join(jobs[g])}' for g in pending[k::nworkers]))
        cmd = [sys.executable, os.path.abspath(__file__), '--worker', jobs_file] + args.passthrough
        log = open(os.path.join(cache, f'worker_{k}.log'), 'w')
        procs.append((subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(args.stagger)   # workers starting at the same instant can collide on the DSSAT files
    t0 = time.perf_counter()
    while any(p.poll() is None for p, _ in procs):
        done = sum(all(os.path.exists(sim_file(cache, g, c)) for c in jobs[g]) for g in pending)
        print(f'\r  simulated {done}/{len(pending)} villages, {time.perf_counter() - t0:.0f} s', end='', flush=True)
        time.sleep(15)
    for _, log in procs:
        log.close()
    print()


# ----------------------------------------------------------------------------------------------
# predictions
# ----------------------------------------------------------------------------------------------
def load_cache(geocodes, cultivars, out):
    frames = []
    for g in sorted(set(geocodes)):
        for c in cultivars:
            fn = sim_file(os.path.join(out, 'cache'), g, c)
            if os.path.exists(fn):
                frames.append(pd.read_csv(fn, dtype={'geocode': str, 'cultivar': str}, parse_dates=['pdate']))
    sim = pd.concat(frames, ignore_index=True)
    sim['year'] = sim['pdate'].dt.year
    sim['doy'] = sim['pdate'].dt.dayofyear
    sim['hwah_kg_ha'] = sim['hwah_kg_ha'].where(sim['hwah_kg_ha'] >= 0)   # -99 = no harvest
    sim['weight'] = sim['weight'].fillna(0)
    return sim


def village_prediction(sim_v, season, years, planting_doy):
    """Texture-weighted mean yield (kg/ha) of one village, cultivar and season."""
    sub = sim_v.loc[sim_v['season'] == season]
    if sub.empty:
        return np.nan, False
    matched = sub['year'].isin(years)
    year_matched = bool(matched.any())
    if year_matched:
        sub = sub.loc[matched]
    if np.isfinite(planting_doy):
        windows = sub['doy'].unique()
        nearest = windows[np.argmin(np.abs(windows - planting_doy))]
        sub = sub.loc[(sub['doy'] - nearest).abs() <= 3]
    by_texture = sub.groupby('texture').agg(y=('hwah_kg_ha', 'mean'), w=('weight', 'first')).dropna()
    if by_texture.empty:
        return np.nan, year_matched
    w = by_texture['w'] if by_texture['w'].sum() > 0 else pd.Series(1.0, index=by_texture.index)
    return float(np.average(by_texture['y'], weights=w)), year_matched


def predict(obs, sim, cultivars, moisture, annual=('primera', 'postrera')):
    grouped = {k: v for k, v in sim.groupby(['geocode', 'cultivar'])}
    rows = []
    for _, r in obs.iterrows():
        seasons = list(annual) if r['sim_season'] == 'annual' else [r['sim_season']]
        for code, label in cultivars.items():
            values, weights, year_flags = [], [], []
            for g, w in zip(r['geocodes'], r['village_weights']):
                sim_v = grouped.get((g, code))
                if sim_v is None:
                    continue
                per_season = [village_prediction(sim_v, s, r['years'], r['planting_doy']) for s in seasons]
                vals = [v for v, _ in per_season if np.isfinite(v)]
                if vals:
                    values.append(np.mean(vals))
                    weights.append(w)
                    year_flags.append(any(f for _, f in per_season))
            pred = np.average(values, weights=weights) / 1000 / (1 - moisture) if values else np.nan
            share = sum(weights) / sum(r['village_weights']) if values else 0.0
            rows.append({'obs_id': r['obs_id'], 'cultivar': label, 'cultivar_code': code,
                         'pred_t_ha': pred, 'n_villages_simulated': len(values), 'weight_simulated': share,
                         'year_matched': bool(year_flags) and all(year_flags)})
    return obs.merge(pd.DataFrame(rows), on='obs_id')


AGG_KEY = ['level', 'source_type', 'unit', 'sim_season']


def aggregate(pred, by_year=False):
    """One value per unit, season and source type (and year): mean of the observed and predicted rows."""
    key = AGG_KEY + (['year'] if by_year else [])
    agg = (pred.dropna(subset=['pred_t_ha'])
           .groupby(key + ['cultivar', 'cultivar_code'], dropna=False)
           .agg(obs_t_ha=('obs_t_ha', 'mean'), pred_t_ha=('pred_t_ha', 'mean'), n_rows=('obs_id', 'size'),
                obs_sd=('obs_t_ha', 'std'), department=('department', 'first'),
                years=('year', lambda s: ';'.join(sorted(set(map(str, s))))),
                year_matched=('year_matched', 'mean'))
           .reset_index())
    agg['row_key'] = agg[key].astype(object).fillna('').astype(str).agg('|'.join, axis=1)
    return agg


# ----------------------------------------------------------------------------------------------
# metrics, bootstrap and plots
# ----------------------------------------------------------------------------------------------
def metrics(df):
    o, p = df['obs_t_ha'].to_numpy(), df['pred_t_ha'].to_numpy()
    n = len(o)
    if n < 2:
        return pd.Series({'n': n})
    rmse = np.sqrt(np.mean((p - o) ** 2))
    sst = np.sum((o - o.mean()) ** 2)
    return pd.Series({
        'n': n, 'n_units': df['unit'].nunique(), 'obs_mean': o.mean(), 'pred_mean': p.mean(),
        'bias': np.mean(p - o), 'mae': np.mean(np.abs(p - o)), 'rmse': rmse, 'nrmse_pct': 100 * rmse / o.mean(),
        'r2': 1 - np.sum((p - o) ** 2) / sst if sst > 0 else np.nan,              # 1 - SSE/SST (can be < 0)
        'r2_pearson': np.corrcoef(o, p)[0, 1] ** 2 if np.std(o) > 0 and np.std(p) > 0 else np.nan,
        'ubrmse': np.std(p - o),                                                     # RMSE without the mean bias
        'r': np.corrcoef(o, p)[0, 1] if np.std(o) > 0 and np.std(p) > 0 else np.nan,
    })


def cluster_bootstrap(obs, preds, clusters, n_boot=2000, seed=42):
    """Resample clusters (units) with replacement; the same resample for every model (paired).

    obs: (n,) observed; preds: (n, k) predictions of k models; clusters: (n,) unit of each row.
    Returns {'rmse', 'bias', 'mae', 'r2', 'ubrmse', 'r'}: arrays (n_boot, k).
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    members = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    out = {m: np.full((n_boot, preds.shape[1]), np.nan) for m in ('rmse', 'bias', 'mae', 'r2', 'ubrmse', 'r')}
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        o, err = obs[idx], preds[idx] - obs[idx, None]
        sst = np.sum((o - o.mean()) ** 2)
        out['rmse'][b] = np.sqrt(np.mean(err ** 2, axis=0))
        out['bias'][b] = err.mean(axis=0)
        out['mae'][b] = np.abs(err).mean(axis=0)
        out['r2'][b] = 1 - np.sum(err ** 2, axis=0) / sst if sst > 0 else np.nan
        out['ubrmse'][b] = err.std(axis=0)
        if o.std() > 0:
            pc, oc = preds[idx] - preds[idx].mean(axis=0), (o - o.mean())[:, None]
            with np.errstate(invalid='ignore', divide='ignore'):
                out['r'][b] = (pc * oc).sum(axis=0) / np.sqrt((pc ** 2).sum(axis=0) * (oc ** 2).sum())
    return out


def subsets(agg):
    """(level, source) groups: every source type of a level, and 'all' when a level has more than one."""
    for level in [lv for lv in LEVELS if (agg['level'] == lv).any()]:
        sources = sorted(agg.loc[agg['level'] == level, 'source_type'].unique())
        for src in sources + (['all'] if len(sources) > 1 else []):
            mask = (agg['level'] == level) & ((agg['source_type'] == src) if src != 'all' else True)
            yield level, src, agg.loc[mask]


def evaluate(agg, cultivars, n_boot, seed):
    """Point metrics, bootstrap 95% intervals, p_best and RMSE difference to the best cultivar per subset."""
    tables = []
    for level, src, sub in subsets(agg):
        wide = sub.pivot_table(index=['row_key', 'unit'], columns='cultivar', values='pred_t_ha')
        wide = wide[[c for c in cultivars.values() if c in wide.columns]].dropna()
        if len(wide) < 3:
            continue
        o = sub.drop_duplicates('row_key').set_index('row_key').loc[wide.index.get_level_values(0), 'obs_t_ha']
        o = o.to_numpy()
        boot = cluster_bootstrap(o, wide.to_numpy(), wide.index.get_level_values(1), n_boot, seed)
        point = pd.DataFrame([metrics(pd.DataFrame({'obs_t_ha': o, 'pred_t_ha': wide[c].to_numpy(),
                                                    'unit': wide.index.get_level_values(1)})) for c in wide.columns])
        point.insert(0, 'cultivar', wide.columns)
        best = int(np.argmin(point['rmse']))
        for m in ('rmse', 'bias', 'r2'):
            point[f'{m}_lo'], point[f'{m}_hi'] = np.nanpercentile(boot[m], [2.5, 97.5], axis=0)
        point['p_best'] = np.bincount(np.argmin(boot['rmse'], axis=1), minlength=len(wide.columns)) / n_boot
        diff = boot['rmse'] - boot['rmse'][:, [best]]
        point['d_rmse_vs_best'] = point['rmse'] - point['rmse'].iloc[best]
        point['d_rmse_lo'], point['d_rmse_hi'] = np.percentile(diff, [2.5, 97.5], axis=0)
        point['rank'] = point['rmse'].rank(method='min').astype(int)
        point.insert(0, 'source_type', src)
        point.insert(0, 'level', level)
        tables.append(point)
    table = pd.concat(tables, ignore_index=True)
    code = {v: k for k, v in cultivars.items()}
    table.insert(3, 'cultivar_code', table['cultivar'].map(code))
    return table.sort_values(['level', 'source_type', 'rmse']).reset_index(drop=True)


def save_figure(fig, path):
    try:
        fig.savefig(path, dpi=150)
    except OSError:  # file open in a viewer or locked by OneDrive
        path = path.replace('.png', time.strftime('_%Y%m%d_%H%M%S.png'))
        fig.savefig(path, dpi=150)
        print(f'plot file was locked, saved as {path}')


def rmse_plot(table, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = list(table.groupby(['level', 'source_type'], sort=False))
    ncol = min(3, len(groups))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 0.32 * table['cultivar'].nunique() * nrow + 1.6 * nrow),
                             squeeze=False)
    for ax, ((level, src), t) in zip(axes.ravel(), groups):
        t = t.sort_values('rmse', ascending=False)
        ax.barh(t['cultivar'], t['rmse'], color=np.where(t['rank'] == 1, '#d62728', '#7f7f7f'),
                xerr=[t['rmse'] - t['rmse_lo'], t['rmse_hi'] - t['rmse']], error_kw=dict(lw=0.8, capsize=2))
        for y, (_, r) in enumerate(t.iterrows()):
            ax.text(r['rmse_hi'] + 0.03, y, f"p_best {r['p_best']:.2f}", va='center', fontsize=7)
        ax.set_title(f"{level} - {src} (n = {int(t['n'].iloc[0])}, units = {int(t['n_units'].iloc[0])})",
                     fontsize=9)
        ax.set_xlabel('RMSE (t/ha), 95% bootstrap interval')
        ax.set_xlim(0, t['rmse_hi'].max() * 1.35)
    for ax in axes.ravel()[len(groups):]:
        ax.axis('off')
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def scatter_plots(agg, table, labels, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    levels = [lv for lv in LEVELS if (agg['level'] == lv).any()]
    lim = np.nanmax(agg[['obs_t_ha', 'pred_t_ha']].to_numpy()) * 1.05
    colors = {'field trial': '#1f77b4', 'census/survey': '#d62728'}
    markers = {'primera': 'o', 'postrera': 's', 'annual': '^', 'apante': 'D'}
    fig, axes = plt.subplots(len(levels), len(labels), figsize=(4.2 * len(labels), 4.0 * len(levels)),
                             squeeze=False, sharex=True, sharey=True)
    for i, level in enumerate(levels):
        for j, label in enumerate(labels):
            ax = axes[i, j]
            sub = agg.loc[(agg['level'] == level) & (agg['cultivar'] == label)]
            for (src, season), s in sub.groupby(['source_type', 'sim_season']):
                ax.scatter(s['obs_t_ha'], s['pred_t_ha'], s=24, alpha=0.8, color=colors.get(src),
                           marker=markers.get(season, 'o'), label=f'{src}, {season}', edgecolor='white',
                           linewidth=0.4)
            ax.plot([0, lim], [0, lim], color='0.4', lw=1, ls='--')
            m = table.loc[(table['level'] == level) & (table['cultivar'] == label)]
            lines = [f"{r['source_type']}: n={int(r['n'])} RMSE={r['rmse']:.2f} [{r['rmse_lo']:.2f}-"
                     f"{r['rmse_hi']:.2f}] R²={r['r2']:.2f} bias={r['bias']:+.2f}" for _, r in m.iterrows()]
            ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=6.5,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(f'{level}\npredicted (t/ha)')
            if i == len(levels) - 1:
                ax.set_xlabel('observed (t/ha)')
    legend = {}
    for ax in axes.ravel():
        for h, name in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(name, h)
    if legend:
        fig.legend(list(legend.values()), list(legend.keys()), loc='lower center', ncol=min(6, len(legend)),
                   frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, path)
    plt.close(fig)


def parse_cultivars(text):
    return dict(item.split(':') for item in text.split(','))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--crop', default='maize', choices=list(CROPS),
                   help='crop: sets the default config, cultivars, MapSPAM area, seasons and --out')
    p.add_argument('--data', default=DEFAULT_DATA, help='validation table (csv)')
    p.add_argument('--config', default=None, help='crop model configuration (default: per --crop)')
    p.add_argument('--villages-shp', default='data/tb_limitealdeas.shp')
    p.add_argument('--villages-dir', default=None,
                   help='village spatial files, the soil is read from here (default: SPATIAL_INFO.villages_folderpath)')
    p.add_argument('--weather-dir', default=None,
                   help='village files to read weather and dem from (default: --villages-dir); point it to the '
                        'local village files when --villages-dir holds SoilGrids, so both runs share the weather')
    p.add_argument('--nitro', default=None, choices=['Y', 'N'],
                   help='DSSAT NITRO option: Y soil N balance (N-limited), N no N stress '
                        '(default: MANAGEMENT.nitrogen_simulation of the config, else Y)')
    p.add_argument('--cultivars', default=None, help='code:label,code:label,... (default: per --crop)')
    p.add_argument('--out', default=None, help='output folder, one per soil source (default: validation_<crop>)')
    p.add_argument('--work-dir', default='runs_val', help='DSSAT working folder (keep it short)')
    p.add_argument('--village-selection', default='coverage', choices=['coverage', 'pps', 'random'],
                   help='villages of a municipio/department, see the description')
    p.add_argument('--maize-coverage', '--crop-coverage', dest='maize_coverage', type=float, default=0.9,
                   help='coverage: share of the unit\'s crop area covered by the simulated villages')
    p.add_argument('--max-villages-per-unit', type=int, default=15, help='pps/random: villages drawn per unit')
    p.add_argument('--maize-area', '--crop-area', dest='maize_area', default=None,
                   help='csv with the crop area (column maize_ha) per GEOCODIGO, computed from MapSPAM when '
                        'missing (default: per --crop)')
    p.add_argument('--unknown-season', default=None,
                   help='bean: season of rows that state none (default: postrera, the main bean season)')
    p.add_argument('--levels', default=','.join(LEVELS), help='levels to evaluate')
    p.add_argument('--exclude-treatments', default=None,
                   help='regex on treatment_or_system of rows to leave out, e.g. "sin fert|0 kg N|testigo"')
    p.add_argument('--fertilizer-rows', default='prefer-unfertilized',
                   choices=['prefer-unfertilized', 'unfertilized-only', 'all'],
                   help='prefer-unfertilized: units with unfertilized rows use only those; unfertilized-only: '
                        'also drop the fertilized trial rows of the other units; all: average every row')
    p.add_argument('--aggregate-by-year', action='store_true',
                   help='one value per unit, season and year (default: per unit and season)')
    p.add_argument('--n-boot', type=int, default=2000, help='bootstrap replicates')
    p.add_argument('--select-on', default='department:census/survey',
                   help='level:source_type used to pick the best cultivar (summary.json)')
    p.add_argument('--plot-top', type=int, default=4, help='cultivars in the scatter plot (best RMSE first)')
    p.add_argument('--grain-moisture', type=float, default=0.0,
                   help='moisture fraction added to HWAH (dry grain) before comparing, e.g. 0.14')
    p.add_argument('--ncores', type=int, default=4, help='villages simulated in parallel')
    p.add_argument('--stagger', type=float, default=3.0, help='seconds between worker starts')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bin-path', default=None)
    p.add_argument('--dssat-path', default=None)
    p.add_argument('--skip-simulation', action='store_true', help='only use villages already in the cache')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--keep-runs', action='store_true', help='keep the DSSAT run folders')
    p.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    os.chdir(REPO)
    from omegaconf import OmegaConf
    spec = CROPS[args.crop]
    args.config = args.config or spec['config']
    args.cultivars = args.cultivars or spec['cultivars']
    args.maize_area = args.maize_area or spec['area_file']
    args.out = args.out or f'validation_{args.crop}'
    if args.villages_dir is None:
        args.villages_dir = OmegaConf.load(args.config).SPATIAL_INFO.villages_folderpath
    args.weather_dir = args.weather_dir or args.villages_dir
    if args.nitro is None:
        args.nitro = str(OmegaConf.load(args.config).MANAGEMENT.get('nitrogen_simulation', 'Y') or 'Y').upper()
    args.out = os.path.abspath(args.out)
    if args.worker:
        return worker(args.worker, args)
    args.passthrough = ['--crop', args.crop, '--config', args.config, '--villages-shp', args.villages_shp,
                        '--villages-dir', args.villages_dir, '--weather-dir', args.weather_dir, '--out', args.out,
                        '--work-dir', args.work_dir, '--nitro', args.nitro] + (['--bin-path', args.bin_path] if args.bin_path else []) \
        + (['--dssat-path', args.dssat_path] if args.dssat_path else []) + (['--keep-runs'] if args.keep_runs else [])
    os.makedirs(args.out, exist_ok=True)
    cultivars = parse_cultivars(args.cultivars)
    levels = args.levels.split(',')

    obs = load_observations(args.data, args.crop, args.unknown_season)
    villages = load_villages(args.villages_shp)
    maize_ha = maize_area_by_village(villages, args.maize_area, crop=args.crop)
    select = village_selector(maize_ha, args.village_selection, args.max_villages_per_unit, args.maize_coverage,
                              args.seed)
    obs = locate(obs, villages, select)
    treatment_out = (obs['treatment_or_system'].astype(str).str.contains(args.exclude_treatments, case=False)
                     if args.exclude_treatments else pd.Series(False, index=obs.index))
    fertilized = pd.Series(False, index=obs.index)
    if args.fertilizer_rows == 'prefer-unfertilized':
        fertilized = fertilized_at_unfertilized_unit(obs)
    elif args.fertilizer_rows == 'unfertilized-only':
        fertilized = (obs['source_type'] == 'field trial') & (obs['fertilization'] == 'fertilized')
    reason = np.select([obs['obs_t_ha'].isna(), obs['sim_season'] == 'irrigated', obs['level'].isna(),
                        ~obs['level'].isin(levels), redundant_annual(obs), fertilized, treatment_out],
                       ['value not numeric', 'irrigated (model is rainfed)', 'location not found',
                        'level not requested', 'annual row of a unit/year with season rows',
                        f'fertilized trial row (--fertilizer-rows {args.fertilizer_rows})', 'treatment excluded'],
                       default='')
    obs['excluded'] = reason
    obs.loc[obs['excluded'] != ''].drop(columns=['geocodes', 'village_weights']).to_csv(
        os.path.join(args.out, 'excluded_observations.csv'), index=False)
    obs = obs.loc[obs['excluded'] == ''].copy()
    print(obs.groupby(['level', 'source_type']).agg(rows=('obs_id', 'size'), units=('unit', 'nunique'),
                                                    villages=('geocodes', lambda c: len({g for x in c for g in x})))
          .to_string())

    geocodes = sorted({g for codes in obs['geocodes'] for g in codes})
    os.makedirs(os.path.join(args.out, 'cache'), exist_ok=True)
    check_cache_settings(os.path.join(args.out, 'cache'), args)
    migrate_cache(os.path.join(args.out, 'cache'))
    if not args.skip_simulation:
        run_simulations(geocodes, list(cultivars), args)
    sim = load_cache(geocodes, list(cultivars), args.out)
    failed = [g for g in geocodes if os.path.exists(os.path.join(args.out, 'cache', f'failed_{g}.txt'))
              and not os.path.exists(sim_file(os.path.join(args.out, 'cache'), g, list(cultivars)[0]))]
    if failed:
        print(f'{len(failed)} villages failed, see {os.path.join(args.out, "cache", "failed_*.txt")}')

    pred = predict(obs, sim, cultivars, args.grain_moisture, spec['annual'])
    pred['geocodes'] = pred['geocodes'].map(lambda c: ';'.join(c))
    pred['village_weights'] = pred['village_weights'].map(lambda w: ';'.join(f'{x:.1f}' for x in w))
    keep = ['obs_id', 'level', 'match_method', 'unit', 'geocodes', 'village_weights', 'department', 'municipio',
            'aldea_or_site', 'lat', 'lon', 'year', 'sim_season', 'planting_date', 'obs_cultivar',
            'treatment_or_system', 'fertilization', 'n_rate_kg_ha', 'source_type', 'source_citation', 'obs_t_ha', 'cultivar', 'cultivar_code',
            'pred_t_ha', 'n_villages_simulated', 'weight_simulated', 'year_matched']
    pred[keep].to_csv(os.path.join(args.out, 'observed_vs_predicted_rows.csv'), index=False)

    agg = aggregate(pred, args.aggregate_by_year)
    agg.to_csv(os.path.join(args.out, 'observed_vs_predicted.csv'), index=False)
    table = evaluate(agg, cultivars, args.n_boot, args.seed)
    table.to_csv(os.path.join(args.out, 'metrics.csv'), index=False)
    rows_table = (pred.dropna(subset=['pred_t_ha']).groupby(['level', 'source_type', 'cultivar'])
                  .apply(metrics, include_groups=False).reset_index())
    rows_table.to_csv(os.path.join(args.out, 'metrics_rows_not_aggregated.csv'), index=False)
    rmse_plot(table, os.path.join(args.out, 'rmse_by_cultivar.png'))

    sel_level, sel_source = args.select_on.split(':')
    ranking = table.loc[(table['level'] == sel_level) & (table['source_type'] == sel_source)]
    if ranking.empty:
        print(f'--select-on {args.select_on} has no data, ranking on all levels')
        ranking = table.groupby('cultivar', as_index=False)['rmse'].mean()
    top = ranking.sort_values('rmse')['cultivar'].head(args.plot_top).tolist()
    scatter_plots(agg, table, top, os.path.join(args.out, 'scatter_obs_vs_pred.png'),
                  f'{args.crop.capitalize()} yield, one value per unit and season: observed vs predicted '
                  f'({spec["model"]}, '
                  f'{"water- and N-limited" if args.nitro == "Y" else "water-limited, no N stress"})')

    best = {f'{lv}:{src}': t.iloc[0]['cultivar'] for (lv, src), t in table.groupby(['level', 'source_type'])}
    summary = {'crop': args.crop, 'selected_on': args.select_on, 'best_cultivar': top[0],
               'best_cultivar_code': {v: k for k, v in cultivars.items()}[top[0]],
               'best_by_subset': best, 'villages_dir': args.villages_dir, 'weather_dir': args.weather_dir,
               'nitro': args.nitro, 'village_selection': args.village_selection, 'fertilizer_rows': args.fertilizer_rows, 'maize_coverage': args.maize_coverage,
               'observations_used': int(pred['obs_id'].nunique()), 'aggregated_rows': int(agg['row_key'].nunique()),
               'villages_simulated': int(sim['geocode'].nunique()), 'villages_failed': failed}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    pd.set_option('display.width', 220)
    print(table[['level', 'source_type', 'cultivar', 'n', 'n_units', 'rmse', 'rmse_lo', 'rmse_hi', 'bias', 'r2',
                 'p_best', 'rank']].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f'outputs in {args.out}')


if __name__ == '__main__':
    main()
