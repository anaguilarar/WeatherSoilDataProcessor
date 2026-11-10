"""Collect the maize and bean validation outputs into tables for R (ggplot) and layers for QGIS.

Reads validation_<crop>/{local,soilgrids}_nitro{Y,N} and validation_<crop>/comparison_nitro{Y,N}/<cultivar>
(validate_maize_cultivars.py and compare_soil_sources.py outputs) and writes to --out:

  R (long tables, one row per crop, soil source, NITRO and cultivar)
    pred_obs_by_unit.csv          observed vs predicted per unit and season (the values the metrics use)
    pred_obs_by_observation.csv   observed vs predicted per observation row (trial treatment, year, ...)
    metrics.csv                   RMSE, bias, R2, ubRMSE, r with 95% bootstrap intervals per subset
    soil_source_comparison.csv    paired local vs SoilGrids differences (delta = SoilGrids - local)
  QGIS
    validation_gis.gpkg
      departments                 department polygons (dissolved from data/tb_limitealdeas.shp)
      observed_yield_department   polygons x crop, year, season and source type (filter in QGIS)
      validation_department       polygons x crop, soil, NITRO, cultivar, season: observed vs predicted
                                  (department census)
      observation_points          trial and census-by-site observations as points
    observed_yield_department_season_year.csv   the same table without geometry (join on dept_code)
    observation_points.csv        points with lon/lat (EPSG:4326)
  yield_sources.csv               one row per data source: crop, type, levels, years, observations

dept_code is the 2-digit code ('01'), dept_id the same as an integer (CSV readers drop the leading zero).
Point coordinates: reported coordinates when the source gives them, otherwise the centroid of the matched
villages weighted by their MapSPAM crop area (coord_method says which).

Usage (from the repository root)
  python scripts/export_validation_deliverables.py --out validation_deliverables
"""
import argparse
import glob
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_maize_cultivars import DEPARTMENTS, parse_years  # noqa: E402

CROPS = ('maize', 'bean')
RUNS = [(soil, nitro) for nitro in ('Y', 'N') for soil in ('local', 'soilgrids')]
UTM = 'EPSG:32616'
KEEP_ROW_COLS = ['obs_id', 'level', 'source_type', 'match_method', 'unit', 'department', 'municipio',
                 'aldea_or_site', 'year', 'sim_season', 'planting_date', 'obs_cultivar', 'treatment_or_system',
                 'fertilization', 'n_rate_kg_ha', 'source_citation', 'obs_t_ha', 'cultivar', 'cultivar_code',
                 'pred_t_ha', 'n_villages_simulated', 'weight_simulated', 'year_matched']


def run_dir(root, crop, soil, nitro):
    return os.path.join(root, f'validation_{crop}', f'{soil}_nitro{nitro}')


def best_cultivar(root, crop):
    with open(os.path.join(run_dir(root, crop, 'local', 'Y'), 'summary.json')) as f:
        return json.load(f)['best_cultivar_code']


def tag(df, crop, soil, nitro, best):
    df.insert(0, 'nitro', nitro)
    df.insert(0, 'soil_source', soil)
    df.insert(0, 'crop', crop)
    if 'cultivar_code' in df:
        df['best_cultivar'] = df['cultivar_code'] == best
    return df


def year_range(text):
    years = parse_years(text)
    return (min(years), max(years)) if years else (np.nan, np.nan)


def add_years(df, col='year'):
    rng = df[col].astype(str).map(year_range)
    df['year_start'] = [a for a, _ in rng]
    df['year_end'] = [b for _, b in rng]
    return df


def dept_of_geocodes(geocodes, weights):
    """Department code holding most of the matched villages' crop area, and the number of departments."""
    codes = str(geocodes).split(';')
    try:
        w = [float(x) for x in str(weights).split(';')]
    except ValueError:
        w = [1.0] * len(codes)
    by_dept = pd.Series(w if len(w) == len(codes) else [1.0] * len(codes), index=[c[:2] for c in codes])
    by_dept = by_dept.groupby(level=0).sum()
    return by_dept.idxmax(), len(by_dept)


def read_rows(root, crop, soil, nitro, best):
    fn = os.path.join(run_dir(root, crop, soil, nitro), 'observed_vs_predicted_rows.csv')
    df = pd.read_csv(fn, dtype={'cultivar_code': str, 'unit': str, 'geocodes': str, 'village_weights': str})
    return tag(df, crop, soil, nitro, best)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='.', help='folder holding validation_maize and validation_bean')
    p.add_argument('--villages', default='data/tb_limitealdeas.shp')
    p.add_argument('--out', default='validation_deliverables')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    best = {c: best_cultivar(args.root, c) for c in CROPS}

    # ---- R tables --------------------------------------------------------------------------------------
    units, rows, metrics, comparisons = [], [], [], []
    for crop in CROPS:
        for soil, nitro in RUNS:
            folder = run_dir(args.root, crop, soil, nitro)
            u = pd.read_csv(os.path.join(folder, 'observed_vs_predicted.csv'), dtype={'cultivar_code': str, 'unit': str})
            units.append(tag(u, crop, soil, nitro, best[crop]))
            r = read_rows(args.root, crop, soil, nitro, best[crop])
            rows.append(r)
            m = pd.read_csv(os.path.join(folder, 'metrics.csv'), dtype={'cultivar_code': str})
            metrics.append(tag(m, crop, soil, nitro, best[crop]))
        for nitro in ('Y', 'N'):
            for fn in sorted(glob.glob(os.path.join(args.root, f'validation_{crop}', f'comparison_nitro{nitro}', '*',
                                                    'soil_source_comparison.csv'))):
                c = pd.read_csv(fn, dtype={'cultivar_code': str})
                c.insert(0, 'nitro', nitro)
                c.insert(0, 'crop', crop)
                c['best_cultivar'] = c['cultivar_code'] == best[crop]
                comparisons.append(c)

    units = pd.concat(units, ignore_index=True)
    units['error_t_ha'] = units['pred_t_ha'] - units['obs_t_ha']
    units = add_years(units, 'years')
    units.to_csv(os.path.join(args.out, 'pred_obs_by_unit.csv'), index=False)

    rows = pd.concat(rows, ignore_index=True)
    dept = rows.drop_duplicates(['crop', 'obs_id']).set_index(['crop', 'obs_id'])
    dept = dept.apply(lambda r: dept_of_geocodes(r['geocodes'], r['village_weights']), axis=1)
    rows['dept_code'] = [dept[(c, o)][0] for c, o in zip(rows['crop'], rows['obs_id'])]
    rows['n_departments'] = [dept[(c, o)][1] for c, o in zip(rows['crop'], rows['obs_id'])]
    keep = ['crop', 'soil_source', 'nitro'] + KEEP_ROW_COLS + ['best_cultivar', 'dept_code', 'n_departments']
    obs_rows = rows[keep].copy()
    obs_rows['error_t_ha'] = obs_rows['pred_t_ha'] - obs_rows['obs_t_ha']
    obs_rows = add_years(obs_rows)
    obs_rows.to_csv(os.path.join(args.out, 'pred_obs_by_observation.csv'), index=False)

    pd.concat(metrics, ignore_index=True).to_csv(os.path.join(args.out, 'metrics.csv'), index=False)
    pd.concat(comparisons, ignore_index=True).to_csv(os.path.join(args.out, 'soil_source_comparison.csv'),
                                                    index=False)

    # ---- observed yields (one row per observation: any run and cultivar holds the same observed value) --
    obs = obs_rows.loc[(obs_rows['soil_source'] == 'local') & (obs_rows['nitro'] == 'Y') & obs_rows['best_cultivar']]
    obs = obs.drop(columns=['soil_source', 'nitro', 'cultivar', 'cultivar_code', 'pred_t_ha', 'error_t_ha',
                            'best_cultivar', 'n_villages_simulated', 'weight_simulated', 'year_matched'])
    obs['department_name'] = obs['dept_code'].map(DEPARTMENTS)
    yield_dept = (obs.groupby(['crop', 'dept_code', 'department_name', 'year', 'year_start', 'year_end',
                               'sim_season', 'source_type', 'level'], dropna=False)
                  .agg(obs_t_ha_mean=('obs_t_ha', 'mean'), obs_t_ha_min=('obs_t_ha', 'min'),
                       obs_t_ha_max=('obs_t_ha', 'max'), n_obs=('obs_id', 'size'),
                       multi_department=('n_departments', lambda s: bool((s > 1).any())),
                       sources=('source_citation', lambda s: ' | '.join(sorted(set(s.dropna())))))
                  .reset_index().rename(columns={'sim_season': 'season', 'level': 'spatial_level'}))
    yield_dept.insert(1, 'dept_id', yield_dept['dept_code'].astype(int))   # numeric key for QGIS joins
    yield_dept.to_csv(os.path.join(args.out, 'observed_yield_department_season_year.csv'), index=False)

    sources = (obs.groupby(['crop', 'source_citation', 'source_type'])
               .agg(levels=('level', lambda s: ';'.join(sorted(set(s)))),
                    year_min=('year_start', 'min'), year_max=('year_end', 'max'),
                    departments=('department_name', lambda s: ';'.join(sorted(set(s.dropna())))),
                    n_obs=('obs_id', 'size'))
               .reset_index())
    sources.to_csv(os.path.join(args.out, 'yield_sources.csv'), index=False)

    # ---- geometry ---------------------------------------------------------------------------------------
    villages = gpd.read_file(args.villages)[['GEOCODIGO', 'geometry']]
    villages['GEOCODIGO'] = villages['GEOCODIGO'].astype(str).str.zfill(6)
    villages['dept_code'] = villages['GEOCODIGO'].str[:2]
    departments = villages.dissolve('dept_code').reset_index()[['dept_code', 'geometry']]
    departments = departments.loc[departments['dept_code'].isin(DEPARTMENTS)]   # drops the disputed zone
    departments['department_name'] = departments['dept_code'].map(DEPARTMENTS)
    departments['dept_id'] = departments['dept_code'].astype(int)
    cent = villages.to_crs(UTM).set_index('GEOCODIGO').geometry.centroid

    def centroid(geocodes, weights):
        codes = [c for c in str(geocodes).split(';') if c in cent.index]
        if not codes:
            return np.nan, np.nan
        w = np.array([float(x) for x in str(weights).split(';')])
        allc = str(geocodes).split(';')
        w = np.array([w[allc.index(c)] for c in codes]) if len(w) == len(allc) else np.ones(len(codes))
        w = w if w.sum() > 0 else np.ones(len(codes))
        x = np.average(cent.loc[codes].x, weights=w)
        y = np.average(cent.loc[codes].y, weights=w)
        pt = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=UTM).to_crs('EPSG:4326').iloc[0]
        return pt.x, pt.y

    first = rows.loc[(rows['soil_source'] == 'local') & (rows['nitro'] == 'Y') & rows['best_cultivar']]
    pts = obs.loc[obs['level'] != 'department'].merge(
        first[['crop', 'obs_id', 'lat', 'lon', 'geocodes', 'village_weights']], on=['crop', 'obs_id'])
    reported = pts['lat'].notna() & pts['lon'].notna()
    xy = [(lo, la) if rep else centroid(g, w)
          for rep, lo, la, g, w in zip(reported, pts['lon'], pts['lat'], pts['geocodes'], pts['village_weights'])]
    pts['lon'] = [a for a, _ in xy]
    pts['lat'] = [b for _, b in xy]
    pts['coord_method'] = np.where(reported, 'reported',
                                   'centroid of matched villages (crop-area weighted): ' + pts['match_method'])
    pts['n_villages_matched'] = pts['geocodes'].str.count(';') + 1
    pts = pts.drop(columns=['geocodes', 'village_weights'])
    pts.to_csv(os.path.join(args.out, 'observation_points.csv'), index=False)
    pts_gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts['lon'], pts['lat']), crs='EPSG:4326')

    census = units.loc[(units['level'] == 'department') & (units['source_type'] == 'census/survey')].copy()
    name_to_code = {v.lower(): k for k, v in DEPARTMENTS.items()}
    census['dept_code'] = census['unit'].str.lower().map(name_to_code)
    census = census.drop(columns=['row_key'])

    gpkg = os.path.join(args.out, 'validation_gis.gpkg')
    if os.path.exists(gpkg):
        os.remove(gpkg)
    departments.to_file(gpkg, layer='departments', driver='GPKG')
    departments.merge(yield_dept, on=['dept_code', 'dept_id', 'department_name']).to_file(
        gpkg, layer='observed_yield_department', driver='GPKG')
    departments.merge(census, on='dept_code').to_file(gpkg, layer='validation_department', driver='GPKG')
    pts_gdf.to_file(gpkg, layer='observation_points', driver='GPKG')

    print(f'outputs in {args.out}')
    for fn in sorted(os.listdir(args.out)):
        print(f'  {fn}')
    print(f'points: {len(pts)} ({reported.sum()} with reported coordinates); best cultivars: {best}')


if __name__ == '__main__':
    main()
