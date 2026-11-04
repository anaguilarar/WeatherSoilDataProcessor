"""Derive total nitrogen of the local (SH) village soil files from their organic carbon.

Village files built from an SH cube made before ``sh_soil_to_soilgrids.py --nitrogen`` existed carry
SoilGrids nitrogen next to SH organic carbon (C:N about 4-8 in the topsoil). This script rewrites only the
soil file of each village with N = soc / C:N at the SH depths (0-5, 5-15, 15-30 cm), where C:N is the
SoilGrids ratio of the same pixel and depth (nearest SoilGrids cell), or --cn-ratio where SoilGrids has
none or with --cn-fixed. Deeper layers, which are SoilGrids in both sources, are left as they are.

The output folder only holds soil_.zip per village: run the simulations with
  --villages-dir <out-dir> --weather-dir <the original village folder>

Usage (from the repository root)
  python scripts/fix_village_nitrogen.py --villages-dir data/village_sp_files \
      --soilgrids <...>/hnd_soilgrids_data.nc --out-dir data/village_sp_files_localN
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import xarray as xr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sh_soil_to_soilgrids import CN_LIMITS, nitrogen_from_soc, open_reference, soilgrids_on_sh_grid  # noqa: E402

SH_DEPTHS = ('0-5', '5-15', '15-30')
SCALE_VALUE, NODATA = 100, -9999


def write_soil_zip(soil, zip_path):
    """Same layout as the village files (int32 x 100, -9999, zipped); values rounded, not truncated."""
    soil = soil.copy()
    for v in soil.data_vars:
        a = soil[v].values
        soil[v] = (soil[v].dims, np.where(np.isfinite(a), np.round(a * SCALE_VALUE), NODATA).astype(np.int32))
        soil[v].encoding = {'zlib': True, 'coordinates': 'spatial_ref'}
    soil.attrs.update(nodata=NODATA, dtype='int', scale_values_factor=SCALE_VALUE, count=len(soil.data_vars))
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    nc_path = zip_path.replace('.zip', '.nc')
    soil.to_netcdf(nc_path, engine='netcdf4')
    with zipfile.ZipFile(zip_path + '.tmp', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(nc_path, os.path.basename(nc_path))
    os.replace(zip_path + '.tmp', zip_path)
    os.remove(nc_path)


def fix_village(soil_zip, out_zip, ref, cn_fixed, cn_ratio, tmp):
    import rioxarray  # noqa: F401
    from spatialdata.utils import read_compressed_xarray
    shutil.copy2(soil_zip, os.path.join(tmp, 'soil_.zip'))
    read_compressed_xarray(os.path.join(tmp, 'soil_.zip'), tmp)          # int x 100 -> float, nodata -> NaN
    with xr.open_dataset(os.path.join(tmp, 'soil_.nc')) as ds:
        soil = ds.load()
    attrs = dict(soil.attrs)
    template = soil['soc'].isel(depth=0).rio.write_crs(attrs.get('crs') or soil.rio.crs)
    stats = {}
    for depth in SH_DEPTHS:
        if depth not in soil.depth.values:
            continue
        if cn_fixed:
            cn = cn_ratio
        else:
            sg_soc = soilgrids_on_sh_grid(ref['soc'].sel(depth=depth), template)
            sg_n = soilgrids_on_sh_grid(ref['nitrogen'].sel(depth=depth), template)
            cn = (sg_soc * 10.0 / sg_n.where(sg_n > 0)).clip(*CN_LIMITS).fillna(cn_ratio)
            cn = cn.drop_vars(['spatial_ref', 'depth'], errors='ignore')
        soc = soil['soc'].sel(depth=depth)
        new_n = nitrogen_from_soc(soc, cn).where(soc.notnull())
        old_n = soil['nitrogen'].sel(depth=depth).copy()
        soil['nitrogen'].loc[{'depth': depth}] = new_n.where(new_n.notnull(), old_n).values
        fallback = 1.0 if cn_fixed else float(np.isnan(sg_n.values[np.isfinite(soc.values)]).mean())
        stats[depth] = (float((soc * 10 / old_n).median()),
                        float((soc * 10 / soil['nitrogen'].sel(depth=depth)).median()), fallback)
    soil.attrs = attrs
    soil.attrs['nitrogen'] = (f'SH soc / C:N {cn_ratio}' if cn_fixed else
                              'SH soc / SoilGrids C:N (same pixel and depth)') + f' at {", ".join(SH_DEPTHS)} cm'
    write_soil_zip(soil, out_zip)
    for fn in os.listdir(tmp):
        os.remove(os.path.join(tmp, fn))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--villages-dir', required=True, help='local village files (<GEOCODIGO>/soil_.zip)')
    p.add_argument('--soilgrids', required=True, help='SoilGrids datacube (hnd_soilgrids_data.nc)')
    p.add_argument('--out-dir', required=True, help='folder for the corrected soil files')
    p.add_argument('--cn-fixed', action='store_true', help='use --cn-ratio everywhere instead of SoilGrids C:N')
    p.add_argument('--cn-ratio', type=float, default=10.0, help='C:N where SoilGrids has none (or with --cn-fixed)')
    p.add_argument('--geocodes', nargs='*', default=None, help='only these villages (test)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    ref = open_reference(args.soilgrids)
    codes = args.geocodes or sorted(d for d in os.listdir(args.villages_dir)
                                    if os.path.exists(os.path.join(args.villages_dir, d, 'soil_.zip')))
    tmp = tempfile.mkdtemp(prefix='fixn_')
    failed = []
    for i, g in enumerate(codes, 1):
        out_zip = os.path.join(args.out_dir, g, 'soil_.zip')
        if os.path.exists(out_zip) and not args.overwrite:
            continue
        try:
            stats = fix_village(os.path.join(args.villages_dir, g, 'soil_.zip'), out_zip, ref, args.cn_fixed,
                                args.cn_ratio, tmp)
            if i % 100 == 1 or args.geocodes:
                print(f'[{i}/{len(codes)}] {g} median C:N old -> new (share of pixels with --cn-ratio) ' +
                      ', '.join(f'{d}: {a:.1f} -> {b:.1f} ({c:.0%})' for d, (a, b, c) in stats.items()), flush=True)
        except Exception as exc:  # keep going
            failed.append(g)
            print(f'[{i}/{len(codes)}] {g} FAILED: {exc!r}', flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'done: {len(codes) - len(failed)} villages in {args.out_dir}, failed {len(failed)} {failed[:20]}')


if __name__ == '__main__':
    main()
