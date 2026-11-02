"""Clip weather, soil and DEM to every village (with a buffer) for any crop model.

Offline step that replaces the per-request extraction in ``SpatialCM``.  For
every village it writes the files that ``modeladocultivo.py`` / ``cm_code.py``
already pick up from ``SPATIAL_INFO.villages_folderpath``::

    <out>/<GEOCODIGO>/
        weather_.zip   weather_.nc  (date, y, x)   precipitation, tmax, tmin, srad, wn, vp
        soil_.zip      soil_.nc     (depth, y, x)  every variable of the soil cube
        dem_.zip       dem_.nc      (y, x)         every variable of the DEM cube (if --dem)

All files share one grid (EPSG:4326) and use the format of
``spatialdata.utils.compress_xarray_dataset``: int32 values x ``scale_values_factor``
(100), nodata -9999, zlib, zipped.  Read them back with
``spatialdata.utils.read_compressed_xarray``.

Grid and masking
----------------
* The grid is the weather grid refined ``--scale-factor`` times (default 10,
  as ``SPATIAL_INFO.scale_factor``), so every cell sits inside exactly one
  weather pixel and takes its value (nearest), with no interpolation.
* Soil and DEM are aggregated onto that grid with an area average of the source pixels.
* All cubes are masked with the village polygon buffered by ``--buffer-m``
  metres (all touched), not with the exact polygon, so edge cells are not lost
  to the coarse resolution.  Cells outside the buffer are nodata.
* Weather units are converted with ``WeatherTransformer`` (K->C, J->MJ, hPa->kPa),
  the same step the pipeline runs before exporting.  Soil and DEM keep the source units.
* A re-run only writes the files a village is missing, so adding ``--dem`` later
  adds ``dem_.zip`` without rebuilding weather and soil.

Usage
-----
python build_village_inputs.py \
    --villages data/tb_limitealdeas.shp --id-column GEOCODIGO \
    --weather climate_hnd_1990_1990_2025.nc \
    --soil hnd_sh_soilgrids_format.nc \
    --dem terrain_.nc \
    --out village_sp_files --buffer-m 1000 --ncores 8 [--codes 070510 032118]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin

from crop_modeling.utils.u_weather import WeatherTransformer

logger = logging.getLogger(__name__)

#: Weather names in the source cube -> names the pipeline expects.
WEATHER_NAMES = {
    "pr": "precipitation", "precipitation": "precipitation",
    "tasmax": "tmax", "tmax": "tmax",
    "tasmin": "tmin", "tmin": "tmin",
    "rsds": "srad", "srad": "srad",
    "vp": "vp",
    "sfcWind": "wn", "wn": "wn",
}
SCALE_VALUE = 100
NODATA = -9999
SOURCE_PAD_PIXELS = 3

_WEATHER: xr.Dataset | None = None
_SOIL: xr.Dataset | None = None
_DEM: xr.Dataset | None = None


# ---------------------------------------------------------------------------
# Data access (opened lazily once per worker process)
# ---------------------------------------------------------------------------

def _open_raster_cube(path: str) -> xr.Dataset:
    """Open a soil/DEM cube lazily, keeping only (y, x) variables, with its CRS set."""
    ds = xr.open_dataset(path, chunks={}, decode_coords="all")
    crs = ds.rio.crs or ds.attrs.get("crs")
    if crs is None:
        raise ValueError(f"No CRS found in {path} (spatial_ref or attrs['crs'])")
    return ds[[v for v in ds.data_vars if {"y", "x"} <= set(ds[v].dims)]].rio.write_crs(crs)


def _open_sources(weather_path: str, soil_path: str, dem_path: str | None = None) -> None:
    global _WEATHER, _SOIL, _DEM
    w = xr.open_dataset(weather_path, chunks={}, decode_coords="all")
    w = w.rename({k: v for k, v in WEATHER_NAMES.items() if k in w.data_vars and k != v})
    if "time" in w.dims:
        w = w.rename({"time": "date"})
    if w.rio.crs is None:
        w = w.rio.write_crs("EPSG:4326")
    if not w.rio.crs.is_geographic:
        raise ValueError("the weather cube must be in geographic coordinates (EPSG:4326)")
    _WEATHER = w[[v for v in dict.fromkeys(WEATHER_NAMES.values()) if v in w.data_vars]]

    _SOIL = _open_raster_cube(soil_path)
    _DEM = _open_raster_cube(dem_path) if dem_path else None


# ---------------------------------------------------------------------------
# One village
# ---------------------------------------------------------------------------

def buffer_geometry(geom_4326: gpd.GeoSeries, buffer_m: float) -> gpd.GeoSeries:
    """Buffer in metres, using an equidistant projection centred on the village."""
    if not buffer_m:
        return geom_4326
    c = geom_4326.union_all().centroid
    local = f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m"
    return geom_4326.to_crs(local).buffer(buffer_m).to_crs("EPSG:4326")


def village_grid(bounds, weather: xr.Dataset, scale_factor: int):
    """Fine grid aligned to the weather pixel edges that covers ``bounds``."""
    rx = float(abs(weather.x[1] - weather.x[0]))
    ry = float(abs(weather.y[1] - weather.y[0]))
    x_edge0 = float(weather.x.min()) - rx / 2   # west edge of the weather grid
    y_edge0 = float(weather.y.max()) + ry / 2   # north edge
    fx, fy = rx / scale_factor, ry / scale_factor
    xmin, ymin, xmax, ymax = bounds
    c0 = int(np.floor((xmin - x_edge0) / fx))
    c1 = int(np.ceil((xmax - x_edge0) / fx))
    r0 = int(np.floor((y_edge0 - ymax) / fy))
    r1 = int(np.ceil((y_edge0 - ymin) / fy))
    west, north = x_edge0 + c0 * fx, y_edge0 - r0 * fy
    width, height = c1 - c0, r1 - r0
    x = west + fx * (np.arange(width) + 0.5)
    y = north - fy * (np.arange(height) + 0.5)
    return x, y, from_origin(west, north, fx, fy)


def clip_weather(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> xr.Dataset:
    """Weather on the fine grid: each cell takes its parent weather pixel."""
    ix = _WEATHER.indexes["x"].get_indexer(x, method="nearest")
    iy = _WEATHER.indexes["y"].get_indexer(y, method="nearest")
    sub = _WEATHER.isel(x=slice(ix.min(), ix.max() + 1), y=slice(iy.min(), iy.max() + 1))
    sub = sub.load()
    jx, jy = ix - ix.min(), iy - iy.min()
    out = {}
    for v in sub.data_vars:
        a = sub[v].transpose("date", "y", "x").values.astype("float32")
        a = a[:, jy][:, :, jx]
        a[:, ~mask] = np.nan
        out[v] = (("date", "y", "x"), a)
    dates = pd.DatetimeIndex(sub["date"].values).strftime("%Y%m%d").values
    ds = xr.Dataset(out, coords={"date": dates, "y": y, "x": x})
    return WeatherTransformer()(ds)


def clip_cube(source: xr.Dataset, x: np.ndarray, y: np.ndarray, transform,
              mask: np.ndarray) -> xr.Dataset:
    """Soil or DEM cube aggregated (area average) onto the fine grid, any source CRS."""
    src_crs = source.rio.crs
    # densified box edges, so the window also covers curved edges in projected CRSs
    edge = np.linspace(0, 1, 11)
    bx = np.concatenate([x.min() + (x.max() - x.min()) * edge, np.full(11, x.max()),
                         x.min() + (x.max() - x.min()) * edge, np.full(11, x.min())])
    by = np.concatenate([np.full(11, y.min()), y.min() + (y.max() - y.min()) * edge,
                         np.full(11, y.max()), y.min() + (y.max() - y.min()) * edge])
    bounds = gpd.GeoSeries.from_xy(bx, by, crs="EPSG:4326")
    xmin, ymin, xmax, ymax = bounds.to_crs(src_crs).total_bounds
    sx, sy = (abs(r) for r in source.rio.resolution())
    # one fine cell of margin (in source units) plus a few source pixels
    cell = max(abs(transform.a), abs(transform.e))
    p0, p1 = gpd.GeoSeries.from_xy([x.mean(), x.mean() + cell], [y.mean(), y.mean() + cell],
                                   crs="EPSG:4326").to_crs(src_crs)
    cx, cy = abs(p1.x - p0.x), abs(p1.y - p0.y)
    pad_x, pad_y = sx * SOURCE_PAD_PIXELS + cx, sy * SOURCE_PAD_PIXELS + cy
    ysl = (slice(ymax + pad_y, ymin - pad_y) if source.y[0] > source.y[-1]
           else slice(ymin - pad_y, ymax + pad_y))
    sub = source.sel(x=slice(xmin - pad_x, xmax + pad_x), y=ysl).load()
    out = {}
    for v in sub.data_vars:
        da = sub[v].astype("float32")
        da = da.where(np.abs(da) < 1e30)  # float nodata (e.g. -3.4e38) not decoded by the file
        da = da.rio.write_crs(src_crs).rio.write_nodata(np.nan, encoded=False)
        da = da.rio.reproject("EPSG:4326", shape=(y.size, x.size), transform=transform,
                              resampling=Resampling.average, nodata=np.nan)
        a = da.values
        a[..., ~mask] = np.nan
        out[v] = (da.dims, a)
    coords = {"y": y, "x": x}
    if "depth" in sub.dims:
        coords["depth"] = sub["depth"].values
    return xr.Dataset(out, coords=coords)


def write_compressed(ds: xr.Dataset, nc_path: str, transform) -> None:
    """Same layout as ``compress_xarray_dataset`` (int32 x 100, -9999, zipped).

    Values are rounded (not truncated) and converted one variable at a time.
    The zip is written to a temporary name and renamed, so a half-written file
    is never mistaken for a finished one.
    """
    for v in ds.data_vars:
        a = ds[v].values
        a = np.where(np.isfinite(a), np.round(a * SCALE_VALUE), NODATA).astype(np.int32)
        ds[v] = (ds[v].dims, a)
    ds = ds.rio.write_crs("EPSG:4326").rio.write_transform(transform)
    ds.attrs.update(crs="EPSG:4326", nodata=NODATA, dtype="int", width=ds.sizes["x"],
                    height=ds.sizes["y"], count=len(ds.data_vars),
                    scale_values_factor=SCALE_VALUE, transform=list(transform)[:6])
    # list spatial_ref in each variable's "coordinates" so a plain xr.open_dataset (as used by
    # read_compressed_xarray and create_texture_layer) keeps it as a coordinate and the CRS survives
    for v in ds.data_vars:
        ds[v].encoding.update(zlib=True, coordinates="spatial_ref")
    ds.to_netcdf(nc_path, engine="netcdf4")
    zip_path = nc_path.replace(".nc", ".zip")
    with zipfile.ZipFile(zip_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(nc_path, os.path.basename(nc_path))
    os.replace(zip_path + ".tmp", zip_path)
    os.remove(nc_path)


def build_village(geocode: str, geom_4326: gpd.GeoSeries, out_root: str,
                  buffer_m: float = 1000, scale_factor: int = 10) -> dict:
    vdir = os.path.join(out_root, str(geocode))
    names = ["weather", "soil"] + (["dem"] if _DEM is not None else [])
    todo = [n for n in names if not os.path.exists(os.path.join(vdir, f"{n}_.zip"))]
    if not todo:
        return {"geocode": geocode, "status": "cached"}
    os.makedirs(vdir, exist_ok=True)

    # the grid only depends on the geometry and the weather grid, so files written
    # in separate runs (e.g. dem_ added later) still line up
    buffered = buffer_geometry(geom_4326, buffer_m)
    x, y, transform = village_grid(buffered.total_bounds, _WEATHER, scale_factor)
    shape = (y.size, x.size)
    inside = geometry_mask(buffered.geometry, shape, transform, all_touched=True, invert=True)
    exact = geometry_mask(geom_4326.geometry, shape, transform, all_touched=True, invert=True)
    log = {"geocode": geocode, "status": "ok", "written": "+".join(todo), "ny": y.size, "nx": x.size,
           "cells_buffer": int(inside.sum()), "cells_village": int(exact.sum())}

    cubes = {}
    if "weather" in todo:
        cubes["weather"] = clip_weather(x, y, inside)
        ok = np.isfinite(cubes["weather"][list(cubes["weather"].data_vars)[0]].isel(date=0).values)
        log["village_cells_with_weather"] = int((ok & exact).sum())
    if "soil" in todo:
        soil = cubes["soil"] = clip_cube(_SOIL, x, y, transform, inside)
        top = soil["sand"].isel(depth=0) if "depth" in soil.dims else soil["sand"]
        ok = np.isfinite(top.values)
        log["village_cells_with_soil"] = int((ok & exact).sum())
        # cells with topsoil but a missing deeper layer (DSSAT stops on these profiles)
        if "depth" in soil.dims:
            all_depths = np.isfinite(soil["sand"].values).all(axis=0)
            log["cells_incomplete_profile"] = int((ok & ~all_depths).sum())
    if "dem" in todo:
        dem = cubes["dem"] = clip_cube(_DEM, x, y, transform, inside)
        ok = np.isfinite(dem[list(dem.data_vars)[0]].values)
        log["village_cells_with_dem"] = int((ok & exact).sum())

    empty = [n for n in todo if log.get(f"village_cells_with_{n}", 0) == 0]
    if empty:
        return {**log, "status": f"no data in the village for: {', '.join(empty)}"}
    for n, cube in cubes.items():
        write_compressed(cube, os.path.join(vdir, f"{n}_.nc"), transform)
    return log


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _worker(args):
    geocode, wkt, out_root, buffer_m, scale_factor = args
    geom = gpd.GeoSeries.from_wkt([wkt], crs="EPSG:4326")
    try:
        return build_village(geocode, geom, out_root, buffer_m, scale_factor)
    except Exception as exc:  # keep the batch going; failures go to the log
        return {"geocode": geocode, "status": f"error: {exc!r}"}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--villages", required=True)
    p.add_argument("--id-column", default="GEOCODIGO")
    p.add_argument("--weather", required=True)
    p.add_argument("--soil", required=True)
    p.add_argument("--dem", help="DEM cube (dem, slope, ...); omit to skip dem_.zip")
    p.add_argument("--out", required=True)
    p.add_argument("--buffer-m", type=float, default=1000,
                   help="buffer around the village polygon in metres (default 1000)")
    p.add_argument("--scale-factor", type=int, default=10,
                   help="grid refinement of the weather pixel (default 10, ~550 m)")
    p.add_argument("--codes", nargs="*", help="only these village codes")
    p.add_argument("--ncores", type=int, default=4)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    villages = gpd.read_file(a.villages).to_crs("EPSG:4326")
    villages = villages[villages[a.id_column].notna()]
    if a.codes:
        villages = villages[villages[a.id_column].astype(str).isin(a.codes)]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(str(r[a.id_column]), r.geometry.wkt, a.out, a.buffer_m, a.scale_factor)
            for _, r in villages.iterrows()]
    logger.info("%d villages -> %s (buffer %.0f m)", len(jobs), a.out, a.buffer_m)

    results = []
    # small batches: one village per task so every worker gets work
    chunksize = max(1, min(4, len(jobs) // (a.ncores * 8)))
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=a.ncores, initializer=_open_sources, initargs=(a.weather, a.soil, a.dem)) as ex:
        for i, res in enumerate(ex.map(_worker, jobs, chunksize=chunksize), 1):
            results.append(res)
            if res["status"] not in ("ok", "cached"):
                logger.warning("%s: %s", res["geocode"], res["status"])
            if i % 50 == 0 or i == len(jobs):
                logger.info("%d/%d done", i, len(jobs))
    pd.DataFrame(results).to_csv(os.path.join(a.out, "build_log.csv"), index=False)


if __name__ == "__main__":
    main()
