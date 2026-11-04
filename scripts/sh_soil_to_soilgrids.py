"""
Convert the Honduras national soil datacube (Suelos de Honduras, SH) into a
SoilGrids-compatible datacube that ``DSSATSoil_fromSOILGRIDS`` can read.

The output keeps the SH ~30 m grid (EPSG:4326, no downsampling) and copies the
depth labels and *raw SoilGrids units* of a reference SoilGrids cube, because
``crop_modeling/dssat/soil.py`` applies the SoilGrids scale factors
unconditionally (clay ÷10, bdod ÷100, soc ÷100 …).  Layers that only SoilGrids
provides are resampled (nearest neighbour) onto the SH grid.

Variable sources
----------------
=========  ==========================  ===========================  ==============
output     SoilGrids unit              SH source                    conversion
=========  ==========================  ===========================  ==============
sand       g/kg                        SAND %                       × 10
clay       g/kg                        CLAY %                       × 10
silt       g/kg                        SILT %                       × 10
bdod       cg/cm³                      DAP g cm⁻³                   × 100
phh2o      pH × 10                     pH                           × 10
soc        dg/kg                       MO %                         ÷ 1.724 × 100
cec        mmol(c)/kg                  CIC cmol(+) kg⁻¹             × 10
k          cmol(+)/kg (not SoilGrids)  K cmol(+) kg⁻¹               as is
p          mg/kg (not SoilGrids)       P mg kg⁻¹                    as is
wv0010     10⁻³ cm³/cm³ (θ at 10 kPa)  — (SoilGrids) or PTF         see ``water``
wv0033     10⁻³ cm³/cm³ (θ at 33 kPa)  — (SoilGrids) or PTF         see ``water``
wv1500     10⁻³ cm³/cm³ (θ at 1.5 MPa) — (SoilGrids) or PTF         see ``water``
cfvo       cm³/dm³                     — (SoilGrids, or 0)           see ``cfvo``
nitrogen   cg/kg                       MO % (via soc)               soc ÷ C:N, see ``nitrogen``
=========  ==========================  ===========================  ==============

Total nitrogen is derived from the SH organic carbon, N = soc / (C:N), so the
profile is consistent with the local organic matter.  By default the C:N ratio
is SoilGrids' own ratio at the same pixel and depth (``--nitrogen
cn-soilgrids``); ``--nitrogen cn-fixed --cn-ratio 10`` uses one ratio, and
``--nitrogen soilgrids`` copies SoilGrids nitrogen as earlier versions did
(it leaves the SH topsoil with C:N ≈ 4-8, because SH organic carbon is lower
than SoilGrids').

SH has no coarse-fragment layer: ``--cfvo soilgrids`` (default) takes it from
SoilGrids, ``--cfvo zero`` sets 0 at the SH depths so the topsoil is entirely
SH (no stones assumed).

The SH ``cec`` layer must hold cation exchange capacity (CIC), not
electrical conductivity; a warning is logged if its values look like CE.
``k`` and ``p`` have no SoilGrids counterpart and are kept in their original
units (the DSSAT soil writer ignores them).

The SH maps hold one topsoil layer replicated over 0-5/5-15/15-30 cm.  Depths
deeper than that (30-60 cm) are taken from SoilGrids; ``k`` and ``p`` repeat
the topsoil value there, because the DSSAT reader drops any depth that has a
NaN in any variable.  SoilGrids has gaps (mostly urban areas) where SH has
data; at those deeper depths each gap cell takes the value of the nearest valid
SoilGrids cell (up to ``--deep-fill-km``), otherwise the profile of those
pixels would end at 30 cm.

Memory: variables are computed and written one at a time, so the peak is about
one variable at full resolution — roughly (number of depths + 2) x 1.1 GB for
the whole of Honduras.  The output is written uncompressed-equivalent ~55 GB
(12 variables x 4 depths x 287 M pixels, float32) and zlib-compressed on disk.

Example
-------
    # quick local test on a small box
    python sh_soil_to_soilgrids.py --sh soil_hnd_sh.nc --ref hnd_soilgrids_data.nc ^
        --out test.nc --bbox -87.3 14.0 -87.1 14.2

    # full run (on a machine with >= 16 GB RAM)
    python sh_soil_to_soilgrids.py --sh soil_hnd_sh.nc --ref hnd_soilgrids_data.nc ^
        --out hnd_sh_soilgrids_format.nc
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#: SH variable -> (SoilGrids name, factor to raw SoilGrids units)
SH_TO_SOILGRIDS: dict[str, tuple[str, float]] = {
    "sand": ("sand", 10.0),
    "clay": ("clay", 10.0),
    "silt": ("silt", 10.0),
    "bdod": ("bdod", 100.0),
    "phh2o": ("phh2o", 10.0),
    "som": ("soc", 100.0 / 1.724),  # % organic matter -> % C (van Bemmelen) -> dg/kg
    "cec": ("cec", 10.0),  # CIC cmol(+)/kg -> mmol(c)/kg
    "k": ("k", 1.0),  # no SoilGrids counterpart: original units
    "p": ("p", 1.0),
}
#: Units written to each output variable's attributes.
UNITS: dict[str, str] = {
    "sand": "g/kg", "clay": "g/kg", "silt": "g/kg", "bdod": "cg/cm3",
    "phh2o": "pH*10", "soc": "dg/kg", "cec": "mmol(c)/kg", "k": "cmol(+)/kg",
    "p": "mg/kg", "cfvo": "cm3/dm3", "nitrogen": "cg/kg",
    "wv0010": "10^-3 cm3/cm3", "wv0033": "10^-3 cm3/cm3", "wv1500": "10^-3 cm3/cm3",
}
#: Variables the SH survey does not provide; taken from SoilGrids (nitrogen only with --nitrogen soilgrids).
FROM_SOILGRIDS: tuple[str, ...] = ("cfvo", "nitrogen")
#: Limits of the C:N ratio used to derive nitrogen (guards against tiny SoilGrids soc or N).
CN_LIMITS: tuple[float, float] = (5.0, 25.0)
WATER_VARS: tuple[str, ...] = ("wv0010", "wv0033", "wv1500")
#: SoilGrids nodata (float32 max) — anything above this is treated as missing.
_NODATA_THRESHOLD = 1e20


# ---------------------------------------------------------------------------
# Pedotransfer function
# ---------------------------------------------------------------------------

def saxton_rawls_2006(sand, clay, om) -> dict:
    """Water content at 10, 33 and 1500 kPa (Saxton & Rawls, 2006).

    Uses only element-wise arithmetic and ufuncs, so it works on NumPy arrays
    and on (lazy) xarray DataArrays alike.

    Parameters
    ----------
    sand, clay : array-like
        Mass fractions (0-1).
    om : array-like
        Organic matter, % by weight (the equations are fitted for OM <= 8 %).

    Returns
    -------
    dict
        ``wv0010``, ``wv0033``, ``wv1500`` as volumetric fractions (cm³/cm³).
    """
    S, C, OM = sand, clay, np.minimum(np.maximum(om, 0.0), 8.0)

    t1500 = (-0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM
             - 0.013 * C * OM + 0.068 * S * C + 0.031)
    th1500 = t1500 + (0.14 * t1500 - 0.02)

    t33 = (-0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM
           - 0.027 * C * OM + 0.452 * S * C + 0.299)
    th33 = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)

    ts33 = (0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM
            - 0.027 * C * OM - 0.584 * S * C + 0.078)
    ths33 = ts33 + (0.636 * ts33 - 0.107)
    th_sat = th33 + ths33 - 0.097 * S + 0.043

    # Air-entry tension (kPa); between it and 33 kPa the curve is linear.
    psi_et = (-21.67 * S - 27.93 * C - 81.97 * ths33 + 71.12 * S * ths33
              + 8.29 * C * ths33 + 14.05 * S * C + 27.16)
    psi_e = np.maximum(psi_et + (0.02 * psi_et ** 2 - 0.113 * psi_et - 0.70), 0.0)
    th10_linear = th33 + (33.0 - 10.0) * (th_sat - th33) / np.maximum(33.0 - psi_e, 1e-6)
    th10 = th_sat + (psi_e < 10.0) * (th10_linear - th_sat)
    return {"wv0010": th10, "wv0033": th33, "wv1500": th1500}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_reference(path: str) -> xr.Dataset:
    """Open the SoilGrids cube, attach its CRS and turn nodata into NaN."""
    ref = xr.open_dataset(path)
    attrs = dict(ref.attrs)  # .where() and .rio.write_crs() both drop "crs"
    ref = ref.where(ref < _NODATA_THRESHOLD)
    ref = ref.rio.write_crs(attrs["crs"])
    ref.attrs = attrs
    return ref


def soilgrids_on_sh_grid(layer: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    """Resample one SoilGrids layer (250 m, Homolosine) onto the SH grid.

    Nearest neighbour keeps SoilGrids' own values: each 250 m cell is repeated
    over the ~30 m SH pixels it covers.
    """
    da = layer.rio.write_nodata(np.nan, encoded=False)
    out = da.rio.reproject_match(template, resampling=Resampling.nearest)
    return out.where(np.isfinite(out)).assign_coords(x=template.x, y=template.y)


def fill_nearest(layer: xr.DataArray, max_km: float) -> xr.DataArray:
    """Fill NaN cells of a SoilGrids layer with the nearest valid cell, up to ``max_km`` away."""
    a = layer.values
    gaps = ~np.isfinite(a)
    if max_km <= 0 or not gaps.any() or gaps.all():
        return layer
    res_km = abs(float(layer.x[1] - layer.x[0])) / 1000.0  # SoilGrids grid is in metres
    dist, (iy, ix) = ndimage.distance_transform_edt(gaps, return_indices=True)
    filled = a[iy, ix]
    filled[dist * res_km > max_km] = np.nan
    return layer.copy(data=filled)


def nitrogen_from_soc(soc: xr.DataArray, cn: xr.DataArray | float) -> xr.DataArray:
    """Total nitrogen (cg/kg) from organic carbon (dg/kg) and a mass C:N ratio."""
    return soc * 10.0 / cn   # dg/kg x 10 = cg/kg


def soilgrids_cn(ref: xr.Dataset, depth: str, template: xr.DataArray) -> xr.DataArray:
    """SoilGrids C:N ratio (mass) at one depth on the SH grid, limited to CN_LIMITS."""
    soc = soilgrids_on_sh_grid(ref["soc"].sel(depth=depth), template)
    n = soilgrids_on_sh_grid(ref["nitrogen"].sel(depth=depth), template)
    return (soc * 10.0 / n.where(n > 0)).clip(*CN_LIMITS)


def sh_layers_in_soilgrids_units(sh: xr.Dataset, water: str) -> dict[str, xr.DataArray]:
    """SH topsoil layers, lazily converted to SoilGrids units.

    Sand, silt and clay are rescaled to add up to 1000 g/kg.  With
    ``water="ptf"`` the water-retention layers are added from Saxton & Rawls.
    """
    top: dict[str, xr.DataArray] = {}
    for sh_var, (sg_var, factor) in SH_TO_SOILGRIDS.items():
        top[sg_var] = sh[sh_var].isel(depth=0) * factor  # SH depths are copies of one layer
    total = top["sand"] + top["silt"] + top["clay"]
    for v in ("sand", "silt", "clay"):
        top[v] = top[v] * 1000.0 / total

    if water == "ptf":
        om_pct = top["soc"] * 1.724 / 100.0
        wv = saxton_rawls_2006(top["sand"] / 1000.0, top["clay"] / 1000.0, om_pct)
        for k, v in wv.items():
            top[k] = v * 1000.0  # SoilGrids stores 10^-3 cm³/cm³
    return top


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(
    sh_path: str,
    ref_path: str,
    out_path: str,
    sh_depths: tuple[str, ...] = ("0-5", "5-15", "15-30"),
    water: str = "soilgrids",
    fill_gaps: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    nitrogen: str = "cn-soilgrids",
    cn_ratio: float = 10.0,
    cfvo: str = "soilgrids",
    deep_fill_km: float = 5.0,
) -> None:
    """Write the SH cube as a SoilGrids-format cube on the SH ~30 m grid.

    Parameters
    ----------
    sh_path, ref_path, out_path : str
        SH cube, reference SoilGrids cube and output NetCDF.
    sh_depths : tuple[str, ...]
        Depth labels the SH topsoil layer represents.  Remaining reference
        depths are taken from SoilGrids.
    water : {"ptf", "soilgrids"}
        ``"soilgrids"`` (default) copies wv0010/wv0033/wv1500 from the
        reference cube; ``"ptf"`` derives them from SH texture and organic
        matter (Saxton & Rawls, 2006).  Applied to SoilGrids' own texture the
        PTF runs wetter than SoilGrids (mean bias +0.11 / +0.05 / +0.03
        cm³/cm³ for wv0010 / wv0033 / wv1500 over Honduras), so switching
        changes the DSSAT water balance noticeably.
    fill_gaps : bool
        Fill pixels where SH has no data (urban areas, water bodies) but
        SoilGrids does.  The DSSAT reader drops any depth with a NaN, so
        unfilled gaps become pixels without a soil profile.
    bbox : tuple[float, float, float, float] | None
        ``(xmin, ymin, xmax, ymax)`` in degrees to process only part of the
        SH grid — handy for a quick test before the full run.
    nitrogen : {"cn-soilgrids", "cn-fixed", "soilgrids"}
        Total nitrogen at the SH depths: SH soc ÷ SoilGrids C:N of the same
        pixel and depth (default), SH soc ÷ ``cn_ratio``, or SoilGrids nitrogen.
    cn_ratio : float
        C:N ratio for ``nitrogen="cn-fixed"``, and where SoilGrids has no C:N.
    cfvo : {"soilgrids", "zero"}
        Coarse fragments at the SH depths: SoilGrids, or 0 wherever SH has data.
    deep_fill_km : float
        Below the SH depths, fill SoilGrids gaps from the nearest valid
        SoilGrids cell up to this distance (0 = no filling).
    """
    ref = open_reference(ref_path)
    ref_depths = [str(d) for d in ref.depth.values]

    sh = xr.open_dataset(sh_path, decode_coords="all", chunks={"y": 2048, "x": 2048})
    sh_crs = sh.rio.crs
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        ysl = slice(ymax, ymin) if sh.y[0] > sh.y[-1] else slice(ymin, ymax)
        sh = sh.sel(x=slice(xmin, xmax), y=ysl)
    template = sh[list(SH_TO_SOILGRIDS)[0]].isel(depth=0).rio.write_crs(sh_crs)
    logger.info(
        "SH grid %s x %s (%s), reference depths %s",
        template.sizes["y"], template.sizes["x"], sh_crs, ref_depths,
    )

    cec_median = float(
        sh["cec"].isel(depth=0, x=slice(None, None, 50), y=slice(None, None, 50)).compute().median()
    )
    if cec_median < 1.0:
        logger.warning(
            "SH cec median is %.3f - that looks like electrical conductivity "
            "(dS/m), not CEC in cmol(+)/kg. Check the input cube.", cec_median,
        )

    top = sh_layers_in_soilgrids_units(sh, water)
    out_vars = list(top) + [v for v in FROM_SOILGRIDS + WATER_VARS if v not in top]
    missing = [v for v in out_vars if v not in top and v not in ref]
    if missing:
        raise KeyError(f"{missing} are neither in SH nor in the reference cube")

    from_sg = [v for v in FROM_SOILGRIDS if (v != "nitrogen" or nitrogen == "soilgrids")
               and (v != "cfvo" or cfvo == "soilgrids")] + ([] if water == "ptf" else list(WATER_VARS))
    attrs = {
        "crs": sh_crs.to_wkt(),  # read first by get_crs_fromxarray
        "source": "Suelos de Honduras (SH) in SoilGrids units on the SH grid",
        "sh_depths": ", ".join(sh_depths),
        "water_retention": "Saxton & Rawls (2006)" if water == "ptf" else "SoilGrids",
        "from_soilgrids": ", ".join(from_sg)
        + f"; every variable below {sh_depths[-1]} cm"
        + (f" (gaps filled from the nearest SoilGrids cell up to {deep_fill_km:g} km)" if deep_fill_km > 0 else "")
        + ("; SH no-data pixels" if fill_gaps else ""),
        "sh_only": "k, p (topsoil value repeated at every depth)",
        "nitrogen": {"cn-soilgrids": "SH soc / SoilGrids C:N (same pixel and depth)",
                     "cn-fixed": f"SH soc / C:N {cn_ratio}", "soilgrids": "SoilGrids"}[nitrogen],
        "cfvo": "SoilGrids" if cfvo == "soilgrids" else "0 at the SH depths (no SH layer)",
    }

    for i, v in enumerate(out_vars):
        logger.info("[%d/%d] %s", i + 1, len(out_vars), v)
        sh_layer = top[v].compute() if v in top else None
        layers = []
        for depth in ref_depths:
            # SH-only variables (k, p) repeat the topsoil value at every depth
            if v == "nitrogen" and nitrogen != "soilgrids" and depth in sh_depths:
                soc = top["soc"].compute()
                if fill_gaps:
                    soc = soc.fillna(soilgrids_on_sh_grid(ref["soc"].sel(depth=depth), template))
                # where SoilGrids has no C:N, fall back to the fixed ratio
                cn = soilgrids_cn(ref, depth, template).fillna(cn_ratio) if nitrogen == "cn-soilgrids" else cn_ratio
                da = nitrogen_from_soc(soc, cn)
                if fill_gaps:
                    da = da.fillna(soilgrids_on_sh_grid(ref[v].sel(depth=depth), template))
            elif v == "cfvo" and cfvo == "zero" and depth in sh_depths:
                soc = top["soc"].compute()
                da = xr.zeros_like(soc).where(soc.notnull())   # 0 wherever SH has data
                if fill_gaps:
                    da = da.fillna(soilgrids_on_sh_grid(ref[v].sel(depth=depth), template))
            elif sh_layer is not None and (depth in sh_depths or v not in ref):
                da = sh_layer
                if fill_gaps and v in ref and depth in sh_depths:
                    da = da.fillna(soilgrids_on_sh_grid(ref[v].sel(depth=depth), template))
            else:
                sg = ref[v].sel(depth=depth)
                if depth not in sh_depths:
                    sg = fill_nearest(sg, deep_fill_km)
                da = soilgrids_on_sh_grid(sg, template)
            layers.append(da.drop_vars(["depth", "spatial_ref"], errors="ignore"))

        cube = xr.concat(layers, dim="depth").assign_coords(depth=ref_depths)
        ds = cube.transpose("depth", "y", "x").astype("float32").to_dataset(name=v)
        ds[v].attrs = {"units": UNITS.get(v, "")}
        if i == 0:
            ds.attrs = attrs
        chunks = (1, min(1024, ds.sizes["y"]), min(1024, ds.sizes["x"]))
        encoding = {v: {"zlib": True, "complevel": 4, "chunksizes": chunks,
                        "_FillValue": np.float32(np.nan)}}
        ds.to_netcdf(out_path, mode="w" if i == 0 else "a", encoding=encoding)
        del sh_layer, layers, cube, ds

    logger.info("Saved %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sh", required=True, help="Suelos de Honduras datacube (.nc)")
    p.add_argument("--ref", required=True, help="Reference SoilGrids datacube (.nc)")
    p.add_argument("--out", required=True, help="Output NetCDF path")
    p.add_argument("--sh-depths", nargs="+", default=["0-5", "5-15", "15-30"],
                   help="Depth labels covered by the SH topsoil layer")
    p.add_argument("--water", choices=["ptf", "soilgrids"], default="soilgrids",
                   help="Source of wv0010/wv0033/wv1500")
    p.add_argument("--no-fill-gaps", action="store_true",
                   help="Leave SH no-data pixels as NaN instead of filling from SoilGrids")
    p.add_argument("--nitrogen", choices=["cn-soilgrids", "cn-fixed", "soilgrids"], default="cn-soilgrids",
                   help="Total N at the SH depths: SH soc / SoilGrids C:N (default), SH soc / --cn-ratio, "
                        "or SoilGrids N")
    p.add_argument("--cn-ratio", type=float, default=10.0, help="C:N ratio for --nitrogen cn-fixed")
    p.add_argument("--cfvo", choices=["soilgrids", "zero"], default="soilgrids",
                   help="Coarse fragments at the SH depths: SoilGrids, or 0 (topsoil entirely SH)")
    p.add_argument("--deep-fill-km", type=float, default=5.0,
                   help="Below the SH depths, fill SoilGrids gaps from the nearest valid cell up to "
                        "this distance in km (0 = off)")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="Only process this box (degrees) — for a quick test")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    convert(a.sh, a.ref, a.out, tuple(a.sh_depths), a.water, not a.no_fill_gaps,
            tuple(a.bbox) if a.bbox else None, a.nitrogen, a.cn_ratio, a.cfvo, a.deep_fill_km)
