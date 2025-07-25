import numpy as np
import rasterio
import geopandas as gpd
import os
import pandas as pd
import xarray

from typing import Optional
from tqdm import tqdm


def model_selection(model: str, working_path: str):
    """
    Factory function to create a SpatialCM class dynamically inheriting from
    DSSATBase or PyCAF based on the `model` parameter.

    Parameters
    ----------
    model : str
        Model type, either 'dssat' or 'caf'.
    working_path : str
        Path to the working directory.
    config : dict
        Configuration dictionary for the SpatialCM instance.

    Returns
    -------
    SpatialCM
        A dynamically generated class that inherits from the appropriate base.
    """
    if model == "dssat":
        from ..dssat.base import DSSATBase

        BaseClass = DSSATBase
    elif model == "caf":
        from ..caf.base import PyCAF

        BaseClass = PyCAF
    elif model == "simple_model":
        from ..simple_model.base import PySimpleModel

        BaseClass = PySimpleModel

    else:
        raise ValueError(
            "Unsupported model type. Choose either 'dssat', 'caf' or 'simple_model'."
        )

    class ModelClass(BaseClass):
        """
        ModelClass to handle model initialization for DSSAT and CAF models.
        """

        def __init__(self, working_path: str):
            super().__init__(working_path)

            self.name = model

    return ModelClass(working_path)

def summarise_array_by_group(nparray: np.ndarray, group_by_layer: np.ndarray = None) -> np.ndarray:
    """_summary_

    Parameters
    ----------

        nparray : np.ndarray
            4-D array of shape, variables,  depth, y x
        group_by_layer (_type_): _description_
    """
    if group_by_layer is None:
        unique_categories = [0]
    else:
        categories = group_by_layer.flatten()
        unique_categories = np.unique(categories)
        unique_categories = unique_categories [~np.isnan(unique_categories)]

    v, d, y, x = nparray.shape

    category_means = np.full((v, len(unique_categories), d), np.nan)

    for j in range(v):
        numpydc = nparray[j].reshape(d, y * x)
        if len(unique_categories)>1:
            for i, cat in enumerate(unique_categories):
                # numpydc[:,categories == unique_categories[0]]
                category_means[j, i] = np.nanmean(numpydc[:, categories == cat], axis=1)
        else:
            category_means[j, 0] = np.nanmean(numpydc, axis=1)

    return category_means


def summarize_datacube_as_df(
    xrdata: xarray.Dataset =None,
    xrdata_path: str = None,
    dimension_name: str = 'date',
    group_by: Optional[str] = None,
    group_by_layer: Optional[np.ndarray] = None,
    pixel_scale: bool = False,
    project_to: Optional[str] = None,
    engine: str = 'netcdf4' 
    ) -> gpd.GeoDataFrame:
    """
    Summarizes a datacube (xarray.Dataset) into a DataFrame or GeoDataFrame with optional grouping and reprojection.

    Parameters
    ----------
    xrdata : xarray.Dataset, optional
        Input xarray Dataset to be summarized. Required if `xrdata_path` is not provided.
    xrdata_path : str, optional
        Path to a NetCDF file containing the dataset. Used if `xrdata` is not passed.
    dimension_name : str, optional
        The name of the dimension to aggregate data by (e.g., 'date'), by default 'date'.
    group_by : Optional[str], optional
        The variable name to group the data by (e.g., spatial units or categories), by default None.
    group_by_layer : np.ndarray, optional
        a numpy array indicating the group category for each spatial location
    pixel_scale : bool, False
        The data will be summarized at pixel scale, by default False.
    engine : str, default='netcdf4'
        Engine to use when loading data from NetCDF file.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the summarized data. Includes coordinates if `project_to` is specified.

    Notes
    -----
    - Requires variables 'x' and 'y' to exist in the xarray Dataset.
    - If `group_by` is not provided, a dummy group (0) is used.
    - The function assumes the input Dataset contains a 3D array per variable.

    Examples
    --------
    >>> summarize_datacube_as_df(xrdata=ds, group_by='region', group_by_layer=mask, project_to='EPSG:4326')
    """
    def get_info_from_xrdataset(xrdata):
        datavar_names = list(xrdata.data_vars.keys())
        xval = np.mean(xrdata['x'].values)
        yval = np.mean(xrdata['y'].values)
        npdata = xrdata.to_array().values
        if dimension_name:
            depth_vals =  xrdata[dimension_name].values 
        else:
            depth_vals = None
            npdata = np.expand_dims(npdata, axis = 1)    
        return xval,yval, depth_vals, datavar_names, npdata
    
    
    if xrdata_path:
        with xarray.open_dataset(xrdata_path, engine = engine) as xrdata:
            xval,yval, depth_vals, datavar_names, npdata = get_info_from_xrdataset(xrdata)
                
    elif xrdata is not None:
        xval,yval, depth_vals, datavar_names, npdata = get_info_from_xrdataset(xrdata)
    else:
        raise ValueError("Either 'xrdata' or 'xrdata_path' must be provided.")
    
    src_crs = get_crs_fromxarray(xrdata)
    
    if group_by is None:
        unique_categories= [0]
        group_by= 'group'
    else:
        if group_by_layer is None:
            raise ValueError("'group_by_layer' must be provided when 'group_by' is set.")
        unique_categories = np.unique(group_by_layer)
        
    if pixel_scale:
        ddf = xrdata.to_dataframe().reset_index().dropna()
        del  xrdata
    else:
        del  xrdata
        npdata = summarise_array_by_group(npdata, group_by_layer)
        v,g,d = npdata.shape
        ddf = []
        for i in range(g):
            df = pd.DataFrame(npdata[:,i].reshape(v, d).swapaxes(0,1), columns=datavar_names)
            if depth_vals is not None: df[dimension_name] = depth_vals
            df['x'] = xval
            df['y'] = yval
            df[group_by] = unique_categories[i]
            ddf.append(df)
            
        ddf = pd.concat(ddf)
    
    if project_to is not None:
        ddf = project_dataframe(ddf, src_crs, target_crs=project_to)
        
    return ddf


def get_variables_to_summarize(xrdata, indexname):
    ## add coords

    datavars = {i: "mean" for i in xrdata.data_vars.keys()}
    for i in xrdata.sizes.keys():
        if indexname == i:
            continue
        datavars.update({i: "mean"})
        datavars.update({i: "mean"})

    return datavars


def summarize_dataframe(date, xrdata, variables, groupby=None):
    ddf = xrdata.to_dataframe().reset_index().dropna()
    if groupby:
        ddf = ddf.groupby([groupby], dropna=True).agg(variables).reset_index()
    else:
        ddf["tmp"] = 0
        groupby = "tmp"
        ddf = ddf.dropna().reset_index()
    ddf["date"] = date

    return ddf


def check_percentage(value):
    return value * 0.1 if np.max(value) > 100 else value


def set_encoding(xrdata, compress_method="zlib"):
    return {k: {compress_method: True} for k in list(xrdata.data_vars.keys())}


def check_crs_inxrdataset(xrdataset):
    if "crs" in xrdataset.attrs.keys():
        crs = xrdataset.attrs["crs"]
        if isinstance(crs, rasterio.crs.CRS):
            xrdataset.attrs["crs"] = crs.to_string()
    return xrdataset


def get_crs_fromxarray(xrdata):
    crs = None
    if "crs" in xrdata.attrs.keys():
        crs = xrdata.attrs["crs"]
    else:
        try:
            crs = xrdata.rio.crs
            crs = (
                xrdata[list(xrdata.data_vars.keys())[0]].rio.crs
                if crs is None
                else None
            )

        except:
            crs = None
    if crs is not None:
        crs = crs.to_string() if not isinstance(crs, str) else crs
    return crs


def project_dataframe(df, source_crs, target_crs):
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    df = df.set_crs(source_crs, allow_override=True)
    df = df.to_crs(target_crs)
    # Calculate the mean coordinates after reprojection
    df["x"] = df.geometry.values.map(lambda x: float(x.coords.xy[0][0]))
    df["y"] = df.geometry.values.map(lambda x: float(x.coords.xy[1][0]))

    return df


def export_data_ascsv(processed_sims, output_data, outputpath, crop):
    date_colname = "PDAT" if crop != "coffee" else "HDAT"
    y_colname = "HWAH" if crop != "coffee" else "harvDM_f_hay"

    completedgroups = [k for k, v in processed_sims.items() if v]

    for gval in tqdm(completedgroups):
        output_data[gval].weather_data().to_csv(
            os.path.join(outputpath, gval, "weather.csv")
        )
        dftmp = output_data[gval].output_data().sort_values(date_colname)
        if crop != "coffee":
            dftmp = dftmp.loc[dftmp[y_colname] != 0]
        dftmp["group"] = gval

        dftmp.to_csv(os.path.join(outputpath, gval, f"{crop}_potential_yield.csv"))


def add_layer_texture_to_datacubes(weather_dc, soil_dc, dem_dc = None):
        weatherdatavars = list(weather_dc.data_vars.keys())
        soilref = get_layer_texture(soil_dc.isel(depth = 0))
        # merge texture to weather and soil
        weather_dc = xarray.merge([weather_dc,soilref])[weatherdatavars+ ['texture']]
        soil_dc = xarray.merge([soil_dc,soilref['texture']])[list(soil_dc.data_vars.keys())+ ['texture']]
        if dem_dc is not None:
            dem_dc = xarray.merge([dem_dc,soilref['texture']])[list(dem_dc.data_vars.keys())+ ['texture']]
        return weather_dc, soil_dc, dem_dc