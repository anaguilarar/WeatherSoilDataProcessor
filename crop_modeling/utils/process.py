
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
    if model == 'dssat':
        from ..dssat.base import DSSATBase
        BaseClass = DSSATBase
    elif model == 'caf':
        from ..caf.base import PyCAF
        BaseClass = PyCAF
    elif model == 'simple_model':
        from ..simple_model.base import PySimpleModel
        BaseClass = PySimpleModel
        
    else:
        raise ValueError("Unsupported model type. Choose either 'dssat', 'caf' or 'simple_model'.")

    class ModelClass(BaseClass):
        """
        ModelClass to handle model initialization for DSSAT and CAF models.
        """
        def __init__(self, working_path: str):
            super().__init__(working_path)

            self.name = model
            
    return ModelClass(working_path)


def summarize_datacube_as_df(
    xrdata: xarray.Dataset,
    dimension_name: str = 'date',
    group_by: Optional[str] = None,
    pixel_scale: bool = False,
    project_to: Optional[str] = None
    ) -> gpd.GeoDataFrame:
    """
    Summarizes a datacube (xarray.Dataset) into a DataFrame or GeoDataFrame with optional grouping and reprojection.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input multi-dimensional datacube to be summarized.
    dimension_name : str, optional
        The name of the dimension to aggregate data by (e.g., 'date'), by default 'date'.
    group_by : Optional[str], optional
        The variable name to group the data by (e.g., spatial units or categories), by default None.
    pixel_scale : bool, False
        The data will be summarized at pixel scale, by default False.
    project_to : Optional[str], optional
        The target coordinate reference system (CRS) for projecting spatial data, by default None.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the summarized data. Includes coordinates if `project_to` is specified.

    Notes
    -----
    - If `group_by` is not provided, the function creates a temporary 'group' column for aggregation.
    - If `project_to` is provided, the data is converted to a GeoDataFrame and projected to the specified CRS.
    - Assumes that the input `xrdata` has variables for latitude ('x') and longitude ('y').

    Examples
    --------
    >>> summarized_df = summarize_datacube_as_df_(xrdata, group_by='region', project_to='EPSG:4326')
    """
    datavar_names = get_variables_to_summarize(xrdata, dimension_name)
    
    df = xrdata.to_dataframe().reset_index().dropna()
    
    if group_by:
        datavar_names.pop(group_by)
        if dimension_name:
            ddf = df.groupby([group_by, dimension_name], dropna = True).agg(datavar_names).reset_index()
        else:
            ddf = df.groupby([group_by], dropna = True).agg(datavar_names).reset_index()
    elif pixel_scale:
        ddf = df.copy()
        
    else:
        ddf = df.copy()
        ddf['group'] = 0
        group_by= 'group'
        if dimension_name:
            ddf = ddf.groupby([group_by,dimension_name], dropna = True).agg(datavar_names).reset_index()
        else:
            ddf = ddf.groupby([group_by], dropna = True).agg(datavar_names).reset_index()
    
    if project_to is not None:

        src_crs = get_crs_fromxarray(xrdata)
        ddf = project_dataframe(ddf, src_crs, target_crs=project_to)
        
    return ddf


def get_variables_to_summarize(xrdata, indexname):
     ## add coords

    datavars = {i:'mean' for i in xrdata.data_vars.keys()}
    for i in xrdata.sizes.keys():
        if indexname == i:
            continue
        datavars.update({i:'mean'})
        datavars.update({i:'mean'})

    return datavars


def summarize_dataframe(date, xrdata, variables, groupby = None):
        ddf = xrdata.to_dataframe().reset_index().dropna()
        if groupby:
            ddf = ddf.groupby([groupby], dropna = True).agg(variables).reset_index()
        else:
            ddf['tmp'] = 0
            groupby= 'tmp'
            ddf = ddf.dropna().reset_index()
        ddf['date'] = date

        return ddf

def check_percentage(value):
    return value *0.1 if np.max(value)>100 else value


def set_encoding(xrdata, compress_method = 'zlib'):
    return {k: {compress_method: True} for k in list(xrdata.data_vars.keys())}

def check_crs_inxrdataset(xrdataset):
    if 'crs' in xrdataset.attrs.keys():
        crs = xrdataset.attrs['crs']
        if isinstance(crs, rasterio.crs.CRS):
            xrdataset.attrs['crs'] = crs.to_string()
    return xrdataset

def get_crs_fromxarray(xrdata):
    crs = None
    if 'crs' in xrdata.attrs.keys():
        crs = xrdata.attrs['crs']
    else:
        try:
            crs = xrdata.rio.crs
            crs = xrdata[list(xrdata.data_vars.keys())[0]].rio.crs if crs is None else None

        except:
            crs = None
    if crs is not None:
        crs = crs.to_string() if not isinstance(crs, str) else crs
    return crs


def project_dataframe(df, source_crs, target_crs):
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x,df.y))
    df = df.set_crs(source_crs, allow_override=True)
    df = df.to_crs(target_crs)
    # Calculate the mean coordinates after reprojection
    df['x'] = df.geometry.values.map(lambda x: float(x.coords.xy[0][0]))
    df['y'] = df.geometry.values.map(lambda x: float(x.coords.xy[1][0]))
    
    return df


def export_data_ascsv(processed_sims, output_data, outputpath, crop):
    date_colname = 'PDAT' if crop != 'coffee' else 'HDAT'
    y_colname = 'HWAH' if crop != 'coffee' else 'harvDM_f_hay'
        
    completedgroups = [k for k,v in processed_sims.items() if v]

    for gval in tqdm(completedgroups):
        output_data[gval].weather_data().to_csv(os.path.join(outputpath, gval, 'weather.csv'))
        dftmp = output_data[gval].output_data().sort_values(date_colname)
        if crop != 'coffee':
            dftmp = dftmp.loc[dftmp[y_colname] != 0]
        dftmp['group'] = gval
        
        dftmp.to_csv(os.path.join(outputpath, gval,f'{crop}_potential_yield.csv'))
    