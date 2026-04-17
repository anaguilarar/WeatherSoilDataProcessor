from spatialdata.gis_functions import (mask_xarray_using_rio, read_raster_data, 
                                       clip_xarraydata, reproject_xrdata, mask_xarray_using_gpdgeometry)
from spatialdata.utils import resample_variables
import numpy as np

from typing import Dict
import copy

from tqdm import tqdm
import xarray
from datetime import datetime

class DataCubeBase():
    """
    A base class for handling data cubes, including clipping,
    reprojection, and stacking multiple layers.

    Attributes
    ----------
    extent : tuple or None
        The spatial extent for cropping data, if provided.
    _date_path : dict
        Dictionary containing paths to data for each variable.
    """
    
    def __init__(self, extent = None) -> None:
        """
        Initialize the DataCubeBase with an optional spatial extent.

        Parameters
        ----------
        extent : tuple, optional
            Spatial extent to clip data to, in the form (xmin, ymin, xmax, ymax).
        """
        self._extent = extent
        
    @property
    def variables(self):
        return list(self._date_path.keys())

    @staticmethod
    def clip_using_box(xrdata, extent):
        """
        Clip xarray data using a bounding box (extent).

        Parameters
        ----------
        xrdata : xr.Dataset
            The xarray dataset to be clipped.
        extent : tuple
            The bounding box for clipping, in the form (xmin, ymin, xmax, ymax).

        Returns
        -------
        xr.Dataset
            The clipped xarray dataset.
        """
        #return mask_xarray_using_gpdgeometry(xrdata, geometry, clip = clip, all_touched = all_touched)
        return clip_xarraydata(xrdata, xyxy = extent)
    
    @staticmethod
    def reproject_xarray(xrdata, crs):
        """
        Reproject xarray data to a new coordinate reference system (CRS).

        Parameters
        ----------
        xrdata : xr.Dataset
            The xarray dataset to be reprojected.
        crs : str
            The target CRS (e.g., "EPSG:4326").

        Returns
        -------
        xr.Dataset
            The reprojected xarray dataset.
        """
        xr_re = reproject_xrdata(xrdata,target_crs=crs)

        return xr_re

    @staticmethod
    def mask_using_geometry(xrdata, geometry, clip = True, all_touched = True, reproject_to_raster = True, userio = False):
        """
        Mask xarray data using a geometry.

        Parameters
        ----------
        xrdata : xr.Dataset
            The xarray dataset to be masked.
        geometry : geopandas.GeoDataFrame
            The geometry to mask the xarray data with.
        clip : bool, optional
            If True, clip the dataset to the geometry. Defaults to False.
        all_touched : bool, optional
            If True, mask all grid cells touched by the geometry. Defaults to True.
        reproject_to_raster : bool, optional
            If True, reproject the geometry to the xarray dataset's CRS. Defaults to True.

        Returns
        -------
        xr.Dataset
            The masked xarray dataset.
        """
        if userio:
            return mask_xarray_using_rio(xrdata, geometry, drop = clip, all_touched = all_touched, reproject_to_raster = reproject_to_raster)
        else:
            return mask_xarray_using_gpdgeometry(xrdata, geometry, clip = clip, all_touched = all_touched) # this is faster for multiple dimensions
        
    

    def read_product(self, path, variable):
        """
        Read a data product for the given variable.

        Parameters
        ----------
        variable : str
            The name of the variable to read.

        Returns
        -------
        xr.Dataset
            The data for the given variable.
        """
        #path = self._date_path.get(variable)
        if path is None:
            raise ValueError(f"No data found for variable '{variable}'.")
    
        path = self._date_path[variable]
        return read_raster_data(path, crop_extent = self._extent)

    def stack_mlt_data(self, data_paths:Dict, reference_variable:str = None, **kwargs):
        self._date_path = data_paths
        xr_variables_list = {k: self.read_product(v, k) for k,v in self._date_path.items()}
        resampled_data = resample_variables(xr_variables_list, 
                                                    reference_variable=reference_variable,
                                                     **kwargs)
        
        return resampled_data
    



# def create_dimension(xrdata_dict, newdim_name = 'date', isdate = True):
#     datacube_mrs = []
#     for k,v in tqdm(xrdata_dict.items()):
#         xrtemp = v.expand_dims(dim = [newdim_name])
#         xrtemp[newdim_name] = [k]
#         datacube_mrs.append(xrtemp)
        
#     datacube_mrs = xarray.concat(datacube_mrs, dim = newdim_name)
#     if datacube_mrs.rio.crs is None:
#         datacube_mrs.rio.write_crs(v.rio.crs, inplace = True)
#         print(' crs ----------------------> ',datacube_mrs.rio.crs)
#     if isdate:
#         datacube_mrs[newdim_name] = [datetime.strptime(i, "%Y%m%d") for i in list(xrdata_dict.keys())]
#     return datacube_mrs

def create_dimension(xrdata_dict, newdim_name='date', isdate=True):
    datacube_mrs = []
    
    # 1. Capture the CRS before it gets destroyed by concat
    # We take it from the first dataset in the dictionary
    reference_crs = list(xrdata_dict.values())[0].rio.crs

    for k, v in tqdm(xrdata_dict.items()):
        xrtemp = v.expand_dims(dim=[newdim_name])
        xrtemp[newdim_name] = [k]
        datacube_mrs.append(xrtemp)
        
    # 2. Concatenate
    datacube_mrs = xarray.concat(datacube_mrs, dim=newdim_name)
    
    # 3. Restore the CRS robustly
    if reference_crs is not None:
        # write_crs on the Dataset creates the spatial_ref coordinate
        datacube_mrs.rio.write_crs(reference_crs, inplace=True)
        
        # VERY IMPORTANT: Force the 'grid_mapping' attribute on all spatial variables
        # so NetCDF recognizes it properly upon reloading.
        for var in datacube_mrs.data_vars:
            # We don't apply this to the spatial_ref variable itself
            if var != 'spatial_ref':
                datacube_mrs[var].attrs['grid_mapping'] = 'spatial_ref'
                
        print(' CRS successfully restored ----------------------> ', datacube_mrs.rio.crs)
        
    if isdate:
        datacube_mrs[newdim_name] = [datetime.strptime(i, "%Y%m%d") for i in list(xrdata_dict.keys())]
        
    return datacube_mrs