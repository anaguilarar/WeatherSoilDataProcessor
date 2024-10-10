from spatialdata.gis_functions import mask_xarray_using_rio, read_raster_data, clip_xarraydata, resample_xarray
from spatialdata.utils import resample_variables
import numpy as np

from typing import Dict

class DataCubeBase():
    @property
    def weather_variables(self):
        return list(self._date_path.keys())

    @staticmethod
    def clip_using_box(xrdata, extent):
        #return mask_xarray_using_gpdgeometry(xrdata, geometry, clip = clip, all_touched = all_touched)
        return clip_xarraydata(xrdata, xyxy = extent)
    

    @staticmethod
    def mask_using_geometry(xrdata, geometry, clip = False, all_touched = True, reproject_to_raster = True):
        #return mask_xarray_using_gpdgeometry(xrdata, geometry, clip = clip, all_touched = all_touched)
        return mask_xarray_using_rio(xrdata, geometry, drop = clip, all_touched = all_touched, reproject_to_raster = reproject_to_raster)
    

    def read_product(self, variable):
        path = self._date_path[variable]
        return read_raster_data(path, crop_extent = self._extent)

    def stack_mlt_data(self, data_paths:Dict, reference_variable:str = None, verbose:bool = False):
        self._date_path = data_paths
        xr_variables_list = {k: self.read_product(k) for k,v in self._date_path.items()}
        resampled_data = resample_variables(xr_variables_list, 
                                                    reference_variable=reference_variable, verbose = verbose)
        
        return resampled_data.where(resampled_data != -9999, np.nan)
    

    def __init__(self, extent = None) -> None:
        
        self._extent = extent


class WeatherDataCube():
    @property
    def weather_variables(self):
        return list(self._date_path.keys())

    @staticmethod
    def mask_using_geometry(xrdata, geometry, clip = False, all_touched = True, reproject_to_raster = True):
        #return mask_xarray_using_gpdgeometry(xrdata, geometry, clip = clip, all_touched = all_touched)
        return mask_xarray_using_rio(xrdata, geometry, drop = clip, all_touched = all_touched, reproject_to_raster = reproject_to_raster)
    
    def read_ind_product(self, variable):
        path = self._date_path[variable]
        return read_raster_data(path, crop_extent = self._extent)

    def stack_single_date_data(self, data_paths, reference_variable):
        self._date_path = data_paths
        xr_variables_list = {k: self.read_ind_product(k) for k,v in self._date_path.items()}
        resampled_data = resample_variables(xr_variables_list, 
                                                    reference_variable=reference_variable)
        
        return resampled_data.where(resampled_data != -9999, np.nan)
    

    def __init__(self, extent = None) -> None:
        
        self._extent = extent

