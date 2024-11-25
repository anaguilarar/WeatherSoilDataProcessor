from .utils.u_soil import get_layer_texture
from spatialdata.datacube import create_dimension, reproject_xrdata
from .utils.process import get_crs_fromxarray
from spatialdata.gis_functions import masking_rescaling_xrdata
from tqdm import tqdm
from spatialdata.data_fromconfig import get_weather_datacube, get_soil_datacube
from spatialdata.xr_dict import CustomXarray
from spatialdata.xr_dict import CustomXarray, from_dict_toxarray
import numpy as np
from .utils.process import set_encoding, check_crs_inxrdataset

from typing import Any
import pickle
from omegaconf import OmegaConf

def reproject_xarray(xrdata, target_crs, src_crs = None):
    if src_crs is None:
        try:
            src_crs = xrdata.rio.crs
        except:
            src_crs = xrdata.attrs.get('crs', None)
            
    assert src_crs is not None, "Please provide the source crs"
                        
    return xrdata.rio.write_crs(src_crs).rio.reproject(target_crs)

import xarray

def get_roi_data(roi, weather_datacube, soil_datacube, aggregate_by = None, min_area = 15, scale_factor = 10):
    """_summary_

    Parameters
    ----------
    min_area : int
        the minimun feature geometry area to apply buffer in km2

    Returns
    -------
    (xarray.Dataset, xarray.Dataset)
        a tuple with the Multi-dimension climate and soil data for the given region of interest.
    """
    area = roi.area.values[0]/ (1000*1000)
    if area < min_area:
        narea = (min_area*1.2) - area
        buffer = (narea*100)
    else:
        buffer = None 

    weather_datacube_m = masking_rescaling_xrdata(weather_datacube, roi, buffer=buffer, scale_factor=scale_factor, return_original_size=True, method = 'nearest')
    xr_reference = weather_datacube_m.isel(date = 0)
    
    weatherdatavars = list(weather_datacube_m.data_vars.keys())
    weather_datacube_m.attrs['crs'] = get_crs_fromxarray(weather_datacube)    
    
    if isinstance(soil_datacube, dict):
        soil_datacube_m = {k: masking_rescaling_xrdata(v, roi, buffer=buffer, resample_ref =xr_reference)  for k,v in tqdm(soil_datacube.items())}
        soil_datacube_m = create_dimension(soil_datacube_m, newdim_name = 'depth', isdate = False)
            
    else:
        soil_datacube_m = masking_rescaling_xrdata(soil_datacube, roi, buffer=buffer, resample_ref =xr_reference, return_original_size=True, method = 'nearest')
    
    if aggregate_by == 'texture':
        soilref = get_layer_texture(soil_datacube_m.isel(depth = 0))
        # merge texture to weather and soil
        weather_datacube_m = xarray.merge([weather_datacube_m,soilref])[weatherdatavars+ ['texture']]
        soildatavars = list(soil_datacube_m.data_vars.keys())
        soil_datacube_m = xarray.merge([soil_datacube_m,soilref['texture']])[soildatavars+ ['texture']]

    soil_datacube_m.attrs['crs'] = get_crs_fromxarray(weather_datacube_m)

    return weather_datacube_m, soil_datacube_m


class WeatherTransformer():
    
    def __init__(self, var_names = {'rain':'precipitation','tmax':'tmax', 'tmin': 'tmin', 'srad': 'srad'}) -> None:
        
        self.rain = var_names.get('rain', None)
        self.tmin = var_names.get('tmin', None)
        self.tmax = var_names.get('tmax', None)
        self.srad = var_names.get('srad', None)
    
    def __call__(self,xrdata) -> Any:
        if self.tmax in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[self.tmax].values)>273.15:
            xrdata[self.tmax] -= 273.15
        if self.tmin in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[self.tmin].values)>273.15:
            xrdata[self.tmin] -= 273.15
        if self.srad in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[self.srad].values)>1000000:
            xrdata[self.srad] /= 1000000

        return xrdata

class CM_SpatialData():
    
    def area_extension():
        pass
    
    #def export_data(self, data):
    #    encoding = set_encoding(data)
    @property
    def dim_names(self):
        return {'climate': 'date',
                'soil': 'depth'}
    
    def _open_dataset(self, filepath):
        if filepath.endswith('.nc'):
            data = xarray.open_dataset(filepath, engine = self.config.SPATIAL_INFO.engine)
        if filepath.endswith('.pickle'):
            with open(filepath, 'rb') as fn:
                data = pickle.load(fn)
        
        print(f'loaded from {filepath}')
        return data
    
    def _save_asnc(self, xrdata, fn):
        encoding = set_encoding(xrdata)
        xrdata = check_crs_inxrdataset(xrdata)
        xrdata.to_netcdf(fn, encoding = encoding, engine = self.config)
        
    def _setup(self):
        self.climate = None
        self.soil = None
        self.ndvi = None
        self.weather_transformer = WeatherTransformer()
    
    def __init__(self, configuration:str) -> None:
        self.config = OmegaConf.load(configuration)
        self._setup()
    
    def set_databases(self, climate = True, soil = True):
        if climate: self.get_climate_data()
        if soil: self.get_soil_data()

    
    def get_climate_data(self, export_data = False):
        if self.config.DATA.get('climate_data_cube_path', None):
            weather_datac =  self._open_dataset(self.config.DATA.climate_data_cube_path)
        else:
            weather_datac = get_weather_datacube(self.config)
            weather_datac = create_dimension(weather_datac, newdim_name=self.dim_names['climate'], isdate=False)
            if export_data:
                fn = fn if fn else f'weather_data{self.config.SPATIAL_INFO.engine}.nc'
                self._save_asnc(weather_datac, fn)
        
        self.climate = weather_datac.rio.write_crs(get_crs_fromxarray(weather_datac)).rio.reproject(self.config.SPATIAL_INFO.crs)
        self.climate.attrs['crs'] = self.config.SPATIAL_INFO.crs
        
    def get_soil_data(self, export_data = False, fn = None):
        if self.config.DATA.get('soil_datacube_path', None):
            soil_datac =  self._open_dataset(self.config.DATA.soil_datacube_path)

        else:
            soil_datac = get_soil_datacube(self.config)
            if export_data:
                customdict = {k: CustomXarray(v, dataformat = 'CHW').custom_dict for k,v in soil_datac.items()}
                soil_datac =create_dimension({k:from_dict_toxarray(v, dimsformat='CHW') for i, (k, v) in enumerate(customdict.items()) if i <4}, newdim_name= 'depth', isdate=False)
                fn = fn if fn else f'soil_data{self.config.SPATIAL_INFO.engine}.nc'
                self._save_asnc(soil_datac, fn)

        if isinstance(soil_datac, dict):self.soil = {k: reproject_xrdata(v,target_crs=self.config.SPATIAL_INFO.crs) for k, v in soil_datac.items()}
        else:
            self.soil = soil_datac.rio.write_crs(get_crs_fromxarray(soil_datac)).rio.reproject(self.config.SPATIAL_INFO.crs)
            self.soil.attrs['crs'] = self.config.SPATIAL_INFO.crs
    
    def extract_roi_data(self, roi, group_by = None):
        if self.climate is None: self.get_climate_data()
        if self.soil is None: self.get_soil_data()
        
        weatherm, soilm= get_roi_data(roi, self.climate, self.soil, aggregate_by = group_by, scale_factor= self.config.WEATHER.scale_factor)
        
        return weatherm, soilm
    

