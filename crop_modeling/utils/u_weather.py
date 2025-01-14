import pandas as pd
import numpy as np

from typing import Any

from spatialdata.gis_functions import get_boundaries_from_path
from spatialdata.climate_data import MLTWeatherDataCube
from spatialdata.files_manager import IntervalFolderManager

def monthly_amplitude(c):
    d = {}
    d['avgm'] = ((c.iloc[:,0] + c.iloc[:,1])/2).mean()
    return pd.Series(d, index = ['avgm'])



def get_weather_datacube(config):
    """
    Generates a multi-temporal weather data cube.

    Parameters
    ----------
    config : omegaconf.DictConfig
        The configuration object containing file paths, dates, and other parameters.

    Returns
    -------
    xarray.Dataset
        Multi-temporal weather data cube with the requested variables.
    """
    ncores= config.GENERAL_INFO.get('ncores', 0)
    
    boundaries = config.SPATIAL_VECTOR.get('boundaries', None)
    extent = get_boundaries_from_path(boundaries, round_numbers = True, crs = config.WEATHER.setup_parameters.crs) if boundaries else None

    # Specify paths for weather data, such as precipitation and solar radiation
    list_weather_paths = {'precipitation': config.WEATHER.setup_parameters.paths.precipitation,
                        'srad': config.WEATHER.setup_parameters.paths.srad,
                        'tmax': config.WEATHER.setup_parameters.paths.tmax,
                        'tmin': config.WEATHER.setup_parameters.paths.tmin}
    if 'wn' in config.WEATHER.setup_parameters.paths.keys():
        list_weather_paths.update({'wn':  config.WEATHER.setup_parameters.paths.wn})
    if 'vp' in config.WEATHER.setup_parameters.paths.keys():
        list_weather_paths.update({'vp':  config.WEATHER.setup_parameters.paths.vp})

    wdatacube = MLTWeatherDataCube(list_weather_paths, IntervalFolderManager(), extent=extent)

    wdatacube.common_dates_and_file_names(starting_date=config.WEATHER.setup_parameters.period[0], 
                                        ending_date=config.WEATHER.setup_parameters.period[1])
    
    return wdatacube.multitemporal_data(reference_variable=config.WEATHER.setup_parameters.reference_variable, 
                                        ncores=ncores)

class WeatherTransformer():
    
    def __init__(self, var_names = {'rain':'precipitation','tmax':'tmax', 'tmin': 'tmin', 'srad': 'srad', 'vp':'vp', 'wn': 'wn'}) -> None:
        
        self.rain = var_names.get('rain', None)
        self.tmin = var_names.get('tmin', None)
        self.tmax = var_names.get('tmax', None)
        self.srad = var_names.get('srad', None)
        self.vp = var_names.get('vp', None)
        self.wn = var_names.get('wn', None)
    
    
    def __call__(self,xrdata) -> Any:
        if self.tmax in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[self.tmax].values)>273.15: # from K to C
            xrdata[self.tmax] -= 273.15
        if self.tmin in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[self.tmin].values)>273.15: # from K to C
            xrdata[self.tmin] -= 273.15
        if self.srad in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[self.srad].values)>1000000: # from J m-2 day-1 to MJ m-2 day-1
            xrdata[self.srad] /= 1000000
        if self.vp in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[self.vp].values)>10: # from hPa to kPa
            xrdata[self.vp] *= 0.1

        return xrdata