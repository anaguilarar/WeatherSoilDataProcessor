import geopandas as gpd
import numpy as np
import pickle
import os

from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Any, Dict, Optional
from pathlib import Path

from .dssat.base import DSSATBase
from .utils.process import get_crs_fromxarray,set_encoding, check_crs_inxrdataset
from .utils.u_soil import get_layer_texture
from spatialdata.climate_data import MLTWeatherDataCube
from spatialdata.datacube import create_dimension, reproject_xrdata
from spatialdata.files_manager import IntervalFolderManager, SoilFolderManager
from spatialdata.gis_functions import masking_rescaling_xrdata, get_boundaries_from_path
from spatialdata.soil_data import SoilDataCube
from spatialdata.xr_dict import CustomXarray, from_dict_toxarray


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


    wdatacube = MLTWeatherDataCube(list_weather_paths, IntervalFolderManager(), extent=extent)

    wdatacube.common_dates_and_file_names(starting_date=config.WEATHER.setup_parameters.period[0], 
                                                      ending_date=config.WEATHER.setup_parameters.period[1])
    
    return wdatacube.multitemporal_data(reference_variable=config.WEATHER.setup_parameters.reference_variable, 
                                        ncores=ncores)


def get_soil_datacube(config):
    """
    Generates a multi-depth soil data cube.

    Parameters
    ----------
    config : omegaconf.DictConfig
        The configuration object containing soil data paths and CRS reference.

    Returns
    -------
    xarray.Dataset
        Multi-depth soil data cube for the given extent and variables.
    """
    boundaries = config.SPATIAL_VECTOR.get('boundaries', None)
    extent = get_boundaries_from_path(boundaries, round_numbers = True, crs = config.SOIL.setup_parameters.crs) if boundaries else None

    folder_manager = SoilFolderManager(config.SOIL.setup_parameters.path, config.SOIL.setup_parameters.variables)
    soilcube = SoilDataCube(folder_manager, extent=extent)
    return soilcube.multi_depth_data(verbose=False, reference_variable=config.SOIL.setup_parameters.reference_variable)


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
    roi: geopandas.GeoDataFrame
        region of interest with projected data
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

class SpatialData():
    """
    A class for managing spatial data for climate and soil analyses.

    Attributes
    ----------
    config : OmegaConf
        The configuration loaded from a YAML file.
    climate : Optional[xarray.Dataset]
        The climate data cube loaded or processed.
    soil : Optional[Union[xarray.Dataset, dict]]
        The soil data cube loaded or processed.
    ndvi : Optional[Any]
        Placeholder for NDVI data (not implemented yet).
    weather_transformer : WeatherTransformer
        An instance of the WeatherTransformer for climate data transformations.
    """
    
    def __init__(self, configuration_path:str = None, configuration_dict: Dict = None) -> None:
        """
        Initialize the spatial data manager with a configuration file.

        Parameters
        ----------
        configuration_path : str
            Path to the configuration YAML file.
            
        configuration_dict : dict
            Path to the configuration YAML file.
            
        """
        self.config = None
        if configuration_path:
            self.config = OmegaConf.load(configuration_path)
        elif configuration_dict:
            self.config = OmegaConf.create(configuration_dict)
        
        #assert self.config is not None, "Please provide either a configuration path or a dictionary with the configuration"
        #if self.config is not None:
        self._setup()
        
    def area_extension():
        pass
    
    #def export_data(self, data):
    #    encoding = set_encoding(data)
    @property
    def dim_names(self):
        return {'climate': 'date',
                'soil': 'depth'}
        
    @staticmethod
    def _open_dataset(filepath: str, engine = 'netcdf4') -> xarray.Dataset:
        """
        Open a dataset from a file, supporting NetCDF and pickle formats.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.

        Returns
        -------
        xarray.Dataset
            The loaded dataset.
        """
        if filepath.endswith('.nc'):
            data = xarray.open_dataset(filepath, engine = engine)
        elif filepath.endswith('.pickle'):
            with open(filepath, 'rb') as fn:
                data = pickle.load(fn)
        else:
            raise ValueError("Unsupported file format. Use '.nc' or '.pickle'.")
        
        print(f'loaded from {filepath}')
        return data
    
    def _save_asnc(self, xrdata: xarray.Dataset, fn: str) -> None:
        """
        Save a dataset to a NetCDF file with appropriate encoding.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The dataset to save.
        fn : str
            Output file name.
        """
        encoding = set_encoding(xrdata)
        xrdata = check_crs_inxrdataset(xrdata)
        xrdata.to_netcdf(fn, encoding = encoding, engine = self._dcengine )
        
    def _setup(self):
        self.climate = None
        self.soil = None
        self.ndvi = None
        self._projected_crs = 'ESRI:54052'
        self._geo_crs = 'EPSG:4326'
        self.weather_transformer = WeatherTransformer()
        if self.config is not None:
            
            self._weather_datacube = self.config.WEATHER.get('data_cube_path', None)
            self._soil_datacube = self.config.SOIL.get('data_cube_path', None)
            self._dcengine = 'netcdf4'
            self._projected_crs = self.config.GENERAL_INFO.get('projected_crs', 'ESRI:54052')
            self._geo_crs = self.config.GENERAL_INFO.get('geo_crs', 'EPSG:4326')
    
    def set_databases(self, climate: bool = True, soil: bool = True) -> None:
        """
        Set up climate and/or soil data by loading or processing datasets.

        Parameters
        ----------
        climate : bool, optional
            Whether to set up climate data, by default True.
        soil : bool, optional
            Whether to set up soil data, by default True.
        """
        if climate: self.get_climate_data()
        if soil: self.get_soil_data()

    
    def get_climate_data(self, export_data = False, fn = None):
        """
        Load or process climate data.

        Parameters
        ----------
        export_data : bool, optional
            Whether to export the data to a NetCDF file, by default False.
        fn : Optional[str], optional
            Filename for exporting data, by default None.
        """
        
        if self._weather_datacube:
            weather_datac =  self._open_dataset(self._weather_datacube)
        else:
            weather_datac = get_weather_datacube(self.config)
            weather_datac = create_dimension(weather_datac, newdim_name=self.dim_names['climate'], isdate=False)
            if export_data:
                fn = fn if fn else f'weather_data{self._dcengine}.nc'
                self._save_asnc(weather_datac, fn)
        
        self.climate = weather_datac.rio.write_crs(get_crs_fromxarray(weather_datac)).rio.reproject(self._projected_crs)
        self.climate.attrs['crs'] = self._projected_crs
        
    def get_soil_data(self, export_data = False, fn = None):
        """
        Load or process soil data.

        Parameters
        ----------
        export_data : bool, optional
            Whether to export the data to a NetCDF file, by default False.
        fn : Optional[str], optional
            Filename for exporting data, by default None.
        """
        
        if self._soil_datacube:
            soil_datac =  self._open_dataset(self._soil_datacube)

        else:
            soil_datac = get_soil_datacube(self.config)
            customdict = {k: CustomXarray(v, dataformat = 'CHW').custom_dict for k,v in soil_datac.items()}
            soil_datac =create_dimension({k:from_dict_toxarray(v, dimsformat='CHW') for i, (k, v) in enumerate(customdict.items()) if i <4}, newdim_name= 'depth', isdate=False)    
            if export_data:
                fn = fn if fn else f'soil_data{self._dcengine}.nc'
                self._save_asnc(soil_datac, fn)

        if isinstance(soil_datac, dict):self.soil = {k: reproject_xrdata(v,target_crs=self._projected_crs) for k, v in soil_datac.items()}
        else:
            self.soil = soil_datac.rio.write_crs(get_crs_fromxarray(soil_datac)).rio.reproject(self._projected_crs)
            self.soil.attrs['crs'] = self._projected_crs
    

class SpatialCM(DSSATBase):
    """
    A class for managing and processing spatial data for crop modeling systems like DSSAT.

    Parameters
    ----------
    configuration_path : str, optional
        Path to the configuration file in YAML format.
    configuration_dict : dict, optional
        Dictionary containing configuration details. Ignored if `configuration_path` is provided.

    Attributes
    ----------
    config : OmegaConf
        Configuration object loaded from a file or dictionary.
    geo_features : gpd.GeoDataFrame
        Geospatial data loaded from the path specified in the configuration.
    climate : xr.Dataset
        Climate dataset loaded from the configuration.
    soil : xr.Dataset
        Soil dataset loaded from the configuration.
    """

    @property
    def geo_features(self):
        if self._geodata is None:
            geo_path = self.config.SPATIAL_INFO.geospatial_path
            if not os.path.exists(geo_path):
                raise FileNotFoundError(f"Geospatial data path not found: {geo_path}")
            
            self._geodata = gpd.read_file(geo_path)
        
        return self._geodata
    
    @property
    def climate(self):
        """
        Lazy-loaded property for climate data.
        """
        if self._climate is None:
            weather_path = self.config.SPATIAL_INFO.weather_path
            if not os.path.exists(weather_path):
                raise FileNotFoundError(f"Climate data path not found: {weather_path}")
            self._climate = SpatialData()._open_dataset(weather_path)
            
        return self._climate        
        
    @property
    def soil(self):
        if self._soil is None:
            soil_path = self.config.SPATIAL_INFO.soil_path
            if not os.path.exists(soil_path):
                raise FileNotFoundError(f"Soil data path not found: {soil_path}")
            self._soil = SpatialData()._open_dataset(soil_path)
            
        return self._soil        
    
    def _setup(self):
        """
        Initializes essential properties and validates configurations.
        """
        self.soil  # Load soil dataset
        self.climate  # Load climate dataset
        self._model = self.config.GENERAL_INFO.get('model', None)
        self.crop = self.config.CROP.get('name', None)
        self.cultivar = self.config.CROP.get('cultivar', None)
        
        working_path = self.config.GENERAL_INFO.get('working_path', 'tmp')
        if not os.path.exists(working_path): os.mkdir(working_path)
        
        if self._model == 'dssat': ## TODO IMPLEMETN OTHER CROP MODELING SYSTEMS
            super().__init__(working_path)
        
    def __init__(self, 
        configuration_path: Optional[str] = None,
        configuration_dict: Optional[dict] = None,
        ) -> None:
        self.config = None
        self._geodata = None
        self._climate = None
        self._soil = None
        
        if configuration_path:
            self.config = OmegaConf.load(configuration_path)
        elif configuration_dict:
            self.config = OmegaConf.create(configuration_dict)
        else:
            raise ValueError("Either `configuration_path` or `configuration_dict` must be provided.")

        self._setup()
        
    def create_roi_sp_data(self, roi_index: Optional[int] = None,
        roi: Optional[gpd.GeoDataFrame] = None,
        crs: str = "EPSG:4326",
        group_codes: Optional[dict] = None,
        create_group_splayer = False
    ) -> Optional[Path]:
        """
        Extracts and processes data for a region of interest (ROI).

        Parameters
        ----------
        roi_index : int, optional
            Index of the ROI in `geo_features`.
        roi : gpd.GeoDataFrame, optional
            Geodataframe representing the region of interest.
        crs : str, default="EPSG:4326"
            Coordinate Reference System (CRS) for output data.
        group_codes : dict, optional
            Group codes for aggregating soil or climate data.

        Returns
        -------
        Optional[Path]
            Path to the temporary directory containing DSSAT files, or None if no data is found.
        """
        group_by = self.config.SPATIAL_INFO.get('aggregate_by', None)
        if roi_index:
            roi = self.geo_features.iloc[roi_index:roi_index+1]
        if roi is None:
            raise ValueError("Provide either an ROI index or a GeoDataFrame for the region of interest.")
        
        roi = roi.to_crs(self.climate.rio.crs)
        roi_name = roi[self.config.SPATIAL_INFO.feature_name].values[0]
        # extract individual data
        weatherm, soilm = get_roi_data(roi, self.climate, self.soil, aggregate_by=group_by, scale_factor= self.config.SPATIAL_INFO.scale_factor)
        ## check both have data
        datainweather = all(not all(np.isnan(np.unique(weatherm.isel(date = 0)[var].values))) for var in list(weatherm.data_vars.keys()))
        datainsoil = all(not all(np.isnan(np.unique(soilm.isel(depth = 0)[var].values))) for var in list(soilm.data_vars.keys()))
    
        if not (datainweather and datainsoil): return None
        weatherm = WeatherTransformer()(weatherm)
        # create folders 
        self.set_up(country = self.config.GENERAL_INFO.country, site = roi_name)
        
        if group_by and create_group_splayer:
            self.group_spatial_layer(soilm)
            
        # export data as dssat files
        for data, datatype in zip([weatherm, soilm],['climate', 'soil']):
            self.from_datacube_to_dssatfiles(data, data_source=datatype, 
                                                target_crs = crs, group_by = group_by, group_codes = group_codes)
        
        return self._tmp_path
    
    def group_spatial_layer(self, soildata):
        group_by = self.config.SPATIAL_INFO.get('aggregate_by', None)
        grouplayer = soildata.isel(depth = 0)[group_by]
        grouplayer = grouplayer.drop_vars('depth')
        try:

            grouplayer = grouplayer.rio.reproject('EPSG:4326')
            grouplayer.rio.to_raster(os.path.join(self._tmp_path, f'{group_by}.tif'))
        except:
            
            grouplayer = reproject_xrdata(grouplayer, 'EPSG:4326')
            grouplayer = grouplayer.rio.write_crs('EPSG:4326')
            grouplayer.rio.to_raster(os.path.join(self._tmp_path, f'{group_by}.tif'))