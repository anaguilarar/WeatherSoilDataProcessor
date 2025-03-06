import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
import os
import rasterio as rio
import shutil

from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Dict, Optional, Union, Tuple
from pathlib import Path

from .utils.process import get_crs_fromxarray,set_encoding, check_crs_inxrdataset, summarize_datacube_as_df, model_selection
from .utils.u_soil import get_layer_texture, get_soil_datacube
from .utils.u_weather import get_weather_datacube, WeatherTransformer

from spatialdata.datacube import create_dimension, reproject_xrdata
from spatialdata.gis_functions import masking_rescaling_xrdata, list_tif_2xarray
from spatialdata.xr_dict import CustomXarray, from_dict_toxarray

import concurrent.futures

import logging

def reproject_xarray(xrdata, target_crs, src_crs = None):
    if src_crs is None:
        try:
            src_crs = xrdata.rio.crs
        except:
            src_crs = xrdata.attrs.get('crs', None)
            
    assert src_crs is not None, "Please provide the source crs"
                        
    return xrdata.rio.write_crs(src_crs).rio.reproject(target_crs)

import xarray


def create_date_raster(idx, ref_raster, model_data, ycol_name = 'HWAH'):
    import rioxarray as rio
    tmparray = np.full_like(ref_raster.values, np.nan).flatten()
    
    for k,v in model_data.items():
        tmparray[int(k)] = v[ycol_name].values[idx]

    return tmparray.reshape(ref_raster.values.shape)

def create_mlt_yield_raster(ref_raster, model_data, ycol_name = 'HWAH' ):
    #assert os.path.exists(ref_raster_path)
    
    _, v = next(iter(model_data.items()))
    dates = v.output_data().sort_values('PDAT')[['PDAT']]
    
    ref_raster_c = ref_raster.copy()
     
    alldata = {k:v.output_data().sort_values('PDAT')[[ycol_name,'PDAT']] for k, v in model_data.items()}
    rasterlis = []
    for idate in tqdm(range(dates.PDAT.values.shape[0])):
        img_vals = create_date_raster(idate, ref_raster_c, alldata, ycol_name = ycol_name)
        rasterdata = list_tif_2xarray(np.array(img_vals), transform=ref_raster_c.rio.transform(),crs=ref_raster_c.rio.crs, bands_names=['HWAH'],depth_dim_name='date', dimsformat='CHW')
        #img_vals.attrs['long_name'] = ycol_name
        rasterlis.append(rasterdata.expand_dims('date'))
    rasterd = xarray.concat(rasterlis, dim = 'date')
    rasterd = rasterd.assign_coords(date =  dates.PDAT.values,
                                    x =  ref_raster_c.x.values, y =  ref_raster_c.y.values)
    #rasterd.date = dates.PDAT.values
    rasterd.attrs['transform'] = ref_raster_c.rio.transform()
    
    return rasterd
    
    

def get_roi_data(roi: gpd.GeoDataFrame,
    weather_datacube: xarray.Dataset,
    soil_datacube: Union[xarray.Dataset, dict],
    scale_factor: int = 10,
    resample_ref: str = 'weather',
    buffer = None) -> Tuple[xarray.Dataset, xarray.Dataset, Optional[xarray.Dataset]]:
    """
    Extracts and processes climate, soil, and DEM data for a given region of interest (ROI).

    Parameters
    ----------
    roi : geopandas.GeoDataFrame
        Region of interest with projected geographic data.
    weather_datacube : xarray.Dataset
        Climate data as a multi-dimensional xarray dataset.
    soil_datacube : Union[xarray.Dataset, dict]
        Soil data as an xarray dataset or a dictionary of datasets.
    dem_data : Optional[xarray.Dataset], optional
        Digital elevation model data, by default None.
    aggregate_by : Optional[str], optional
        Aggregation criterion for soil data (e.g., "texture"), by default None.
    min_area : int, optional
        Minimum area (in km²) for ROI; if the area is smaller, buffering is applied, by default 15.
    scale_factor : int, optional
        Scale factor for downsampling or upsampling the data, by default 10.

    Returns
    -------
    Tuple[xarray.Dataset, xarray.Dataset, Optional[xarray.Dataset]]
        A tuple containing:
        - Processed weather data as an xarray dataset.
        - Processed soil data as an xarray dataset.
        - Processed DEM data as an xarray dataset (if provided).

    Raises
    ------
    ValueError
        If `roi` has an invalid area or unsupported aggregation criterion.
    """
    
    if resample_ref == 'weather':
        weather_datacube_m = masking_rescaling_xrdata(weather_datacube, roi, buffer=buffer, scale_factor=scale_factor, return_original_size=True, method = 'nearest')
        xr_reference = weather_datacube_m.isel(date = 0)
        weather_datacube_m.attrs['crs'] = get_crs_fromxarray(weather_datacube)    
        
        if isinstance(soil_datacube, dict):
            soil_datacube_m = {k: masking_rescaling_xrdata(v, roi, buffer=buffer, resample_ref =xr_reference)  for k,v in tqdm(soil_datacube.items())}
            soil_datacube_m = create_dimension(soil_datacube_m, newdim_name = 'depth', isdate = False) 
        else:
            soil_datacube_m = masking_rescaling_xrdata(soil_datacube, roi, buffer=buffer, resample_ref =xr_reference, return_original_size=True, method = 'nearest')
        soil_datacube_m.attrs['crs'] = get_crs_fromxarray(weather_datacube_m)
    
    elif resample_ref == 'soil':
        soil_datacube_m = masking_rescaling_xrdata(soil_datacube, roi, buffer=buffer, scale_factor=scale_factor, return_original_size=True,  method = 'nearest')
        soil_datacube_m.attrs['crs'] = get_crs_fromxarray(soil_datacube)
        xr_reference = soil_datacube_m.isel(depth = 0)
        
        weather_datacube_m = masking_rescaling_xrdata(weather_datacube, roi, buffer=buffer, resample_ref =xr_reference, return_original_size=True, method = 'nearest')
        weather_datacube_m.attrs['crs'] = get_crs_fromxarray(soil_datacube_m)

    return weather_datacube_m, soil_datacube_m


def add_layer_texture_to_datacubes(weather_dc, soil_dc, dem_dc = None):
        weatherdatavars = list(weather_dc.data_vars.keys())
        soilref = get_layer_texture(soil_dc.isel(depth = 0))
        # merge texture to weather and soil
        weather_dc = xarray.merge([weather_dc,soilref])[weatherdatavars+ ['texture']]
        soil_dc = xarray.merge([soil_dc,soilref['texture']])[list(soil_dc.data_vars.keys())+ ['texture']]
        if dem_dc is not None:
            dem_dc = xarray.merge([dem_dc,soilref['texture']])[list(dem_dc.data_vars.keys())+ ['texture']]
        return weather_dc, soil_dc, dem_dc

    
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
            with xarray.open_dataset(filepath, engine = engine, chunks  = 'auto') as ds:
                data = ds.copy()
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
        dcengine = 'netcdf4' if self._dcengine is None else self._dcengine
        encoding = set_encoding(xrdata)
        xrdata = check_crs_inxrdataset(xrdata)
        xrdata.to_netcdf(fn, encoding = encoding, engine = dcengine)
        
    def _setup(self):
        self.climate = None
        self.soil = None
        self.ndvi = None
        self._dcengine= None
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

class SpatialCM():
    
        
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
    
    def __init__(self, 
        configuration_path: Optional[str] = None,
        configuration_dict: Optional[dict] = None,
        ) -> None:
        self.config = None
        self._geodata = None
        self._climate = None
        self._soil = None
        self._dem = None
        
        if configuration_path:
            self.config = OmegaConf.load(configuration_path)
        elif configuration_dict:
            self.config = OmegaConf.create(configuration_dict)
        else:
            raise ValueError("Either `configuration_path` or `configuration_dict` must be provided.")

        self._setup()
    
    def _read_geosp_data(self, attr):
        path = self.config.SPATIAL_INFO.get(attr,None)
        if path is None:
            return None
        elif not os.path.exists(path):
            raise FileNotFoundError(f"{attr} data path not found: {path}")
        else:
            return SpatialData()._open_dataset(path)

    @property
    def crop(self):
        return self.config.CROP.get('name', None)
    
    @property
    def cultivar(self):
        return self.config.CROP.get('cultivar', None)
        
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
            self._climate = self._read_geosp_data('weather_path')
            
        return self._climate        
        
    @property
    def soil(self):
        """
        Lazy-loaded property for soil data.
        """
        if self._soil is None:
            self._soil = self._read_geosp_data('soil_path')
        return self._soil
    
    @property
    def dem(self):
        if self._dem is None:
            self._dem = self._read_geosp_data('dem_path')
            
        return self._dem  
    
    def _setup(self):
        """
        Initializes essential properties and validates configurations.
        """
        self.soil  # Load soil dataset
        self.climate  # Load climate dataset
        self.dem
        
        model = self.config.GENERAL_INFO.get('model', None)
        working_path = self.config.GENERAL_INFO.get('working_path', 'tmp')
        if not os.path.exists(working_path): os.mkdir(working_path)
        self.model = model_selection(model, working_path)
        self._ncores = self.config.GENERAL_INFO.get('ncores', 3)
        
        
    def _buffer(self,roi, min_area = 15, bufferfactor = 4):
        area = roi.area.values[0]/ (1000*1000)
        if area < min_area:
            narea = (min_area*bufferfactor) - area
            buffer = (narea*100)
        else:
            buffer = bufferfactor*100 
            
        return buffer
    
    def get_masked_data(self, roi, buffer, verbose = True):
        """
            Retrieves masked spatial data for a region of interest (ROI).
            Parameters
            ----------
            roi : Any
                The region of interest for which the data is to be retrieved.
            buffer : Any
                The buffer distance around the ROI.
            verbose : bool, optional
                If True, logs information about the data loading process. Defaults to True.
            Returns
            -------
            tuple
                A tuple containing weather data, soil data, and DEM data (if available).
        """
        demm, weatherm, soilm  = None, None, None
        
        if os.path.exists(self._soil_tmppath) and os.path.exists(self._weather_tmppath):
            if verbose: logging.info("Loading data from pre-existing files")
            soilm = SpatialData()._open_dataset(self._soil_tmppath)
            weatherm = SpatialData()._open_dataset(self._weather_tmppath)
            demm = SpatialData()._open_dataset(self._dem_tmppath) if os.path.exists(self._dem_tmppath) else None
        else:
            if verbose: logging.info("Extracting spatial data from source")
            weatherm, soilm = get_roi_data(roi, self.climate, self.soil, scale_factor= self.config.SPATIAL_INFO.scale_factor, buffer= buffer)
        if demm is None and self.model.name in ['caf','simple_model']:
            if verbose: logging.info("Creating DEM file")
            assert self.dem is not None, "Please provide DEM data, check DEM path "
            demm = masking_rescaling_xrdata(self.dem, roi, buffer=buffer, resample_ref =weatherm.isel(date = 0), return_original_size=True, method = 'nearest')
            demm.attrs['crs'] = get_crs_fromxarray(weatherm)
            
        return weatherm, soilm, demm

    def export_spatialdata_asnc(self, weatherm, soilm, demm = None):
        for data, name in zip([weatherm, soilm, demm],['weather','soil','dem']):
            if data is not None and not os.path.exists(self.__dict__.get(f'_{name}_tmppath',"")):
                data.attrs['dtype'] = 'float'
                SpatialData()._save_asnc(data, fn = os.path.join(self._tmp_path, f'{name}_.nc'))
                
    
    def create_roi_sp_data(self, roi_index: Optional[int] = None,
        roi: Optional[gpd.GeoDataFrame] = None,
        crs: str = "EPSG:4326",
        group_codes: Optional[dict] = None,
        create_group_splayer = False,
        export_spatial_data = False,
        verbose = False
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
        demm = None
        
        group_by = self.config.SPATIAL_INFO.get('aggregate_by', None)
        if roi_index:
            roi = self.geo_features.iloc[roi_index:roi_index+1]
        if roi is None:
            raise ValueError("Provide either an ROI index or a GeoDataFrame for the region of interest.")
        
        # create buffer
        roi = roi.to_crs(self.climate.rio.crs)
        buffer = self._buffer(roi)
        self.country = self.config.GENERAL_INFO.get('country', None)
        
        ## get data
        weatherm, soilm, demm = self.get_masked_data(roi, buffer, verbose = verbose)
        
        ## check both sources must have data
        datainweather = all(not all(np.isnan(np.unique(weatherm.isel(date = 0)[var].values))) for var in list(weatherm.data_vars.keys()))
        datainsoil = all(not all(np.isnan(np.unique(soilm.isel(depth = 0)[var].values))) for var in list(soilm.data_vars.keys()))
        if not (datainweather and datainsoil): return None # no data foun
        weatherm = WeatherTransformer()(weatherm)
        
        # export spatial data
        if export_spatial_data:
            self.export_spatialdata_asnc(weatherm, soilm, demm)

        if group_by == 'texture':
            weatherm, soilm, demm = add_layer_texture_to_datacubes(weatherm, soilm, demm)
        
        if self.model.name == 'dssat':
            # export data as dssat files
            self.process_dssat_files(weatherm, soilm, group_by, create_group_splayer, group_codes, crs)
        if self.model.name in ['caf', 'simple_model']:
            if create_group_splayer: self.group_spatial_layer(soilm)
            for data, datatype in zip([weatherm, soilm, demm],['climate', 'soil', 'dem']):
                if data is not None:
                    self.model.from_datacube_to_files(data, data_source= datatype, target_crs=crs, group_by = group_by, group_codes = group_codes,
                                                outputpath= self._tmp_path)
                
        return self._tmp_path
    
    def process_dssat_files(self, weatherm, soilm, group_by, create_group_splayer, group_codes, crs: str = "EPSG:4326"):
        pixel_scale =  group_by == 'pixel'

        if not pixel_scale:
            if group_by and create_group_splayer: self.group_spatial_layer(soilm)
                
            for data, datatype in zip([weatherm, soilm],['climate', 'soil']):
                
                dim_name = 'date' if datatype == 'climate' else 'depth'
                dfdata = summarize_datacube_as_df(data, dimension_name= dim_name, group_by = group_by, 
                                                project_to= crs, pixel_scale = pixel_scale)
                
                self.model.from_datacube_to_files(dfdata, data_source=datatype, 
                                                    group_by = group_by, 
                                                    group_codes = group_codes,
                                                    outputpath= self._tmp_path,
                                                    country = self.country.upper(),
                                                    site = self.site)
            self.check_dssat_env_paths()
                
        if pixel_scale:
            xrref, pxs_withdata = self.create_env_variables_at_pixellevel(weatherm, soilm, target_crs =crs)
            pd.DataFrame(pxs_withdata, columns = ['pixel','x','y']).to_csv(os.path.join(self._tmp_path, 'pixel_coords.csv'))
            xrref[list(xrref.data_vars)[0]].rio.to_raster(os.path.join(self._tmp_path, 'ref_raster.tif'))
            
                
    def _process_individual_pixel(self, env_list, env_list_names, idpx, x, y):
        if not np.isnan(x) and not np.isnan(y):
            #x, y = float(commoncoords[idpix].split('_')[0]),float(commoncoords[idpix].split('_')[1])
            for data, datatype in zip(env_list,env_list_names):
                dfdata = data.sel(x = x, y = y, method = 'nearest').to_dataframe().reset_index().dropna()
                if dfdata.shape[0]>0:
                    self.model.from_datacube_to_files(dfdata, data_source=datatype, 
                                                outputpath= self._tmp_path,
                                                country = self.country.upper(),
                                                site = self.site,
                                                sub_working_path = str(idpx), verbose = False)
        return x, y
    
    def create_env_variables_at_pixellevel(self, weatherm, soilm, target_crs = "EPSG:4326"):
        """
        create environment variables for running crop models at pixel level

        Parameters
        ----------
        weatherm : xarray.Dataset
            Weather datacube.
        soilm : xarray.Dataset
            Soil datacube.
        target_crs : str, optional
            Target coordinate reference system (CRS) for reprojecting data, by default "EPSG:4326".
            
        """
        #reproject
        
        if target_crs:
            soilm = soilm.rio.reproject(target_crs)
            weatherm = weatherm.rio.reproject(target_crs)
            
        soilm = soilm.where(soilm[list(soilm.data_vars)[0]]<3.4028234663852886e+20, np.nan)
        weatherm = weatherm.where(weatherm[list(weatherm.data_vars)[0]]<3.4028234663852886e+20, np.nan)
        
        xrref = xarray.merge([weatherm.isel(date = 0), soilm.isel(depth = 0)])
        xrmask = xrref.notnull()[list(weatherm.data_vars)[0]] * xrref.notnull()[list(soilm.data_vars)[0]]
        #commoncoords = xrref.to_dataframe().reset_index().dropna().apply(lambda x: f'{x.x}_{x.y}', axis = 1).unique()
        xgrid, ygrid = np.meshgrid(xrref.x,xrref.y)
        xgrid = np.where(xrmask.values,xgrid,np.nan).flatten()
        ygrid = np.where(xrmask.values,ygrid,np.nan).flatten()
        pxswithdata = np.where(~np.isnan(ygrid))[0]
        processed_pxs = []
        #for idpix in range(commoncoords.shape[0]):
        with tqdm(total=len(pxswithdata)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._ncores) as executor:
                future_to_tr ={executor.submit(self._process_individual_pixel, [weatherm, soilm], ['climate', 'soil'],
                                            idpx, xgrid[idpx], ygrid[idpx]): (idpx) for idpx in pxswithdata}

                for future in concurrent.futures.as_completed(future_to_tr):
                    idpx = future_to_tr[future]
                    try:
                        x,y = future.result()
                        processed_pxs.append([idpx, x, y])
                            
                    except Exception as exc:
                            print(f"Request for treatment {idpx} generated an exception: {exc}")
                    pbar.update(1)
                    
        self.check_dssat_env_paths()
        return xrref, processed_pxs

    def set_up_folders(self, site = None) -> None:
        """
        Set up the working directory, temporary paths, and general information about the site and country.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to configure site and country.

            - site : str, optional
                Site name to be used as a subdirectory for temporary files.
        """
        assert os.path.exists(self.model.path)
        
        self.site = site
        if self.site is None:
            self._tmp_path = os.path.join(self.model.path, 'tmp')
        else:
            self._tmp_path = os.path.join(self.model.path, self.site)
            
        if not os.path.exists(self._tmp_path): os.mkdir(self._tmp_path)
        
        self._weather_tmppath = os.path.join(self._tmp_path, 'weather_.nc')
        self._soil_tmppath = os.path.join(self._tmp_path, 'soil_.nc')
        self._dem_tmppath = os.path.join(self._tmp_path, 'dem_.nc')
        
    def check_dssat_env_paths(self):

        subprocess_paths_weather = self.model.find_envworking_paths(self._tmp_path, 'WTH')
        subprocess_paths_soil = self.model.find_envworking_paths(self._tmp_path, 'SOL')
        if len(subprocess_paths_soil) != len(subprocess_paths_weather):
            longestpath = subprocess_paths_soil if len(subprocess_paths_soil)>len(subprocess_paths_weather) else subprocess_paths_weather
            smallest_path = subprocess_paths_weather if len(subprocess_paths_soil)>len(subprocess_paths_weather) else subprocess_paths_soil
            for spath in longestpath:
                if not spath in smallest_path and os.path.exists(spath):
                    shutil.rmtree(spath, ignore_errors=False, onerror=None)
            
            self.model._process_paths = smallest_path
            
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