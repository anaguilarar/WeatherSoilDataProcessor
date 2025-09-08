import os
import shutil
import copy
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional

import cdsapi
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
import tqdm
import xarray

import os
import cdsapi
import copy
from typing import List, Optional, Dict
import concurrent.futures
import pandas as pd
from shapely.geometry import Polygon

from .datacube import DataCubeBase
from .files_manager import (
    create_yearly_query,
    days_range_asstring,
    find_date_instring,
    months_range_asstring,
    split_date,
    uncompress_zip_path
)
from .gis_functions import (
    from_polygon_2bbox,
    from_xyxy_2polygon,
    list_tif_2xarray,
    read_raster_data
)
from .utils import download_file


def transform_dates_for_AgEraquery(
        year: int, 
        init_day: Optional[int] = None, 
        end_day: Optional[int] = None, 
        init_month: Optional[int] = None, 
        end_month: Optional[int] = None
        ) -> Dict[str, List[str]]:
    """Transforms date components into a query dictionary for AgEra5.

    This function generates lists of days and months as strings, formatted
    for use in a cdsapi query.

    Parameters
    ----------
    year : int
        The year for the query.
    init_day : int, optional
        The starting day of the month (1-31). Defaults to 1.
    end_day : int, optional
        The ending day of the month (1-31). Defaults to 31.
    init_month : int, optional
        The starting month of the year (1-12). Defaults to 1.
    end_month : int, optional
        The ending month of the year (1-12). Defaults to 12.

    Returns
    -------
    dict[str, list[str]]
        A dictionary containing formatted 'year', 'month', and 'day' keys
        suitable for an AgEra5 data request.

    Example
    -------
    >>> transform_dates_for_AgEraquery(2023, init_day=1, end_day=10, init_month=3, end_month=4)
    {'year': '2023', 'month': ['03', '04'], 'days': ['01', '02', ..., '09']}
    """

    init_day = 1 if init_day is None else init_day
    init_month = 1 if init_month is None else init_month
    end_month = 12 if end_month is None else end_month
    end_day = 31 if end_day is None else end_day

    days = days_range_asstring(init_day, end_day)
    months = months_range_asstring(init_month, end_month)    
    return {
        'year': [str(year)],
        'month': months,
        'day': days
    }


def download_mlt_data_from_agera5(
    variable: str, 
    starting_date: str, 
    ending_date: str, 
    output_folder: str, 
    aoi_extent: List[float], 
    product: str= "sis-agrometeorological-indicators", 
    statistic: Optional[str] = None,
    ncores: int = 10,
    version: str ="2_0",
    max_attempts = 3
) -> None:
    """
    Download multiple layers of data from AgEra5 for a given variable and time range.

    Parameters
    ----------
    variable : str
        The variable to download (e.g., 'temperature', 'precipitation', 'solar_radiation_flux').
    starting_date : str
        The start date in 'YYYY-MM-DD' format.
    ending_date : str
        The end date in 'YYYY-MM-DD' format.
    output_folder : str
        Folder path to save the downloaded files.
    aoi_extent : List[float]
        Area of interest (bounding box) in the format [lat_max, lon_min, lat_min, lon_max].
    product : str
        AgEra5 product type (e.g., 'sis-agrometeorological-indicators').
    statistic : Optional[str], optional
        Statistic to be retrieved (if applicable). Defaults to None.
    statistic : Optional[str], optional
        Product Version currently there is 1_1  and 2_0. Defaults 1_1.
    statistic : Optional[INT], optional
        Max number of download attempts.
    Returns
    -------
    None
        Downloads data and saves it in the specified output folder.

    Example
    -------
    >>> download_mlt_data_from_agera5('temperature', '2023-01-01', '2023-12-31', './data', [-10, 35, 10, 50], 'sis-agrometeorological-indicators')
    """
    def download_one_year_data(year, query_dict, init_day, end_day, init_month, end_month, init_year, end_year):
        try:
            if year == init_year:
                datesquery = transform_dates_for_AgEraquery(year, init_day = init_day, end_day = 31, init_month = init_month, end_month = 12)
            if year == end_year:
                datesquery = transform_dates_for_AgEraquery(year, init_day = 1, end_day = end_day, init_month = 1, end_month = end_month)
            else:
                datesquery = transform_dates_for_AgEraquery(year)

            year_query = copy.deepcopy(query_dict)
            year_query.update(datesquery) 
            filename = os.path.join(output_folder, "{}.zip".format(year))
            
            print(f"Requesting data for year {year} with query: {year_query}")
            
            client = cdsapi.Client()
            client.retrieve(product, year_query, filename)
            print(f"Successfully downloaded data for year {year} to {filename}")
            return filename
        except Exception:
            return None
    
    init_year, init_month, init_day = split_date(starting_date)
    end_year, end_month, end_day = split_date(ending_date)


    years = list(range(init_year,end_year+1))
    query_dict = {"version": version,
                    "area":aoi_extent,
                    "variable": variable if isinstance(variable, list) else [variable],
                    }
    
    if statistic is not None: query_dict.update({"statistic":  statistic})
        
    file_path_peryear = {}
    
    if ncores>0:
            
        tasks_to_retry = {year: 1 for year in years}  # {year: attempt_count}
        
        while tasks_to_retry:
            tasks_this_round = tasks_to_retry.copy()
            tasks_to_retry.clear()

            with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
                future_to_year = {
                    executor.submit(download_one_year_data, year, query_dict, init_day, end_day, init_month, end_month, init_year, end_year): year
                    for year in tasks_this_round
                }

                for future in concurrent.futures.as_completed(future_to_year):
                    year = future_to_year[future]
                    attempt_num = tasks_this_round[year]
                    try:
                        file_path = future.result()
                        if file_path:
                            file_path_peryear[str(year)] = file_path
                        else:
                            raise Exception("Download function returned None")
                    except Exception as exc:
                        print(f"Attempt {attempt_num} for year {year} failed: {exc}")
                        if attempt_num < max_attempts:
                            tasks_to_retry[year] = attempt_num + 1
                        else:
                            print(f"Failed to download data for year {year} after {max_attempts} attempts.")
    else:
        for year in years:
            for attempt in range(1, max_attempts + 1):
                print(f"Downloading year {year} (Attempt {attempt}/{max_attempts})...")

                result_filepath = download_one_year_data(year, query_dict, init_day, end_day, init_month, end_month, years[0], years[-1])
                if result_filepath:
                    file_path_peryear[str(year)] = result_filepath
                    break
            else:
                print(f"Failed to download data for year {year} after {max_attempts} attempts.")
                
    return file_path_peryear
    

class CHIRPS_download:
    """Handles the downloading of CHIRPS precipitation data.

    This class provides methods to construct URLs, query by date range,
    and download data for specified areas of interest.

    Parameters
    ----------
    frequency : str, optional
        The temporal frequency of the data ('daily', 'monthly', etc.). 
        Defaults to 'daily'.
    sp_resolution : str, optional
        The spatial resolution ('05' for 0.05 degree). Defaults to '05'.
    """


    def __init__(self, frequency: str = 'daily', sp_resolution: str = '05', version = 'v2.0') -> None:
        self._frequency = frequency
        self.resolution = sp_resolution
        self.version = version
        #self._url = if ver
        #TODO: implement version 3 options era https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/ERA5/2000/
        #                                  IMERGLATE https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/IMERGlate-v07/

    def set_url(self, year, date):

        return "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_{}/cogs/p{}/{}/chirps-v2.0.{}.cog".format(
            self._frequency,
            self.resolution,
            year,
            date
        )

    def current_date(self):
        self._current_date + 1

    def set_area(self, xyxy):
        self.extent = gpd.GeoSeries([from_xyxy_2polygon(*xyxy)])

    @staticmethod
    def set_days(init_day, end_day):
        return days_range_asstring(init_day, end_day)
    
    def _create_yearly_query(self):
        return create_yearly_query(self._initdate, self._enddate)
    
    def download_data_per_year(self, year:str, output_path:str, extent:List[float]) -> str:
        """Downloads and compiles CHIRPS data for a single year into one NetCDF file.

        Instead of saving one file per day, this method downloads all daily
        data for a given year, stacks it into a single xarray.Dataset, and
        saves it as one compressed NetCDF file.

        Parameters
        ----------
        year : str
            The year to download data for.
        output_path : str
            The directory to save the annual NetCDF file.
        extent : list[float]
            The bounding box [xmin, ymin, xmax, ymax] for the data.

        Returns
        -------
        str or None
            The path to the created NetCDF file, or None if no data was found.
        """
        if not os.path.exists(os.path.join(output_path,year)): os.mkdir(os.path.join(output_path,year)) 
        
        for month in self._date[year].keys():
            stackimages = []
            for day in self._date[year][month]:
                date_str = '{}.{}.{}'.format(year, month,day)
                urlpath = self.set_url(year, date_str)
                print(urlpath)
                with rasterio.open(urlpath) as src:
                    meta = src.profile
                    masked, mask_transform = mask(dataset=src, shapes=gpd.GeoSeries([from_xyxy_2polygon(*extent)]), crop=True)
                    stackimages.append(masked)
                xrm = list_tif_2xarray(masked, mask_transform, crs=str(meta['crs']), bands_names=['precipitation'], dimsformat= 'CHW')
                xrm.to_netcdf(os.path.join(output_path,year, 'chirps_precipitation_{}{}{}.nc'.format(year,month,day)))

        return os.path.join(output_path,year)
        
    def download_chirps(self, extent: List[float], init_date: str, ending_date: str, output_path:str = None, ncores: int = 10):
        """Downloads CHIRPS data for a given area and date range.

        Parameters
        ----------
        extent : list[float]
            The bounding box [xmin, ymin, xmax, ymax] for the data.
        init_date : str
            The starting date in 'YYYY-MM-DD' format.
        ending_date : str
            The ending date in 'YYYY-MM-DD' format.
        output_path : str
            The directory to save downloaded files.
        ncores : int, optional
            Number of parallel workers for downloading. Defaults to 10.

        Returns
        -------
        dict[str, str]
            A dictionary mapping each year to the path of the downloaded file.
        """
        self._initdate = init_date
        self._enddate = ending_date
        self._current_date = init_date

        self._date = self._create_yearly_query()
        file_path_peryear = {}
        if ncores > 0:
    
            with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
                future_to_year ={executor.submit(self.download_data_per_year, year, output_path, extent): (year) for year in self._date.keys()}
                
                for future in concurrent.futures.as_completed(future_to_year):
                    year = future_to_year[future]
                    try:
                        file_path = future.result()
                        file_path_peryear[str(year)] = file_path
                        print(f"Requested Year {year} \n downloaded in {file_path}")
                    except Exception as exc:
                        print(f"Request for year {year} generated an exception: {exc}")
        else:
            for year in self._date.keys():
                file_path_peryear[str(year)] = self.download_data_per_year(year, output_path, extent)
                
        return file_path_peryear
def process_file(year_path_folder:str, filename:str, date:str, xdim_name:str, ydim_name:str, depthdim_name:str):
    """Reads and processes a single NetCDF file into an xarray.Dataset.

    Parameters
    ----------
    year_path_folder : str
        The path to the folder containing the file.
    filename : str
        The name of the NetCDF file.
    date : str
        The date string ('YYYYMMDD') associated with the file.
    xdim_name : str
        The name of the x-dimension (e.g., 'longitude').
    ydim_name : str
        The name of the y-dimension (e.g., 'latitude').
    depthdim_name : str
        The name for the new time dimension.

    Returns
    -------
    xarray.Dataset
        A dataset containing the data from the file with a new time dimension.
    """
    dateasdatetime = datetime.strptime(date, '%Y%m%d')
    filepath = os.path.join(year_path_folder, filename)
    xrdata = read_raster_data(filepath, ydim_name=ydim_name, xdim_name=xdim_name)
    
    # Assuming only one variable in the file
    varname = list(xrdata.data_vars.keys())[0]

    # Reorganize data
    two_var = xrdata[varname].values[0] if len(xrdata[varname].values.shape) == 3 else xrdata[varname].values
    xrdata = xarray.Dataset(
        data_vars={str(dateasdatetime.year): ([ydim_name, xdim_name], two_var)},
        coords={xdim_name: xrdata[xdim_name].values, ydim_name: xrdata[ydim_name].values}
    )
    
    # Add a time dimension
    xrdata = xrdata.expand_dims(dim={depthdim_name: 1}, axis=0)
    xrdata[depthdim_name] = [dateasdatetime]

    return xrdata


def read_annual_data(path: str, year:str,xdim_name: str = 'longitude',
                        ydim_name: str = 'latitude',
                        depthdim_name: str = 'time',
                        crs: str = 'EPSG:4326'):
    """
    Reads annual data from NetCDF files for a given year and compiles it into a multi-temporal xarray Dataset.

    Parameters:
    -----------
    path : str
        The folder path containing the yearly data.
    year : str
        The year for which data needs to be read.
    xdim_name : str, optional
        Name of the x-dimension (longitude), default is 'longitude'.
    ydim_name : str, optional
        Name of the y-dimension (latitude), default is 'latitude'.
    depthdim_name : str, optional
        Name of the depth dimension (time), default is 'time'.
    crs : str, optional
        Coordinate Reference System (CRS), default is 'EPSG:4326'.
    
    Returns:
    --------
    annual_data : xarray.Dataset
        A concatenated Dataset with time as one of the dimensions.
    """
        
    # find folder path
    year_path_folder = uncompress_zip_path(path, year)
    #get dates and filenmaes with the extension
    times = ([[fn, find_date_instring(fn, pattern=year)] for fn in os.listdir(year_path_folder) if fn.endswith('.nc')])
    
    #read data
    list_xrdata = [process_file(os.path.join(path, year), fn, date, 
                                ydim_name, xdim_name, depthdim_name) for fn, date in times]
    
    annual_data = xarray.concat(list_xrdata, dim = depthdim_name)
    
    tmp = list_xrdata[0].copy().rio.write_crs(crs)
    spatial_ref = tmp.rio.write_transform(tmp.rio.transform()).spatial_ref
    annual_data =  annual_data.assign(crs = spatial_ref)

    if 'spatial_ref' in list(annual_data.coords.keys()):
        return annual_data.drop_vars('spatial_ref')
    else:
        return annual_data
    
class ClimateDataDownload(object):
    """Manages downloading of climate data from various sources.

    This class orchestrates the download of different weather variables like
    temperature, precipitation, and solar radiation from sources such as
    AgEra5 and CHIRPS.

    Parameters
    ----------
    starting_date : str
        The start date for data queries, 'YYYY-MM-DD'.
    ending_date : str
        The end date for data queries, 'YYYY-MM-DD'.
    aoi : Polygon, optional
        A shapely Polygon defining the area of interest.
    xyxy : list[float], optional
        A bounding box [xmin, ymin, xmax, ymax]. Used if `aoi` is not provided.
    output_folder : str
        The root folder where all downloaded data will be stored.
    """
    def __init__(self, starting_date: str, ending_date: str, 
                 aoi: Optional[Polygon] = None, xyxy: Optional[List] = None, 
                 output_folder: str = None) -> None:

        """
        initialize climate data class
        ----------
        Parameters: 

        starting_date : str
            starting date query period with format 'YYYY-mm-dd'
        starting_date : str
            ending date query period with format 'YYYY-mm-dd'
        aoi : Polygon
            boundary box polygon
        xyxy : List
            Boundary box extent [xmin, ymin, xmax, ymax]
        output_folder : str
            output folder

        """
        self.output_folder = None
        self._aoi = aoi
        self._init_date = starting_date
        self._ending_date = ending_date

        self.aoi_extent = from_polygon_2bbox(self._aoi) if self._aoi else xyxy
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            self.output_folder = output_folder
            
    @property
    def _urls(self):
        return {'datacube': 'https://zarr-query-api-rv7rkv4opa-uc.a.run.app/v1/',
        'chirps': 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_{}/cogs/p{}/{}/chirps-v2.0.{}.cog', ##frequency,resolution,year,date
        'agera5': 'sis-agrometeorological-indicators'
        }

    def download_from(self):
        """
        Specifies which data source to use for each weather variable.

        Returns
        -------
        dict
            Mapping of weather variables to data sources.
        """

        return {   'precipitation': 'chirps',
            'temperature': 'agera5',
            'solar_radiation': 'agera5',
            'wind_speed': 'agera5',
            'relative_humidity': 'agera5',
            'vapour_pressure': 'agera5'
            }
        
    def missions(self):
        """
        Specifies which mission is used for each weather variable.

        Returns
        -------
        dict
            Mapping of weather variables to missions (data source providers).
        """

        return {
            'precipitation': 'chirps',
            'temperature': 'agera5',
            'wind_speed': 'agera5',
            'solar_radiation': 'agera5',
            'relative_humidity': 'agera5',
            'vapour_pressure': 'agera5'
        }

    @staticmethod
    def _create_dowload_folder(variable, path, suffix):
        """
        Creates a directory for downloading raw weather data if it doesn't already exist.

        Parameters
        ----------
        variable : str
            Name of the weather variable (e.g., 'precipitation', 'temperature').
        path : str
            The base path where data should be stored.
        suffix : str, optional
            Optional suffix to differentiate between data types, by default None.

        Returns
        -------
        str
            The path of the created folder.
        """

        output_path = os.path.join(path, f'{variable}_{suffix}_raw') if suffix else os.path.join(path, f'{variable}_raw')
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        return output_path
    
    def download_weather_information(self, weather_variables:Dict, suffix_output_folder:str = None, export_as_netcdf: bool = False, ncores: int = 0, version = '2_0'):
        """
        Downloads weather data for the specified variables.

        Parameters
        ----------
        weather_variables : dict
            A dictionary of weather variables and their respective information (e.g., source, mission). Currently there are only available the following options solar_radiation, temperature_tmax, temperature_tmin, and precipitation
        suffix_output_folder : str, optional
            Suffix for the output folder to differentiate data categories, by default None.
        """
        
        downloader_config = {
            'wind_speed': {'func': self._get_wind_speed, 'params': {}},
            'vapour_pressure': {'func': self._get_vapour_pressure, 'params': {}},
            'relative_humidity': {'func': self._get_relative_humidity, 'params': {}},
            'solar_radiation': {'func': self._get_solar_radiation, 'params': {}},
            'temperature_tmax': {'func': self._get_temperature, 'params': {'statistic': 'tmax'}},
            'temperature_tmin': {'func': self._get_temperature, 'params': {'statistic': 'tmin'}},
            'precipitation': {'func': self._get_precipitation, 'params': {}},
        }
        
        for var,info in weather_variables.items():
            outputpath = self._create_dowload_folder(var, self.output_folder, suffix_output_folder)
            
            key_found = next((key for key in downloader_config if key in var), None)
            
            if key_found:
                config = downloader_config[key_found]
                meteo_var_func = config['func']
                params = {
                    'mission':info.get('mission'),
                    'urlhost':info.get('source'),
                    'output_path':outputpath,
                    'ncores': ncores,
                    'version': version,
                    **config['params']
                }
                if 'relative_humidity' in var: params.update({'time':info.get('time', None)})
                file_paths = meteo_var_func(**params)
                
            else:
                print(f"Variable '{var}' is not implemented yet.")
                file_paths = None

            if export_as_netcdf:
                years = sorted([int(y) for y in file_paths.keys()])
                self.stack_annual_data(outputpath, years[0], years[-1], outputpath)
            


    @staticmethod
    def stack_annual_data(path: str, init_year: int, end_year: int, 
                          output_path: Optional[str] = None, remove_source: bool = True):
        """Reads, stacks, and saves annual data from NetCDF files for a given year range.

        For each year, it processes individual daily/monthly files, concatenates them
        into a single multi-temporal xarray Dataset, and saves it as a single NetCDF file.

        Parameters
        ----------
        path : str
            The directory containing the source yearly zip files or folders.
        init_year : int
            The starting year.
        end_year : int
            The ending year.
        output_path : str, optional
            The directory to save the final .nc files. Defaults to `path`.
        remove_source : bool, optional
            If True, removes the original zip file and unzipped folder after processing.
        """
        assert init_year<=end_year, 'init year must be greather than ending year'
        
        if init_year != end_year:
            for year in tqdm.tqdm(range(init_year, end_year+1)):
                ClimateDataDownload.stack_annual_data(path,year,year, output_path, remove_source)
                
        else:
            try:
                mltd_dataset = read_annual_data(path, str(init_year))
                if output_path is not None:
                    mltd_dataset.to_netcdf(os.path.join(output_path, f'{init_year}.nc'))
                if remove_source:
                    if os.path.isdir(os.path.join(path, str(init_year))):
                        shutil.rmtree(os.path.join(path, str(init_year)))
                    if os.path.isfile(os.path.join(path, str(init_year) + '.zip')):
                        os.remove(os.path.join(path, str(init_year) + '.zip'))
            except (FileNotFoundError, Exception) as e:
                print(f"Could not process year {year}: {e}")
        

    def _query_config(self, product = 'datacube', mission = 'chirps', 
                        variable = 'precipitation', output_folder = None):

        if product == 'datacube':
            variable = mission + '-' + variable

            return {
                'variable': variable,
                'start_date': self._init_date,
                'end_date': self._ending_date,
                'xmin': self.aoi_extent[0],
                'xmax': self.aoi_extent[2],
                'ymin': self.aoi_extent[1],
                'ymax': self.aoi_extent[3],
                'url': self._urls['datacube'] + 'getdataArea',
                'download_path':  output_folder
            }
            
    def _download_agera5_variable(self, variable: str, statistic: Optional[List] = None,
                                  output_path: str = None, ncores: int = 1, version: str = '2_0') -> Dict:
        """Generic method to download a variable from AgEra5."""
        return download_mlt_data_from_agera5(
            variable,
            starting_date=self._init_date,
            ending_date=self._ending_date,
            aoi_extent=[self.aoi_extent[3], self.aoi_extent[0], self.aoi_extent[1], self.aoi_extent[2]],
            output_folder=output_path,
            statistic=statistic,
            ncores=ncores,
            version=version
        )
        

    def _get_wind_speed(self, mission = None, urlhost = None, output_path = None, statistic = "24_hour_mean", version = '2_0', ncores:int = 1):
        """
        function for downloading 10m_wind_speed data.

        Parameters
        ----------
        mission : str
            The mission associated with the data (e.g., 'agera5').
        urlhost : str
            The base URL for the data source.
        output_path : str
            The directory to save the downloaded data.
        version : str
            AgEra5 product's version default 2_0 other option 1_1
        """
        mission = self.missions()['wind_speed'] if mission is None else mission
        urlhost = self.download_from()['wind_speed'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path
        if mission == 'agera5' and urlhost == 'datacube':
            raise ValueError("There is no wind_speed product on data cube implemention yet")
        
        if mission == 'agera5' and urlhost == 'agera5':
            return self._download_agera5_variable(variable='10m_wind_speed', statistic = statistic if isinstance(statistic, list) else [statistic],
                                           output_path=output_path, ncores=ncores, version=version)
        else:
            return None

    def _get_vapour_pressure(self, mission = None, urlhost = None, output_path = None, statistic = "24_hour_mean", version = '2_0', ncores:int = 1):
        """
        function for downloading vapour_pressure data.

        Parameters
        ----------
        mission : str
            The mission associated with the data (e.g., 'agera5').
        urlhost : str
            The base URL for the data source.
        output_path : str
            The directory to save the downloaded data.
        version : str
            AgEra5 product's version default 2_0 other option 1_1
        """
        mission = self.missions()['vapour_pressure'] if mission is None else mission
        urlhost = self.download_from()['vapour_pressure'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path
        if mission == 'agera5' and urlhost == 'datacube':
            raise ValueError("There is no vapour_pressure product on data cube implemention yet")
        
        if mission == 'agera5' and urlhost == 'agera5':
            return self._download_agera5_variable(variable='vapour_pressure', statistic = statistic if isinstance(statistic, list) else [statistic],
                                           output_path=output_path, ncores=ncores, version=version)
        else:
            return None

    def _get_relative_humidity(self, mission = None, urlhost = None, output_path = None, ncores = 10, version = '2_0', **kwargs):
        """
        function for downloading relativity_humidity data.

        Parameters
        ----------
        mission : str
            The mission associated with the data (e.g., 'agera5').
        urlhost : str
            The base URL for the data source.
        output_path : str
            The directory to save the downloaded data.
        version : str
            AgEra5 product's version default 2_0 other option 1_1
        """
        mission = self.missions()['relative_humidity'] if mission is None else mission
        urlhost = self.download_from()['relative_humidity'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path

        if mission == 'agera5' and urlhost == 'datacube':
            request = self._query_config(product = 'datacube', mission = mission, 
            variable = 'relativehumidity', output_folder= output_path)
            print('request: {}'.format(request))
            dc_f = download_file(**request)
            return dc_f    

        if mission == 'agera5' and urlhost == 'agera5':
            return self._download_agera5_variable(variable='2m_relative_humidity', output_path=output_path, ncores=ncores, version=version)
        else:
            return None
    
    def _get_solar_radiation(self, mission = None, urlhost = None, output_path = None, ncores = 10, version = '2_0'):
        """
        Placeholder function for downloading solar radiation data.

        Parameters
        ----------
        mission : str
            The mission associated with the data (e.g., 'agera5').
        urlhost : str
            The base URL for the data source.
        output_path : str
            The directory to save the downloaded data.
        version : str
            AgEra5 product's version default 2_0 other option 1_1
        """
        mission = self.missions()['solar_radiation'] if mission is None else mission
        urlhost = self.download_from()['solar_radiation'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path

        if mission == 'agera5' and urlhost == 'datacube':
            raise ValueError("There is no solar radiation product on data cube implemention yet")

        if mission == 'agera5' and urlhost == 'agera5':
            return self._download_agera5_variable(
                'solar_radiation_flux', None, output_path, ncores, version)
        else:
            return None

    def _get_precipitation(self, mission = None, urlhost = None, output_path = None, ncores = 10, **kwargs):
        """
        Placeholder function for downloading precipitation data.

        Parameters
        ----------
        output_path : str
            The directory to save the downloaded data.
        urlhost : str
            The base URL for the data source.
        mission : str
            The mission associated with the data (e.g., 'chirps').
        """
        mission = self.missions()['precipitation'] if mission is None else mission
        urlhost = self.download_from()['precipitation'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path

        if mission == 'chirps' and urlhost == 'datacube':

            request = self._query_config(product = 'datacube', mission = 'chirps', 
            variable = 'precipitation', output_folder= output_path)
            print('request: {}'.format(request))
            dc_f = download_file(**request)
            return dc_f        
        
        if mission == 'chirps' and urlhost == 'chirps':
            chirps = CHIRPS_download()
            return chirps.download_chirps(self.aoi_extent,self._init_date,self._ending_date, output_path=output_path, ncores = ncores)


    def _get_temperature(self, mission = None, urlhost = None, output_path = None, statistic = "tmax", ncores = 10, version = '2_0'):
        """
        Placeholder function for downloading temperature data (e.g., max or min).

        Parameters
        ----------
        mission : str
            The mission associated with the data (e.g., 'agera5').
        urlhost : str
            The base URL for the data source.
        output_path : str
            The directory to save the downloaded data.
        statistic : str
            The temperature statistic to retrieve (e.g., 'tmax' for maximum temperature).
        version : str
            AgEra5 product's version default 2_0 other option 1_1
        """
        mission = self.missions()['temperature'] if mission is None else mission
        urlhost = self.download_from()['temperature'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path
        
        if mission == 'agera5' and urlhost == 'datacube':
            url = self._urls['datacube']
            request = self._query_config(product = 'datacube', mission = 'agera5', variable = 'temperature', output_folder= output_path)
            print('request: {}'.format(request))
            dc_f = download_file(*request)

        if mission == 'agera5' and urlhost == 'agera5':
            summ_statistic = {
                'tmax':["24_hour_maximum"],
                'tmin': ["24_hour_minimum"],
                'tmean': ["24_hour_mean"]
            }
            
            key_found = next((var for var in summ_statistic if statistic in var), None)
            if key_found is None: print("Check the statitisc"); return None
            
            return self._download_agera5_variable('2m_temperature', statistic=summ_statistic[key_found], 
                                                    output_path=output_path, ncores=ncores, version=version)
            
        else:
            return None

            

class MLTWeatherDataCube(DataCubeBase):
    @property
    def variables(self):
        return list(self.directory_paths.keys())

    
    @staticmethod
    def add_date_dim(xrdata, dim_value , dim_name ='date', new_dimpos = 0):
        xrdate = xrdata.expand_dims(dim = {dim_name:1}, axis = new_dimpos)
        xrdate[dim_name] = [dim_value]
        return xrdate

    def query_date(self, date):
        yeard = date[:4]
        filepaths = self._query_dates.get(date)
        return {k:os.path.join(self.directory_paths[k],yeard,v) for k, v in filepaths.items()}

    def common_dates_and_file_names(self, starting_date, ending_date):
        def filter_by_common_elements(dates, filenames):
            common_elements = set(dates[0]).intersection(*dates[1:])
            listdates = list(common_elements)
            listdates.sort()
            filenamesfiltered = []
            for j, da in enumerate(dates):
                filenamesfiltered.append([filenames[j][i] for i,v in enumerate(da) if v in listdates])

            return listdates, filenamesfiltered
        
        alldates, allfiles = [], []

        for var in self.variables:
            dates, files = self.get_date_paths(var, starting_date=starting_date, ending_date=ending_date)
            alldates.append(dates)
            allfiles.append(files)

        query_dates, query_paths = filter_by_common_elements(dates=alldates, filenames=allfiles)
        dicttmp = {}
        
        for i,d in enumerate(query_dates): 
            dicttmp[d] = {self.variables[j]:query_paths[j][i] for j in range(len(query_paths))}

        self._query_dates = dicttmp

        return dicttmp

    def get_date_paths(self, variable,starting_date, ending_date):
        """
        Find files for each variable in the date range

        Parameters
        ----------
        variable : str
            Weather variable.
        starting_date : str
            Start date in 'YYYY-MM-DD' format.
        ending_date : str
            End date in 'YYYY-MM-DD' format.

        Returns
        -------
        List[List[str]]
            A list of lists where each sublist contains the date and file path for files matching the query.
        """
        #if self.available_dates.get(variable) is None:
        out = self.folder_manager(self.directory_paths[variable], starting_date= starting_date, ending_date=ending_date)
        dates, filenames = np.array(out).T
        self.available_dates[variable], self.available_files[variable] = dates.tolist(), filenames.tolist()

        return self.available_dates[variable], self.available_files[variable]
    
    def multitemporal_data(self, reference_variable = 'precipitation', ncores = 0, **kwargs):
        xr_dict = {}
        if ncores == 0:
            for d in tqdm.tqdm(self._query_dates.keys()):
                dir_single_date_path = self.query_date(d)
                xrsingledate = self.stack_mlt_data(dir_single_date_path, reference_variable=reference_variable, **kwargs)
                #dval = datetime.strptime(d, '%Y%m%d') 
                #xrsingledate = self.add_date_dim(xrsingledate, dim_value=dval)
                xr_dict[d] = xrsingledate
        else:
            with tqdm.tqdm(total=len(list(self._query_dates.keys()))) as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
                    
                    future_to_day ={executor.submit(self.stack_mlt_data, self.query_date(d), 
                                                    reference_variable,
                                                **kwargs): (d) for d in self._query_dates.keys()}

                    for future in concurrent.futures.as_completed(future_to_day):
                        date = future_to_day[future]
                        try:
                                rs = future.result()
                                xr_dict[date] = rs
                                
                        except Exception as exc:
                                print(f"Request for year {date} generated an exception: {exc}")
                        pbar.update(1)
                            

        return xr_dict
    
    
    @staticmethod
    def mask_mldata(xr_dict,geometry, clip = True, ncores = 0):
        if ncores == 0:
            xrdict_masked = {}    
            for d, v in tqdm.tqdm(xr_dict.items()):
                xrdict_masked[d] = DataCubeBase.mask_using_geometry(v,geometry, clip = clip)
        #ncores = 10
        else:

            xrdict_masked = {}
            with tqdm.tqdm(total=len(list(xr_dict.keys()))) as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
                    
                    future_to_day ={executor.submit(DataCubeBase.mask_using_geometry, v,geometry, clip): (d) for d, v in xr_dict.items()}

                    for future in concurrent.futures.as_completed(future_to_day):
                        date = future_to_day[future]
                        try:
                                rs = future.result()
                                xrdict_masked[date] = rs
                                
                        except Exception as exc:
                                print(f"Request for year {date} generated an exception: {exc}")
                        pbar.update(1)
                            

        mlist = list(xrdict_masked.keys())
        mskedsorted = {}
        for i in np.argsort(mlist):
            mskedsorted[mlist[i]] = xrdict_masked[mlist[i]]

        return mskedsorted
    
    @staticmethod
    def to_dataframe(xr_dict):
        data= []
        for d, v in tqdm.tqdm(xr_dict.items()):
            df = v.to_dataframe()
            df['date'] = datetime.strptime(d, '%Y%m%d') 
            data.append(df)
            
        return pd.concat(data)        

    def __init__(self, directory_paths: Dict, folder_manager, extent=None) -> None:
        self.available_dates = {}
        self.available_files = {}
        self.directory_paths = directory_paths
        self.folder_manager = folder_manager
        super().__init__( extent)

