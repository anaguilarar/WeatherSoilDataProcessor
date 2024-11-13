import os
import cdsapi
import copy
from typing import List, Optional, Dict
import concurrent.futures
import pandas as pd
from shapely.geometry import Polygon

from .gis_functions import (from_polygon_2bbox, from_xyxy_2polygon, 
                            list_tif_2xarray, read_raster_data)

from .files_manager import (days_range_asstring, months_range_asstring, 
                            split_date, create_yearly_query, uncompress_zip_path, find_date_instring)

from datetime import datetime
from .utils import download_file
from .datacube import DataCubeBase

from rasterio.mask import mask
import rasterio
import geopandas as gpd

import numpy as np

import tqdm
import xarray

import shutil

def tansform_dates_for_AgEraquery(
        year: int, 
        init_day: Optional[int] = None, 
        end_day: Optional[int] = None, 
        init_month: Optional[int] = None, 
        end_month: Optional[int] = None
        ) -> Dict[str, List[str]]:
    """
    Transform input year, day, and month into a formatted query for AgEra5 data download.

    Parameters
    ----------
    year : int
        Year of the query.
    init_day : int, optional
        Starting day of the month. If None, defaults to 0 (beginning of month).
    end_day : int, optional
        Ending day of the month. If None, defaults to 31 (end of the month).
    init_month : int, optional
        Starting month of the year (1-12). If None, defaults to 0 (beginning of the year).
    end_month : int, optional
        Ending month of the year (1-12). If None, defaults to 12 (end of the year).

    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing year, month, and day strings formatted for the query.

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


def donwload_mlt_data_from_agera5(
    variable: str, 
    starting_date: str, 
    ending_date: str, 
    output_folder: str, 
    aoi_extent: List[float], 
    product: str= "sis-agrometeorological-indicators", 
    statistic: Optional[str] = None,
    ncores: int = 10
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

    Returns
    -------
    None
        Downloads data and saves it in the specified output folder.

    Example
    -------
    >>> download_mlt_data_from_agera5('temperature', '2023-01-01', '2023-12-31', './data', [-10, 35, 10, 50], 'sis-agrometeorological-indicators')
    """
    def download_one_year_data(year, query_dict, init_day, end_day, init_month, end_month, init_year, end_year):

        if year == init_year:
            datesquery = tansform_dates_for_AgEraquery(year, init_day = init_day, end_day = 31, init_month = init_month, end_month = 12)
        if year == end_year:
            datesquery = tansform_dates_for_AgEraquery(year, init_day = 1, end_day = end_day, init_month = 1, end_month = end_month)
        else:
            datesquery = tansform_dates_for_AgEraquery(year)

        year_query = copy.deepcopy(query_dict)
        year_query.update(datesquery) 
        print(year, year_query)
        filename = os.path.join(output_folder, "{}.zip".format(year))
        client = cdsapi.Client()
        client.retrieve(product, year_query).download(target = filename)
        return filename
    

    init_year, init_month, init_day = split_date(starting_date)
    end_year, end_month, end_day = split_date(ending_date)


    years = list(range(init_year,end_year+1))
    query_dict = {"version": "1_1",
                    "area":aoi_extent,
                    "variable": variable if isinstance(variable, list) else [variable],
                    "statistic": [""] if statistic is None else statistic}
    
    ## create queryies per month
    #if len(years)<10:
    #    for year in years:
    #        download_one_year_data(year, query_dict, init_day, end_day, init_month, end_month, years[0], years[-1])
    #else:
    file_path_peryear = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
        future_to_year ={executor.submit(download_one_year_data, year, query_dict, init_day, end_day, init_month, end_month, years[0], years[-1]): (year) for year in years}
    
        for future in concurrent.futures.as_completed(future_to_year):
            year = future_to_year[future]
            try:
                    file_path = future.result()
                    file_path_peryear[str(year)] = file_path
                    print(f"Requested Year {year} \n downloaded in {file_path}")
            except Exception as exc:
                    print(f"Request for year {year} generated an exception: {exc}")

    return file_path_peryear
    



class CHIRPS_download:

    def set_url(self, year, date):

        return "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_{}/cogs/p{}/{}/chirps-v2.0.{}.cog".format(
            self._frequency,
            self.resolution,
            year,
            date
        )


    def __init__(self, frequency = 'daily', sp_resolution = '05') -> None:
        self._frequency = frequency
        self.resolution = sp_resolution


    def current_date(self):
        self._current_date + 1

    def set_area(self, xyxy):
        self.extent = gpd.GeoSeries([from_xyxy_2polygon(*xyxy)])

    @staticmethod
    def set_days(init_day, end_day):
        return days_range_asstring(init_day, end_day)
    
        
    def _create_yearly_query(self):
        return create_yearly_query(self._initdate, self._enddate)
    
    def download_data_per_year(self, year, output_path, extent):
        if not os.path.exists(os.path.join(output_path,year)): os.mkdir(os.path.join(output_path,year)) 
        
        for month in self._date[year].keys():
            stackimages = []
            for day in self._date[year][month]:
                urlpath = self.set_url(year, '{}.{}.{}'.format(year, month,day))
                print(urlpath)
                with rasterio.open(urlpath) as src:
                    meta = src.profile
                    masked, mask_transform = mask(dataset=src, shapes=gpd.GeoSeries([from_xyxy_2polygon(*extent)]), crop=True)
                    stackimages.append(masked)
                xrm = list_tif_2xarray(masked, mask_transform, crs=str(meta['crs']), bands_names=['precipitation'], dimsformat= 'CHW')
                xrm.to_netcdf(os.path.join(output_path,year, 'chirps_precipitation_{}{}{}.nc'.format(year,month,day)))

        return os.path.join(output_path,year)
        
    
    def download_chirps(self, extent , init_date, ending_date, output_path = None, ncores = 10):
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
def process_file(year_path_folder, filename, date, xdim_name, ydim_name, depthdim_name):
    """
    Helper function to read and process a single NetCDF file.
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
    """
    A class to handle downloading weather data (precipitation, temperature, solar radiation, etc.)
    from different sources like CHIRPS, AgEra5, and custom data cubes.
    """

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

        return {   'precipitation': 'datacube',
            'temperature': 'datacube',
            'solar_radiation': 'agera5',
            'relative_humidity': 'datacube'
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
            'solar_radiation': 'agera5',
            'relative_humidity': 'agera5'
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
    
    def download_weather_information(self, weather_variables:Dict, suffix_output_folder:str = None, export_as_netcdf: bool = False, ncores: int = 0):
        """
        Downloads weather data for the specified variables.

        Parameters
        ----------
        weather_variables : dict
            A dictionary of weather variables and their respective information (e.g., source, mission). Currently there are only available the following options solar_radiation, temperature_tmax, temperature_tmin, and precipitation
        suffix_output_folder : str, optional
            Suffix for the output folder to differentiate data categories, by default None.
        """
        for var,info in weather_variables.items():
            
            outputpath = self._create_dowload_folder(var, self.output_folder, suffix_output_folder)

            if 'relative_humidity' in var:
                file_paths = self._get_relative_humidity(mission=info['mission'], 
                                            urlhost=info['source'],
                                            output_path=outputpath)
                
            elif 'solar_radiation' in var:
                file_paths = self._get_solar_radiation(mission=info['mission'], 
                                            urlhost=info['source'],
                                            output_path=outputpath)
            
            elif 'temperature_tmax' in var:
                file_paths = self._get_temperature(mission=info['mission'], 
                                            urlhost=info['source'],
                                        output_path=outputpath,
                                        statistic= 'tmax')
            
            elif 'temperature_tmin' in var:
                file_paths = self._get_temperature(mission=info['mission'], 
                                            urlhost=info['source'],
                                        output_path=outputpath,
                                        statistic= 'tmin')
                
            elif 'precipitation' in var:
                file_paths = self._get_precipitation(output_path=outputpath, urlhost='chirps', mission='chirps', ncores = ncores)

            else:
                print(f"{var} is Not implemented yet!!")

            if export_as_netcdf:
                yearlist = list(file_paths.keys())
                self.stack_annual_data(outputpath, yearlist[0], yearlist[-1], outputpath)
            


    @staticmethod
    def stack_annual_data(path, init_year, end_year, output_path: str = None, removefolder = True):
        assert init_year<=end_year, 'init year must be greather than ending year'
        
        if init_year != end_year:
            for year in tqdm.tqdm(range(init_year, end_year+1)):
                ClimateDataDownload.stack_annual_data(path,year,year, output_path, removefolder)
                
        else:
            mltd_dataset = read_annual_data(path, str(init_year))
            if output_path is not None:
                mltd_dataset.to_netcdf(os.path.join(output_path, f'{init_year}.nc'))
            if removefolder:
                if os.path.isdir(os.path.join(path, str(init_year))):
                    shutil.rmtree(os.path.join(path, str(init_year)))
                if os.path.isfile(os.path.join(path, str(init_year) + '.zip')):
                    os.remove(os.path.join(path, str(init_year) + '.zip'))



    def __init__(self, starting_date = None, ending_date = None, aoi: Polygon = None, xyxy: List[float] = None, output_folder:str = None) -> None:

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

        ## TODO assert extent
        

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

    def _get_relative_humidity(self, mission = None, urlhost = None, output_path = None, ncores = 10):
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
        """
        mission = self.missions()['relativity_humidity'] if mission is None else mission
        urlhost = self.download_from()['relativity_humidity'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path

        if mission == 'agera5' and urlhost == 'datacube':
            raise ValueError("not implemented yet") ##TODO

        if mission == 'agera5' and urlhost == 'agera5':
            return donwload_mlt_data_from_agera5('2m_relative_humidity', 
                                        starting_date= self._init_date,
                                        ending_date=self._ending_date, 
                                        aoi_extent= [self.aoi_extent[3],self.aoi_extent[0],self.aoi_extent[1],self.aoi_extent[2]], 
                                        output_folder= output_path,
                                        statistic= [""], ncores = ncores)
    
    def _get_solar_radiation(self, mission = None, urlhost = None, output_path = None, ncores = 10):
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
        """
        mission = self.missions()['solar_radiation'] if mission is None else mission
        urlhost = self.download_from()['solar_radiation'] if urlhost is None else urlhost
        output_path = self.output_folder if output_path is None else output_path

        if mission == 'agera5' and urlhost == 'datacube':
            raise ValueError("There is no solar radiation product on data cube implemention yet")

        if mission == 'agera5' and urlhost == 'agera5':
            return donwload_mlt_data_from_agera5('solar_radiation_flux', 
                                        starting_date= self._init_date,
                                        ending_date=self._ending_date, 
                                        aoi_extent= [self.aoi_extent[3],self.aoi_extent[0],self.aoi_extent[1],self.aoi_extent[2]], 
                                        output_folder= output_path,
                                        statistic= [""], ncores = ncores)

    def _get_precipitation(self, mission = None, urlhost = None, output_path = None, ncores = 10):
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


    def _get_temperature(self, mission = None, urlhost = None, output_path = None, statistic = "tmax", ncores = 10):
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
            if statistic == 'tmax':
                strstatistic = ["24_hour_maximum"]
            elif statistic == 'tmin':
                strstatistic = ["24_hour_minimum"]
            elif statistic == 'tmean':
                strstatistic = ["24_hour_mean"]
            else:
                raise ValueError("It is not included")
            return donwload_mlt_data_from_agera5('2m_temperature', 
                                        starting_date= self._init_date,
                                        ending_date=self._ending_date, 
                                        aoi_extent= [self.aoi_extent[3],self.aoi_extent[0],self.aoi_extent[1],self.aoi_extent[2]], 
                                        output_folder= output_path,
                                        statistic=strstatistic, ncores = ncores)
            

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
        


        
