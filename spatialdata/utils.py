from datetime import datetime, timedelta
import copy

import os
import requests
import json
import pandas as pd
import tqdm

import zipfile
import numpy as np

import xarray
from .gis_functions import list_tif_2xarray,resample_xarray,reproject_xrdata


def download_file(start_date:str,
                end_date:str,
                xmin:float,
                xmax:float,
                ymin:float,
                ymax:float,
                url:str,
                variable:str,
                download_path:str
                ):

    """
    Downloads a file from a specified URL using a POST request.

    Parameters
    ----------
    start_date : str
        The start date for the data download, in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data download, in 'YYYY-MM-DD' format.
    xmin : float
        The minimum longitude for the bounding box.
    xmax : float
        The maximum longitude for the bounding box.
    ymin : float
        The minimum latitude for the bounding box.
    ymax : float
        The maximum latitude for the bounding box.
    url : str
        The URL to download the file from.
    variable : str
        The name of the variable to download.
    download_path : str
        The path to the directory where the file will be saved.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    
    headers = {'Accept': 'application/json'}
    
    url_params = {"startDt":start_date,
                "endDt":end_date,
                "xmin":xmin,
                "xmax":xmax,
                "ymin":ymin,
                "ymax":ymax,
                "variableName":variable
    }
    
    response = requests.post(url,
                            headers = headers,
                            data=json.dumps(url_params),
                            stream=True)
    if response.status_code == 200:
        file_name = f"{variable}_{start_date}_to_{end_date}.nc"
        file_path = os.path.join(download_path, file_name)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return file_path
    else:
        response.raise_for_status()

def check_depth_name_dims(xrdata):
    """
    Selects the first slice of a multi-dimensional xarray Dataset.

    This function is used to reduce a Dataset to a 2D array by selecting the
    first slice along the time or band dimension.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input Dataset.

    Returns
    -------
    xarray.Dataset
        The 2D Dataset.
    """
    
    if 'date' in list(xrdata.sizes.keys()):
        xrdata = xrdata.isel({'date': 0})
    elif 'time' in list(xrdata.sizes.keys()):
        xrdata = xrdata.isel({'time': 0})
    elif 'band' in list(xrdata.sizes.keys()):
        xrdata = xrdata.isel({list(xrdata.sizes.keys())[0]: 0})
    else:
        raise ValueError('check depth order')
    return xrdata

def set_xr_attributes(xrdata, xdimref_name = 'x', ydimref_name = 'y'):
    if 'transform' not in list(xrdata.attrs.keys()):
        xrdata.attrs['transform'] = xrdata.rio.transform()
    if 'crs' not in list(xrdata.attrs.keys()):
        xrdata.attrs['crs'] = xrdata.rio.crs
    if 'width' not in list(xrdata.attrs.keys()):
        xrdata.attrs['width'] = len(xrdata[xdimref_name].values)
    if 'height' not in list(xrdata.attrs.keys()):
        xrdata.attrs['height'] = len(xrdata[ydimref_name].values)

    return xrdata

def stack_xrdata_variable(xrdata, xrrefence, xrefdim_name, yrefdim_name,method, target_crs, only_use_first_date =True):

        if len(xrdata.sizes.keys()) >= 3 and only_use_first_date:
            xrdata = check_depth_name_dims(xrdata)
    
        resampled_data = resample_xarray(xrdata, xrrefence, xrefdim_name=xrefdim_name, yrefdim_name=yrefdim_name, method = method, target_crs = target_crs)
        variable_name = list(resampled_data.data_vars.keys())[0]
        resampled_data = resampled_data[variable_name].values
        return resampled_data
    

def resample_variables(dict_xr,reference_variable = None, only_use_first_date = True, 
                       verbose = False, method: str = 'linear', target_crs = None):

    
    listvariables = list(dict_xr.keys())
    if reference_variable is None:
        reference_variable = listvariables[0]
    listvariables.remove(reference_variable)

    xr_reference = dict_xr[reference_variable].copy()

    target_crs = target_crs if target_crs is not None else xr_reference.rio.crs
    if len(xr_reference.sizes.keys()) >= 3 and only_use_first_date:
        xr_reference = check_depth_name_dims(xr_reference)
    if str(target_crs) != str(xr_reference.rio.crs):
        xr_reference = reproject_xrdata(xr_reference, target_crs)

    if 'x' in list(xr_reference.sizes.keys()):
        xdimref_name , ydimref_name = 'x', 'y'
    elif 'lon' in list(xr_reference.sizes.keys()):
        xdimref_name , ydimref_name = 'lon', 'lat'
    else:
        xdimref_name = xdimref_name.isel({list(xdimref_name.sizes.keys())[0]: 0})
    
    xr_reference = set_xr_attributes(xr_reference, xdimref_name = xdimref_name, ydimref_name = ydimref_name)
    metadata = xr_reference.attrs

    variable_name = list(xr_reference.data_vars.keys())[0]
    resampled_list = [xr_reference[variable_name].values]
    for var, xr_data in dict_xr.items():
        if var == reference_variable:
            continue
        resampled_data = stack_xrdata_variable(xr_data, xr_reference, xdimref_name, ydimref_name, method, target_crs, only_use_first_date=only_use_first_date)
        resampled_list.append(resampled_data)
        if verbose: print('{} resampled ..'.format(var))

    
    return list_tif_2xarray(resampled_list, metadata['transform'], 
                    crs = metadata['crs'], nodata=-9999, 
                    bands_names = [reference_variable]+listvariables,
                    dimsformat = 'CHW',
                    dimsvalues = {'x': xr_reference[xdimref_name].values, 
                                'y': xr_reference[ydimref_name].values})
    

def compress_file(input_filepath: str, output_zip_filepath: str) -> None:
    """
    Compresses a single file into a new ZIP archive.

    Parameters
    ----------
    input_filepath : str
        The path to the file to compress.
    output_zip_filepath : str
        The path to the output ZIP archive.
    """
    
    with zipfile.ZipFile(output_zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(input_filepath, os.path.basename(input_filepath))
    #print(f"File '{input_filepath}' compressed to '{output_zip_filepath}'")


def set_encoding(xrdata: xarray.Dataset, compress_method: str = 'zlib') -> dict:
    """
    Creates an encoding dictionary for compressing an xarray Dataset.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input Dataset.
    compress_method : str, optional
        The compression method to use, by default 'zlib'

    Returns
    -------
    dict
        The encoding dictionary.
    """
    return {k: {compress_method: True} for k in list(xrdata.data_vars.keys())}


def compress_xarray_dataset(xrdata: xarray.Dataset, ouput_filename: str, to_zipfile: bool = True, engine: str = 'netcdf4',scale_value: int = 100, nodata: int = -9999) -> xarray.Dataset:
    """
    Compresses an xarray Dataset and saves it to a NetCDF file.

    This function uses masked arrays to handle NoData values, which is a more
    robust method than replacing them with a placeholder value.

    Parameters
    ----------
    xrdata : xarray.Dataset
        The input Dataset.
    ouput_filename : str
        The path to the output NetCDF file.
    to_zipfile : bool, optional
        Whether to compress the NetCDF file into a ZIP archive, by default True
    engine : str, optional
        The NetCDF engine to use, by default 'netcdf4'
    scale_value : int, optional
        The scale factor to apply to the data before saving, by default 100
    nodata : int, optional
        The NoData value to use, by default -9999

    Returns
    -------
    xarray.Dataset
        The compressed Dataset.
    """
    data = copy.deepcopy(xrdata)
    data = data * scale_value

    # Use a masked array to handle NoData values
    for k in data.data_vars.keys():
        data[k] = data[k].where(~np.isnan(data[k]), nodata)
    data = data.astype(np.int32)
    data.attrs['nodata'] = nodata
    data.attrs['dtype'] = 'int'
    data.attrs['width'] = data.x.values.shape[0]
    data.attrs['height'] = data.y.values.shape[0]
    data.attrs['count'] = len(list(data.data_vars.keys()))
    data.attrs['scale_values_factor'] = scale_value
    
    encoding = set_encoding(data)
    data.to_netcdf(ouput_filename, encoding = encoding, engine = engine)
    if to_zipfile:
        compress_file(ouput_filename, ouput_filename.replace('.nc','.zip'))
        data = None
        os.remove(ouput_filename)
    return data

def read_compressed_xarray(input_filepath: str, output_path:str = None, engine: str = 'netcdf4', update_file = True) -> xarray.Dataset:
    """
    Reads a compressed xarray Dataset from a NetCDF or ZIP file.

    Parameters
    ----------
    input_filepath : str
        The path to the input file.
    output_path : str, optional
        The path to extract the file to if it is a ZIP archive, by default None
    engine : str, optional
        The NetCDF engine to use, by default 'netcdf4'

    Returns
    -------
    xarray.Dataset
        The loaded Dataset.
    """
    input_path = os.path.dirname(input_filepath)
    input_filepath = os.path.basename(input_filepath)
    output_path = output_path if output_path else input_path
    
    if input_filepath.endswith('zip'):
        with zipfile.ZipFile(os.path.join(input_path,input_filepath), 'r') as zip_object:
            zip_object.extractall(output_path)        
        input_filepath = input_filepath.replace('.zip', '.nc')

    output_filepath = os.path.join(output_path, input_filepath)
    
    with xarray.open_dataset(output_filepath, engine = engine) as ds:
        data = ds.copy()
    
    del ds    
    
    data = data.astype(float)
    scale_values_factor = data.attrs['scale_values_factor']

    if data.attrs.get('dtype', None):
        data.attrs['dtype'] = 'float'
    
    data = data/scale_values_factor
    nodata = data.attrs.get('nodata', None)
    if nodata:
        data.attrs['nodata'] = nodata/scale_values_factor
        for k in data.data_vars.keys():
            data[k] = data[k].where(data[k] != nodata/scale_values_factor, np.nan)
    if update_file:
        encoding = set_encoding(data)
        data.to_netcdf(output_filepath, encoding = encoding, engine = engine)
        
    del data


