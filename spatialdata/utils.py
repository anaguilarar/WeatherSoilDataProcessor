
from datetime import datetime, timedelta
import argparse
import concurrent.futures
import os
import requests
import json
import pandas as pd

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

        if len(xr_data.sizes.keys()) >= 3 and only_use_first_date:
            xr_data = check_depth_name_dims(xr_data)
        
        resampled_data = resample_xarray(xr_data, xr_reference, xrefdim_name=xdimref_name, yrefdim_name=ydimref_name, method = method, target_crs = target_crs)
        variable_name = list(resampled_data.data_vars.keys())[0]
        resampled_data = resampled_data[variable_name].values

        if verbose: print('{} resampled ..'.format(var))

        resampled_list.append(resampled_data)


    return list_tif_2xarray(resampled_list, metadata['transform'], 
                    crs = metadata['crs'], nodata=-9999, 
                    bands_names = [reference_variable]+listvariables,
                    dimsformat = 'CHW',
                    dimsvalues = {'x': xr_reference[xdimref_name].values, 
                                'y': xr_reference[ydimref_name].values})

