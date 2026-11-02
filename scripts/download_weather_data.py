from omegaconf import OmegaConf
from spatialdata.climate_data import ClimateDataDownload
from spatialdata.gis_functions import get_boundaries_from_path
import logging
import argparse
from spatialdata.files_manager import IntervalFolderManager
from spatialdata.climate_data import MLTWeatherDataCube

from pathlib import Path
import os
from datetime import datetime
import xarray

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Getting weather data')
    parser.add_argument('--config', 
                        default='weather_data_downloading_config.yaml', help='config file path')

    args = parser.parse_args()
    return args


def set_encoding(xrdata, compress_method="zlib"):
    encoding = {}
    for k in list(xrdata.data_vars.keys()):
        encoding[k] = {compress_method: True}
        # Explicitly preserve spatial reference in encoding if present
        if 'grid_mapping' in xrdata[k].attrs:
            encoding[k]['grid_mapping'] = xrdata[k].attrs['grid_mapping']
            # Remove it from attributes to prevent xarray raising an error
            del xrdata[k].attrs['grid_mapping']
            
    # Keep the spatial_ref encoding intact if it's explicitly stored
    if 'spatial_ref' in xrdata.variables:
        encoding['spatial_ref'] = {}
        
    return encoding


def stack_datacube_temporally(xrdata_dict, time_dim_name='date', parse_dates=True):
    """
    Stacks a dictionary of xarray Datasets along a new temporal dimension,
    preserving spatial resolution and CRS securely.
    """
    datasets = []

    if parse_dates:
        time_coords = [datetime.strptime(k, "%Y%m%d") for k in xrdata_dict.keys()]
    else:
        time_coords = list(xrdata_dict.keys())
        
    for time_coord, dataset in zip(time_coords, xrdata_dict.values()):

        ds_expanded = dataset.assign_coords({time_dim_name: time_coord}).expand_dims(time_dim_name)
        datasets.append(ds_expanded)

    stacked_cube = xarray.concat(datasets, dim=time_dim_name, combine_attrs="override")
    

    first_ds = datasets[0]
    if first_ds.rio.crs is not None:
        stacked_cube.rio.write_crs(first_ds.rio.crs, inplace=True)
    if first_ds.rio.transform() is not None:
        stacked_cube.rio.write_transform(first_ds.rio.transform(), inplace=True)

    if 'spatial_ref' in stacked_cube.variables:
        for var in stacked_cube.data_vars:
            stacked_cube[var].attrs['grid_mapping'] = 'spatial_ref'
            
    # Overwrite the global attributes dict with the source ones
    stacked_cube.attrs.update(first_ds.attrs)

    return stacked_cube

def main():
    logging.info("Starting to download data")
    args = parse_args()

    # reading configuration
    config = OmegaConf.load(args.config)

    logging.info(f"Reading configuration from {args.config}")
    if config.SPATIAL_INFO.get('spatial_file',None):
        extent = get_boundaries_from_path(config.SPATIAL_INFO.get('spatial_file',None), round_numbers = True)
    else:
        extent = config.SPATIAL_INFO.extent

    if config.GENERAL.task == 'download':
        # Initialize ClimateDataDownload object with config parameters
        climatedata = ClimateDataDownload(starting_date= config.DATES.starting_date,
                                    ending_date= config.DATES.ending_date, 
                                    xyxy= extent, 
                                    output_folder= config.PATHS.output_path)
        
        
        for var, info in config.WEATHER.variables.items():
            logging.info(f"-----> Download {var} from {info}")
            climatedata.download_weather_information({var:info}, 
                                                        suffix_output_folder=config.GENERAL.suffix, 
                                                        export_as_netcdf=config.GENERAL.export_as_netcdf,
                                                        ncores = config.GENERAL.ncores)
                
        logging.info("Data download completed!")

    if config.GENERAL.task == 'datacube':

        meteo_names = {
            'precipitation': 'precipitation',
            'solar_radiation': 'srad',
            'temperature_tmax': 'tmax',
            'temperature_tmin': 'tmin',
            'vapour_pressure': 'vpd',
            'wind_speed': 'ws',
            'reference_evapotranspiration': 'etr'
        }


        list_weather_paths = {}
        for k, v in config.WEATHER.variables.items():
            list_weather_paths[meteo_names[k]] = os.path.join(config.PATHS.output_path, k + "_" + config.GENERAL.suffix + '_raw')

        logging.info(f"Data cube creation started! {list_weather_paths}")
        folder_manager = IntervalFolderManager()

        wdatacube = MLTWeatherDataCube(list_weather_paths, folder_manager)

        filnames = wdatacube.common_dates_and_file_names(starting_date=config.DATES.starting_date, ending_date=config.DATES.ending_date)
        mltdata = wdatacube.multitemporal_data(reference_variable=config.GENERAL.reference_variable)
        climate_path = os.path.join(config.PATHS.output_path, f'weather_{config.GENERAL.suffix}_{config.DATES.starting_date[:4]}_{config.DATES.ending_date[:4]}.nc')

        dcengine = 'netcdf4'

        weather_datac = stack_datacube_temporally(mltdata, time_dim_name='date', parse_dates=True)

        encoding = set_encoding(weather_datac)
        weather_datac.to_netcdf(climate_path, encoding = encoding, engine = dcengine)

        logging.info("Data preprocessing completed!")
        
if __name__ == "__main__":
    main()