
import argparse
import copy
import geopandas as gpd
import logging
import numpy as np
import os
import xarray

from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm

from spatialdata.climate_data import MLTWeatherDataCube
from spatialdata.files_manager import IntervalFolderManager, SoilFolderManager
from spatialdata.gis_functions import get_boundaries_from_path, reproject_xrdata, re_scale_xarray, resample_xarray, add_2dlayer_toxarrayr
from spatialdata.soil_data import (TEXTURE_CLASSES,SoilDataCube, find_soil_textural_class_in_nparray)

try:
    from crop_modeling.utils import from_weather_to_dssat, check_weatherxr_scales
    from crop_modeling.utils import from_soil_to_dssat
except:
    from crop_modeling.utils import from_weather_to_dssat, check_weatherxr_scales
    from crop_modeling.utils import from_soil_to_dssat

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
    extent = get_boundaries_from_path(config.SPATIAL_INFO.boundaries, round_numbers = True)

    # Specify paths for weather data, such as precipitation and solar radiation
    list_weather_paths = {'precipitation': config.WEATHER.paths.precipitation,
                        'srad': config.WEATHER.paths.srad,
                        'tmax': config.WEATHER.paths.tmax,
                        'tmin': config.WEATHER.paths.tmin}


    wdatacube = MLTWeatherDataCube(list_weather_paths, IntervalFolderManager(), extent=extent)

    wdatacube.common_dates_and_file_names(starting_date=config.WEATHER.starting_date, 
                                                      ending_date=config.WEATHER.ending_date)
    
    return wdatacube.multitemporal_data(reference_variable=config.WEATHER.reference_variable, 
                                        ncores=config.GENERAL.ncores)

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
    gfd = gpd.read_file(config.SPATIAL_INFO.boundaries)
    gfd = gfd.to_crs(config.SOIL.crs_reference)
    folder_manager = SoilFolderManager(config.SOIL.path, config.SOIL.variables)
    soilcube = SoilDataCube(folder_manager)
    return soilcube.multi_depth_data(verbose=False, reference_variable='sand')


def parse_args():
    """
    Parses command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the config file path.
    """
    parser = argparse.ArgumentParser(description='Generating DSSAT files')
    parser.add_argument('--config', 
                        default='', help='config file path')

    args = parser.parse_args()
    return args

def main():
    """
    Main function to process weather and soil data cubes and convert them into DSSAT files.
    """

    logging.info("DSSAT files processing")
    args = parse_args()

    # reading configuration
    config = OmegaConf.load(args.config)

    logging.info(f"Reading configuration from {args.config}")
    weather_datacube = get_weather_datacube(config)
    logging.info("Weather data cube created from {} to {}".format(config.WEATHER.starting_date,config.WEATHER.ending_date))
    soil_datacube = get_soil_datacube(config)
    logging.info("Soil data cube created with the following variables: {}".format(config.SOIL.variables))


    crs = config.GENERAL.crs_reference
    #weather_datacube = {k: reproject_xrdata(v,target_crs=crs) for k, v in weather_datacube.items()}
    soil_datacube = {k: reproject_xrdata(v,target_crs=crs) for k, v in soil_datacube.items()}
    
    gdf = gpd.read_file(config.ROI.path)
    rois = gdf[config.ROI.roi_column].values

    for roi_name in rois:
        logging.info("----- Processing {}".format(roi_name))
        subset = gdf.loc[gdf[config.ROI.roi_column] == roi_name]
        logging.info("  Masking")
        print("==> soil")
        soil_datacube_m = SoilDataCube.mask_mldata(soil_datacube,subset.geometry)
        print("==> weather")
        weather_datacube_m = MLTWeatherDataCube.mask_mldata(weather_datacube,subset.geometry, ncores=config.GENERAL.ncores)


        ## 
        scale_factor = config.WEATHER.scale_factor ## factor for dowsampling
        logging.info("  Rescaling")
        # weaather rescale
        weather_datacube_r = {}
        for k,v in tqdm(weather_datacube_m.items()):
            weather_datacube_r[k] = re_scale_xarray(v, scale_factor= scale_factor)

        # soil rescale
        xr_reference = weather_datacube_r[list(weather_datacube_r.keys())[0]]

        soil_datacube_r = {}
        for k,v in tqdm(soil_datacube_m.items()):
            soil_datacube_r[k] = resample_xarray(v, xrreference=xr_reference)

        logging.info("  Grouping")
        soil_datacube_rmrt = {}

        for k,v in soil_datacube_r.items():
            sand = v.sand.values*0.1 if np.nanmax(v.sand.values) > 300 else v.sand.values
            clay = v.clay.values*0.1 if np.nanmax(v.clay.values) > 300 else v.clay.values
            texturemap = find_soil_textural_class_in_nparray(sand, clay).astype(float)
            texturemap[texturemap == 0] = np.nan
            soil_datacube_rmrt[k] = add_2dlayer_toxarrayr(texturemap, v, variable_name=config.GROUPBY.variable)


        soilref = soil_datacube_rmrt[config.SOIL.depth_refernce]
        # getting datavars
        weatherdatavars = list(weather_datacube_r[list(weather_datacube_r.keys())[0]].data_vars.keys())

        weather_datacube_mrs = []
        for k,v in tqdm(weather_datacube_r.items()):
            xrtemp = xarray.merge([v,soilref])[weatherdatavars+ [config.GROUPBY.variable]]
            xrtemp = xrtemp.expand_dims(dim = ['date'])
            xrtemp['date'] = [k]
            weather_datacube_mrs.append(xrtemp)
        weather_datacube_mrs = xarray.concat(weather_datacube_mrs, dim = 'date')

        weather_datacube_mrs['date'] = [datetime.strptime(i, "%Y%m%d") for i in list(weather_datacube_r.keys())]

        

        output = os.path.join(config.PATHS.output_path, roi_name )
        if not os.path.exists(output):
            os.mkdir(output)
        
        ### DSSAT FILES
                
        soil_datacube_mrs = []
        for ks, vs in soil_datacube_rmrt.items():
            xrtemp = vs.expand_dims(dim = ['depth'])
            xrtemp['depth'] = [ks]
            soil_datacube_mrs.append(xrtemp)

        soil_datacube_mrs = xarray.concat(soil_datacube_mrs, dim = 'depth')
        logging.info(f"  creating DSSAT soil file in {output}")

        soil_df = from_soil_to_dssat(soil_datacube_mrs, groupby=config.GROUPBY.variable, outputpath=output, outputfn='SOIL'+roi_name, codes=TEXTURE_CLASSES, country = config.GENERAL.country.upper(),site = roi_name)
        soil_df.to_csv(os.path.join(output,f'SOIL{roi_name}.csv'))
        soil_datacube_mrs.rio.to_file(os.path.join(output,f'SOIL{roi_name}.nc'))

        logging.info(f"  creating DSSAT weather file in {output}")
        from_weather_to_dssat(copy.deepcopy(weather_datacube_mrs), date_name = 'date', 
                            groupby = config.GROUPBY.variable, 
                            params_df_names=config.DSSAT.variable_names,
                            outputpath=output, outputfn = 'WHTE'+roi_name, codes=TEXTURE_CLASSES, ncores=config.GENERAL.ncores)
    
        vars_metric = {}

        for i in weather_datacube_mrs.data_vars.keys():
            vars_metric.update({i: 'mean'})
        vars_metric.pop(config.GROUPBY.variable)
        df = check_weatherxr_scales(weather_datacube_mrs).to_dataframe().dropna().reset_index().groupby([config.GROUPBY.variable,'date']).agg(vars_metric).reset_index()

        df.to_csv(os.path.join(output,f'WHTE{roi_name}.csv'))


if __name__ == "__main__":
    main()