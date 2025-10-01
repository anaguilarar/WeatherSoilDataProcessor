import rioxarray as rio

from crop_modeling.spatial_process import SpatialCM
from crop_modeling.dssat.output import update_dssat_data_using_path
from crop_modeling.spatial_process import create_mlt_yield_raster
from crop_modeling.utils.output_transforms import summarize_spatial_yields_by_time_window
from crop_modeling.utils.process import get_crs_fromxarray,set_encoding, check_crs_inxrdataset
    
from omegaconf import OmegaConf
import os
import requests

import numpy as np
import pandas as pd
import geopandas as gpd

import logging
import argparse


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Run crop simulation')
    parser.add_argument('--config', 
                        default='options.yaml', help='config file path')

    args = parser.parse_args()
    return args

def export_data(xrdata, fn):


    dcengine = 'netcdf4'
    encoding = set_encoding(xrdata)
    xrdata = check_crs_inxrdataset(xrdata)
    xrdata.to_netcdf(fn, encoding = encoding, engine = dcengine)


def simulate_roi(spatial_cm, feature_code):
    logging.info(f"-----> Running simulation in {feature_code}")
    feat_attr = spatial_cm.config.SPATIAL_INFO.get('feature_name', None)
    assert feat_attr, 'check feature column name'
    roi = spatial_cm.geo_features.loc[spatial_cm.geo_features[feat_attr]==str(feature_code)]
    roi_name = roi[spatial_cm.config.SPATIAL_INFO.feature_name].values[0]
        
    spatial_cm.set_up_folders(site = roi_name)

    # Create soil and weather files for the selected region
    workingpath = spatial_cm.create_roi_sp_data(
        roi= roi,
        export_spatial_data= True
    )
    if workingpath is not None:
        # Locate environmental working paths
        spatial_cm.model.find_envworking_paths(spatial_cm._tmp_path, 'WTH')

        # Set up crop files
        spatial_cm.model.set_up_crop(crop=spatial_cm.crop, cultivar=spatial_cm.cultivar)

        # Set up management files
        spatial_cm.model.set_up_management(crop=spatial_cm.crop, cultivar=spatial_cm.cultivar, **spatial_cm.config.MANAGEMENT)

        # run the simulation
        dssath_path = spatial_cm.config.GENERAL_INFO.get('dssat_path', None)
        completed_sims = spatial_cm.model.run(spatial_cm.model.crop_code, crop=spatial_cm.crop,planting_window=spatial_cm.config.MANAGEMENT.plantingWindow,
                                        ncores = spatial_cm.config.GENERAL_INFO.ncores,
                                            bin_path = spatial_cm.config.GENERAL_INFO.bin_path, remove_tmp_folder=True)
        
        
        refraster = rio.open_rasterio(os.path.join(completed_sims._tmp_path,'ref_raster.tif'))
        model_data = update_dssat_data_using_path(completed_sims._tmp_path)

        mlt_pot_yield = create_mlt_yield_raster(refraster, model_data, ycol_name='HWAH')
        summ_yield  = summarize_spatial_yields_by_time_window(mlt_pot_yield, 
                                                            plantingWindow = spatial_cm.config.MANAGEMENT['plantingWindow'])

        cs = spatial_cm.config.GENERAL_INFO.country_code
        export_data(mlt_pot_yield, f'simulations_{cs}_{roi_name}.nc')
        export_data(summ_yield, f'summarized_simulations_{cs}_{roi_name}.nc')
        

def main():
    logging.info("Starting to download data")
    args = parse_args()
    
    # reading configuration
    cm_configuration = OmegaConf.load(args.config)
    
        ## get the spatial boundaries if the file is not provided
    cs = cm_configuration.GENERAL_INFO.country_code.upper()

    if cm_configuration.SPATIAL_INFO.get('geospatial_path', None) is None:
        
        adm_level = cm_configuration.SPATIAL_INFO.adm_level
        url = f"https://www.geoboundaries.org/api/current/gbOpen/{cs}/ADM{adm_level}/"
        
        gpd.read_file(requests.get(url).json()["gjDownloadURL"]).to_file(f'data/country_{cs}_ADM{adm_level}.shp')
        cm_configuration.SPATIAL_INFO.geospatial_path = f'data/country_{cs}_ADM{adm_level}.shp'
        
        
    # Initialize the spatial crop modeling class
    cm_sp = SpatialCM(configuration_dict=cm_configuration)
    
    geocode = cm_configuration.SPATIAL_INFO.get('feature', None)
    
    if geocode is not None:
        # Specify the region of interest by its geocode this can be also done using the feature index
        
        simulate_roi(cm_sp, geocode)
        
    else:
        for geocode in cm_sp.config.SPATIAL_INFO.feature_name:
            simulate_roi(cm_sp, geocode)
        

    logging.info("Simulation completed!")


if __name__ == "__main__":
    main()