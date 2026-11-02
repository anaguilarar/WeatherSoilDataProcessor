from omegaconf import OmegaConf
from spatialdata.soil_data import SoilGridDataDonwload
from spatialdata.gis_functions import get_boundaries_from_path
from crop_modeling.utils.u_soil import get_soil_datacube
from spatialdata.datacube import create_dimension
from spatialdata.xr_dict import CustomXarray, from_dict_toxarray
import logging
import argparse
import os
from crop_modeling.utils.process import set_encoding, check_crs_inxrdataset


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Getting weather data')
    parser.add_argument('--config', 
                        default='weather_data_downloading_config.yaml', help='config file path')

    args = parser.parse_args()
    return args


def main():
    logging.info("Starting to download data")
    args = parse_args()

    # reading configuration
    config = OmegaConf.load(args.config)

    logging.info(f"Reading configuration from {args.config}")
    print('spatial_file: ', config.SPATIAL_INFO.get('spatial_file',None))
    if config.SPATIAL_INFO.get('spatial_file',None):
        extent = get_boundaries_from_path(config.SPATIAL_INFO.get('spatial_file',None),
                                          crs = config.SPATIAL_INFO.crs, 
                                          round_numbers = True)
    else:
        extent = config.SPATIAL_INFO.extent
    # Initialize ClimateDataDownload object with config parameters
    outputpath = os.path.join(config.PATHS.output_path, config.GENERAL.suffix)

    if config.GENERAL.task == 'download':
        soildata = SoilGridDataDonwload(soil_layers= config.SOIL.variables,
                                depths= config.SOIL.depths,
                                output_folder= outputpath)

        soildata.download_soilgrid(boundaries= extent)

    elif config.GENERAL.task == 'datacube':
        prim_conf = OmegaConf.to_container(config)

        prim_conf['SPATIAL_VECTOR'] = {'boundaries': config.SPATIAL_INFO.get('spatial_file',None)}
        prim_conf['SOIL'].update({'setup_parameters': {  # Parameters for creating the soil data cube
            'path': outputpath,  # Path to raw soil data
            'variables': prim_conf['SOIL']['variables'],
            'depths': prim_conf['SOIL']['depths'],
            'crs': 'ESRI:54052',  # Spatial Coordinate System for SoilGrids data
            'reference_variable': 'wv1500',  # Variable used as spatial resolution reference
            'target_crs': 'EPSG:4326'
        }})
        
        conf = OmegaConf.create(prim_conf)
        print(conf)
        soil_datac = get_soil_datacube(conf)
        #customdict = {k: CustomXarray(v, dataformat = 'CHW').custom_dict for k,v in soil_datac.items()}
        
        soil_datac =create_dimension(soil_datac, newdim_name= 'depth', isdate=False)    
        if 'band_data' in soil_datac.data_vars:
            soil_datac = soil_datac.drop_vars('band_data')

        fn = conf.PATHS.get('datacube_fn', None) if conf.PATHS.get('datacube_fn', None) else f'soil_data_{config.GENERAL.suffix}.nc'
        fn = os.path.join(config.PATHS.output_path, fn)
        dcengine = 'netcdf4'
        encoding = set_encoding(soil_datac)
        soil_datac.rio.write_crs(conf.SOIL.setup_parameters.target_crs, inplace=True)

        soil_datac = check_crs_inxrdataset(soil_datac)
        
        soil_datac.to_netcdf(fn, encoding = encoding, engine = dcengine)

        print(f'file_saved in {fn}')
    
            
    logging.info("Data download completed!")


if __name__ == "__main__":
    main()