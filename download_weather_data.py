from omegaconf import OmegaConf
from spatialdata.climate_data import ClimateDataDownload
from spatialdata.gis_functions import get_boundaries_from_path
import logging
import argparse


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
    if config.SPATIAL_INFO.get('spatial_file',None):
        extent = get_boundaries_from_path(config.SPATIAL_INFO.get('spatial_file',None), round_numbers = True)
    else:
        extent = config.SPATIAL_INFO.extent
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


if __name__ == "__main__":
    main()