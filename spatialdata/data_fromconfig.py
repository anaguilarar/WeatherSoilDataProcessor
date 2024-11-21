import geopandas as gpd
from .gis_functions import get_boundaries_from_path
from .climate_data import MLTWeatherDataCube
from .soil_data import SoilDataCube
from .files_manager import IntervalFolderManager, SoilFolderManager


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