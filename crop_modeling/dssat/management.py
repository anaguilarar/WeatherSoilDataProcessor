from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob
import subprocess
import yaml

from .soil import DSSATSoil_base
from .weather import DSSAT_Weather

from typing import Dict, Tuple
import pandas as pd
import os


class DSSATManagement_base():
    """
    A class for managing DSSAT experiments and configuration files.

    Provides utilities for handling DSSAT crop, soil, and weather data, as well as
    generating configuration files for simulations.
    """
    
    @property
    def extensions(self) -> Dict[str, str]:
        """
        Returns file extensions for DSSAT data types.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping data types to file extensions.
        """
        
        return {
        'weather': 'WTH',
        'soil' : 'SOL',
        'experiment' : 'MZX'
        }
        
    @property
    def crop_codes(self) -> Dict[str, str]:
        """
        Returns crop codes for DSSAT simulations.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping crop names to their DSSAT crop codes.
        """
        
        return {
        "Maize": "MZ",
        'Millet': "ML",
        'Sugarbeet': "BS",
        'Rice': "RI",
        'Sorghum': "SG",
        'Sweetcorn': "SW",
        'Alfalfa': "AL",
        'Bermudagrass': "BM",
        'Soybean': "SB",
        'Canola': "CN",
        'Sunflower': "SU",
        'Potato': "PT",
        'Tomato': "TM",
        'Cabbage': "CB",
        'Sugarcane': "SC",
        "Wheat": "WH",
        "Beans": "BN",
        "Cassava": "CS"
        }
        
    
    def __init__(self, crop: str, variety: str, planting_date: str, harvesting_date: str) -> None:
        """
        Initialize a DSSAT management object.

        Parameters
        ----------
        crop : str
            Name of the crop (e.g., "Maize").
        variety : str
            Crop variety identifier.
        planting_date : str
            Planting date in 'YYYY-MM-DD' format.
        harvesting_date : str
            Harvesting date in 'YYYY-MM-DD' format.
        """
        self.crop = crop
        self.variety = variety
        self.crop_code = self.crop_codes.get(crop, None)
        self.planting_date = datetime.strptime(planting_date,  '%Y-%m-%d')
        self.harvesting_date = datetime.strptime(harvesting_date,  '%Y-%m-%d')
        self.starting_date = self.planting_date - relativedelta(months=1)
        
    
    def total_climate_years(self) -> int:
        """
        Calculates the total number of years in the climate data.

        Returns
        -------
        int
            Total number of years covered by the climate data.
        """
        weather_dates = DSSAT_Weather.get_dates_from_file(self._weather[0])
        years = [datetime.strptime(d,  '%Y%j').year for d in weather_dates]
        return years[-1] - years[0]
    

    def check_files(self, template: str) -> None:
        """
        Validates the existence of required files based on the template.

        Parameters
        ----------
        template : str
            Path to the experiment template file.

        Raises
        ------
        AssertionError
            If required soil or weather files are missing.
        """
        assert template.endswith(self.extensions['experiment']),  f"Template must end with '{self.extensions['experiment']}'"
        self._soil = glob.glob(self.path+'/*{}*'.format(self.extensions['soil']))
        self._weather = glob.glob(self.path+'/*{}*'.format(self.extensions['weather']))
        assert len(self._weather)>0, "No weather files found."
        assert len(self._soil)>0, "No soil files found."
    
    def general_info(self, **kwargs) -> None:
        """
        Updates general experiment configuration with provided keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            General experiment configuration options, such as: output_name, roi_id, plantingWindow, and index_soilwat.
        """
        self.output_name = kwargs.get('output_name', 'EXP')
        self.roi_id = kwargs.get('roi_id', 1)
        self.plantingWindow = kwargs.get('plantingWindow',1)
        self.fertilizer = kwargs.get('fertilizer',False) # TODO: implement true scenario
        self.index_soilwat = kwargs.get('index_soilwat',1)
        
    def soil_data(self) -> Tuple[str, pd.DataFrame]:
        """
        Retrieves soil properties and ID.

        Returns
        -------
        Tuple[str, pd.DataFrame]
            Soil ID and a DataFrame of soil properties.
        """
        
        soildata = DSSATSoil_base.soil_properties_as_df(self._soil[0])
        soildata = soildata[['SDUL', 'SLLL','SLB']].values.T.astype(float)
        soilid =  DSSATSoil_base.check_id(self._soil[0])
        
        return (soilid, soildata)

    def check_weather_fn(self) -> str:
        """
        Ensures the weather file has a valid DSSAT filename.

        Returns
        -------
        str
            Updated weather file name.
        """
        
        weather_fn = os.path.basename(self._weather[0])
        if len(weather_fn)>15:
            dirname = os.path.dirname(self._weather[0])
            weather_fn = 'WHTE0001.WTH'
            os.rename(self._weather[0], os.path.join(dirname, weather_fn))

        return weather_fn
    
    def r_configuration_file(self, filex_template: str, number_years: int, soilid: str,
                             soildata: pd.DataFrame, weather_fn: str) -> Dict:
        
        """
        Generates the DSSAT R configuration file.

        Parameters
        ----------
        filex_template : str
            Path to the experiment template.
        number_years : int
            Number of years for simulation.
        soilid : str
            Soil ID.
        soildata : pd.DataFrame
            Soil properties.
        weather_fn : str
            Weather file name.

        Returns
        -------
        Dict
            Configuration data for the experiment.
        """
        
        return{
            'GENERAL': {
                'roi_id' : 1,
                'working_path': self.path,
                'number_years': number_years,
                'output_name': 'EXP'
            },
            'MANAGEMENT': {
                'template_path':  filex_template,
                'crop_code': self.crop_code,
                'varietyid': self.variety,
                'plantingWindow': self.plantingWindow,
                'startingDate': self.starting_date.strftime('%Y-%m-%d'), ## one month before of actual date
                'plantingDate': self.planting_date.strftime('%Y-%m-%d'),
                'harvestDate':  self.harvesting_date.strftime('%Y-%m-%d'),
                'fertilizer': self.fertilizer
            },
            'SOIL':{
                'source': 'ISRIC V2',
                'index_soilwat': self.index_soilwat,
                'ID_SOIL': soilid,
                'SDUL': soildata[0].tolist(),
                'SLLL': soildata[1].tolist(),
                'SLB': soildata[2].tolist(),
            },
            'WEATHER':{
                'file_name': weather_fn[:-4] ## the .WTH extension is not saved
            }
            
        }
        
        
    
    def create_file_using_rdssat(self, filex_template: str, path: str, **kwargs) -> None:
        """
        Creates an experimental file using R-DSSAT tools.
        To implement this function it is neccesary to previosly setup RScript in terminal
        besides the working path must contains the SOIL and WEATHER dssat files
        Parameters
        ----------
        filex_template : str
            Path to the experiment template file.
        path : str
            Path to the working directory.
        **kwargs : dict
            Additional configuration options.
        """
        self.path = path
        
        self.check_files(filex_template)
        
        number_years = self.total_climate_years()
        soilid, soildata = self.soil_data()
        self.general_info(**kwargs)
        weather_fn = self.check_weather_fn()
        config_file = self.r_configuration_file(filex_template, number_years, soilid, soildata, weather_fn)
        
        config_path = os.path.join(self.path, 'experimental_file_config.yaml')
        #fn = os.path.join(self.path, 'experimental_file_config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(config_file, file)
            
        print(f"Configuration file written: {config_path}")
        
        returned_value = subprocess.call(['RScript', 'r_create_experimental_files.R', f'{config_path}'] , shell= True)
        print(returned_value)