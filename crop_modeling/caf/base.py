import numpy as np

import os
import pandas as pd
import platform
import subprocess
import yaml
import xarray

from . import MODULE_PATH

from .files_export import CAFDEM, CAFSoil, CAFWeather
from .management import CAFManagement
from .trees import Tree
from ..utils.model_base import ModelBase

from datetime import datetime
from typing import List, Optional, Dict, Tuple

from pathlib import Path

def cafos_model():
    opsys = platform.system().lower()

    bin_path = Path(MODULE_PATH).joinpath('crop_modeling/caf/dll')
    if 'windows'in opsys:
        dll_path = bin_path.joinpath('CAF2021_Win.DLL').absolute()
    else: 
        dll_path = bin_path.joinpath('CAF2021.DLL').absolute()

    return str(dll_path)



class PyCAF(ModelBase):
    """
    A class for coffee potential yield simulation, This implementation is based on the CAF decelopment did by
    # Ovalle-Rivera, O. et al. (2020). Assessing the accuracy and robustness of a
    #    process-based model for coffee agroforestry systems in Central America.
    #    Agroforestry Systems 94: 2033-2051.
    van Oijen, M., Dauzat, J., Harmand, J. M., Lawson, G., & Vaast, P. (2010). 
    Coffee agroforestry systems in Central America: II. Development of a simple process-based 
    model and preliminary results. Agroforestry Systems, 80(3).
    https://doi.org/10.1007/s10457-010-9291-1
    
    """
        
    def __init__(self, path):
        super().__init__(path)
        self._paremeters = None
        print(MODULE_PATH)
    
    @staticmethod
    def _change_to_list(array):
        """
        Converts a NumPy array to a list if applicable.

        Parameters
        ----------
        array : Union[np.ndarray, Any]
            Input object to convert to a list if it is a NumPy array.

        Returns
        -------
        list or Any
            A list if the input was an array, otherwise the input itself.
        """
        return array.tolist() if isinstance(array, np.ndarray) else array,
    
    
    def _dict_config_file(self,
        output_path: str,
        fert: np.ndarray,
        coffee_prun: np.ndarray,
        tree_prun: np.ndarray,
        tree_thinning: np.ndarray,
        ndays: Optional[int] = None,
        dll_path: Optional[str] = None
    ) -> Dict:
        """
        Creates a dictionary representation of the configuration file for the model.

        Parameters
        ----------
        output_path : str
            Path to save the configuration file.
        fert : np.ndarray
            Fertilization schedule.
        coffee_prun : np.ndarray
            Coffee pruning schedule.
        tree_prun : np.ndarray
            Tree pruning schedule.
        tree_thinning : np.ndarray
            Tree thinning schedule.
        ndays : int, optional
            Number of simulation days, by default None.
        dll_path : str, optional
            Path to the CAF model DLL file, by default None.

        Returns
        -------
        dict
            Dictionary representation of the configuration file.
        """
        return{
            'GENERAL':
                {   
                    'caf_dll_path' : cafos_model() if dll_path == None else dll_path,
                    'working_path': output_path
                },
            'MANAGEMENT': {
                'coffe_prun':  self._change_to_list(coffee_prun),
                'tree_prun': self._change_to_list(tree_prun),
                'tree_thinning': self._change_to_list(tree_thinning),
                'fertilization': self._change_to_list(fert),
            },
            'PARAMETERS': self._change_to_list(self.parameter_values),
            'WEATHER': self._change_to_list(self._weather),
            'NDAYS': self._calculate_total_days() if ndays == None else ndays
        }
        
        
    def from_datacube_to_files(self, xrdata: xarray.Dataset, 
        data_source: str = 'climate',
        target_crs: str = 'EPSG:4326',
        group_by: Optional[str] = None,
        group_codes: Optional[dict] = None,
        outputpath: Optional[str] = None
    ) -> None:
        """
        Converts a datacube to specific files for the model.

        Parameters
        ----------
        xrdata : Any
            Input datacube for processing.
        data_source : str, optional
            Type of data source ('climate', 'dem', or 'soil'), by default 'climate'.
        target_crs : str, optional
            Target coordinate reference system, by default 'EPSG:4326'.
        group_by : str, optional
            Grouping parameter for the data, by default None.
        group_codes : dict, optional
            Group codes for classification, by default None.
        outputpath : str, optional
            Output path for the generated files, by default None.
        """
        
        outputpath = outputpath if outputpath else 'tmp'
        
        if data_source == 'climate':
            cafweather = CAFWeather(xrdata)
            cafweather(depth_var_name = 'date', group_by= group_by, outputpath = outputpath, codes=group_codes, target_crs=target_crs)
            
        if data_source == 'dem':
            cafdem = CAFDEM(xrdata)
            cafdem(group_by= group_by, outputpath = outputpath, codes=group_codes, target_crs=target_crs)

        if data_source == 'soil':        
            cafsoil = CAFSoil(xrdata)
            cafsoil(depth_var_name = 'depth', group_by= group_by, outputpath = outputpath, codes=group_codes, target_crs=target_crs)

    def _calculate_total_days(self):
    
        init_date = datetime.strptime('{:0.0f}-{:0.0f}'.format(self._weather[0][0],self._weather[0][1]), '%Y-%j')
        end_date = datetime.strptime('{:0.0f}-{:0.0f}'.format(self._weather[-1][0],self._weather[-1][1]), '%Y-%j')
        difdays = end_date - init_date
        return difdays.days-1
    
    
    @property
    def parameter_values(self) -> np.ndarray:
        """
        Returns CAF parameter values as a NumPy array.

        Returns
        -------
        np.ndarray
            Array of parameter values.
        """
        return self._paremeters.iloc[:,1].values

    def change_parameters(self, parameters_to_modify: Dict[str, float]) -> pd.DataFrame:
        """
        Modifies specified parameters.

        Parameters
        ----------
        parameters_to_modify : dict
            A dictionary where keys are parameter names, and values are the new values.

        Returns
        -------
        pd.DataFrame
            Updated parameters DataFrame.
        """
        assert self._paremeters is not None, 'define parameters first'
        paramschanged = []
        for k,v in parameters_to_modify.items():
            if k in self._paremeters.name.values:
                self._paremeters.loc[self._paremeters.name==k,"1"] = v
                paramschanged.append(k)
        
        print(f'Folowing parameters were changed : {paramschanged}')
        return self._paremeters
    
    def set_parameters(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Loads the caf model parameters from a file.

        Parameters
        ----------
        file_path : str, optional
            Path to the parameter file, by default None.

        Returns
        -------
        pd.DataFrame
            The parameters loaded into a DataFrame.
        """
        if self._paremeters is None:
            if file_path is None: file_path = os.path.join(os.getcwd(), 'crop_modeling/caf/parameters/parameters_default.txt')
            self._paremeters = pd.read_csv(file_path, sep="\t")
            self._paremeters.rename(columns={self._paremeters.columns[0]:'name'}, inplace=True)
            
        return self._paremeters
    
    def set_soil_parameters(self, soil_file_path: str, depth: str = '5-15') -> None:
        """
        Sets soil parameters based on the soil file and depth.

        Parameters
        ----------
        soil_file_path : str
            Path to the soil information file.
        depth : str, optional
            Depth range, by default '5-15'.
        """
        
        soil_info = pd.read_csv(soil_file_path)
        som = soil_info.loc[soil_info.DEPTH == depth, 'CSOM0'].values[0]
        self.change_parameters({'CSOM0': round(som,1)})
    
    def set_location_parameters(self, dem_file_path: str) -> None:
        """
        Sets location parameters (e.g., slope and latitude).

        Parameters
        ----------
        dem_file_path : str
            Path to the DEM information file.
        """
        dem_info = pd.read_csv(dem_file_path)
        slope = dem_info['SLOPE'].values[0]
        lat = dem_info['LAT'].values[0]
        self.change_parameters({'SLOPE': round(slope), 'LAT': round(lat, 2)})
        
    def set_tree_parameters(self, species_name: str, tree_density: Optional[float] = None) -> None:
        """
        Sets tree-related parameters.

        Parameters
        ----------
        species_name : str
            Name of the tree species.
        tree_density : float, optional
            Density of the trees, by default None.
        """
        
        tree = Tree()
        assert not isinstance(species_name,list), 'Only support one tree species, CAF2021 model can simulate up to three' #TODO: implement more than one tree
        
        tree_parameters = tree.species_params(species_name)
        if tree_density is not None: tree_parameters['TREEDENS0'] = tree_density
        id_tree = 1
        parameters_to_modify = {'{}({})'.format(k, id_tree):v for k,v in tree_parameters.items()}
        self.change_parameters(paremetersto_modify=parameters_to_modify)
    
    def read_weather(self, weather_path: str, init_year: Optional[int] = None, init_doy: Optional[int] = None) -> np.ndarray:
        """
        Reads weather data from a CSV file and filters it based on initial year and day of the year.

        Parameters
        ----------
        weather_path : str
            Path to the weather data file.
        init_year : int, optional
            Filter records with a year greater than or equal to this value, by default None.
        init_doy : int, optional
            Filter records with a day of year (DOY) greater than or equal to this value, by default None.

        Returns
        -------
        np.ndarray
            Weather data as a NumPy array.
        """
        
        weatherdf = pd.read_csv(weather_path)
        weather_data = np.zeros((weatherdf.shape[0], 8), dtype= float)
        if init_year: weatherdf = weatherdf.loc[weatherdf.year >= init_year]
        if init_doy: weatherdf = weatherdf.loc[weatherdf.doy >= init_doy]

        weather_data[:weatherdf.shape[0],:] = weatherdf.values
        self._weather = weather_data
        
        return weather_data

    
    def run(self, cwd: str = None) -> None:
        """
        Executes the CAF model using the configuration file.

        Parameters
        ----------
        cwd : str
            Current working directory for running the model.
        """
        for pathiprocess in self._process_paths:
            if cwd is None: cwd = MODULE_PATH
            print('RScript', './r_scripts/r_run_caf.R', os.path.join(pathiprocess, 'config_file.yaml'))
            returned_value = subprocess.call(['RScript', './r_scripts/r_run_caf.R', 
                                            os.path.join(pathiprocess, 'config_file.yaml')], shell= True, cwd=cwd)
            print(returned_value)
    
    def set_up_management(self, management_config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sets up management schedules based on the provided configuration.

        Parameters
        ----------
        management_config : dict
            A dictionary containing management schedules.

        Returns
        -------
        tuple of np.ndarray
            Fertilization, coffee pruning, tree pruning, and tree thinning schedules.
        """
        management = CAFManagement()
        
        coffe_prun = management.coffe_prunning_schedule(**management_config.coffee_prunning)
        tree_prun = management.tree_prunning_schedule(**management_config.tree_prunning)
        tree_thinning = management.tree_thinning_schedule(**management_config.tree_thinning)
        fert = management.fertilization_schedule(**management_config.fertilization)
        
        return fert, coffe_prun, tree_prun, tree_thinning
    
    def organize_env(self, **kwargs) -> None:
        """
        Organizes the environment and prepares for model execution.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for setting up the environment.
        """
        
        if len(self._process_paths) == 0: self.find_envworking_paths(file_ext='csv')
        
        for pathiprocess in self._process_paths:
            print(pathiprocess)
            self.set_soil_parameters(os.path.join(pathiprocess, 'cafsoil.csv'))
            self.set_location_parameters(os.path.join(pathiprocess, 'cafdem.csv'))
            _ = self.read_weather(os.path.join(pathiprocess, 'cafweather.csv'))
            self.write_run_config_file(pathiprocess, **kwargs)
            
    def write_run_config_file(self,  output_path: str,
        fert: np.ndarray,
        coffee_prun: np.ndarray,
        tree_prun: np.ndarray,
        tree_thinning: np.ndarray,
        ndays: Optional[int] = None,
        dll_path: Optional[str] = None
    ) -> None:
        """
        Writes the configuration file for the model run.

        Parameters
        ----------
        output_path : str
            Path to save the configuration file.
        fert : np.ndarray
            Fertilization schedule.
        coffee_prun : np.ndarray
            Coffee pruning schedule.
        tree_prun : np.ndarray
            Tree pruning schedule.
        tree_thinning : np.ndarray
            Tree thinning schedule.
        ndays : int, optional
            Number of simulation days, by default None.
        dll_path : str, optional
            Path to the CAF model DLL file, by default None.
        """
        config_info = self._dict_config_file(output_path, fert, coffee_prun, 
                                             tree_prun, tree_thinning, ndays, dll_path)
        
        config_path = os.path.join(output_path, 'config_file.yaml')
        #fn = os.path.join(self.path, 'experimental_file_config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(config_info, file, default_flow_style=False)
        self._config_path = config_path