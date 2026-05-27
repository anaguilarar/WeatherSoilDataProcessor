import os
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import lognorm

import concurrent.futures
import pandas as pd
import math
import copy
import xarray
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

from ._base import PlantParameters, BananaCycle
from .management import BANANAFerti, banana_fertilizer_schedule
from .files_export import BANANAWeatherTable, BANANADEMTable, BANANASoilTable
from .weather import BanWeather
from .soil import BanSoil, BANANASoilMat
from .output import BananaNOutputData, BananaNReporter
from ..utils.model_base import ModelBase

def check_date(date):
    if isinstance(date, str): 
        return datetime.strptime(date, '%Y-%m-%d') if '-' in date else datetime.strptime(str(date),'%Y%m%d')
    elif isinstance(date, int):
        return datetime.strptime(str(date),'%Y%m%d')
    return date
            
def set_up_weather_soil_parameters( dem_path, soil_path, weather_path):
    """
    Run the simple model with the provided parameters.

    Parameters
    ----------
    sm_params : SpatialCM
        The parameters for the simple model.
    dem_filename : str 
        The filename for the DEM data.
    soil_filename : str
        The filename for the soil data.
    weather_filename : str
        The filename for the weather data.
    """
    location_params = pd.read_csv(dem_path)
    weatherdata = BanWeather(weather_path, latitude=float(location_params['LAT'].values[0]), longitude=float(location_params['LONG'].values[0]), altitude=int(location_params['ELEV'].values[0]), date_format = '%Y-%m-%d')
    bansoil = BanSoil(path = soil_path)
    
    return  weatherdata, bansoil

def banana_cycle_weekly_weather(bananasp_env, total_weeks: int):

    weather_manager = bananasp_env['weathert']
    starting_date = bananasp_env['planting_date']
    # weather
    
    starting_date = check_date(starting_date)
    ending_date = starting_date + timedelta(days=(total_weeks+1)*7)
    ending_date = datetime.strftime(ending_date, '%Y-%m-%d')

    weather_df = weather_manager.weekly_weather(starting_date = starting_date, ending_date = ending_date)

    return weather_df.reset_index().rename(columns= {'index': 'week'}).to_dict(orient = 'records')


def soil_initial_conditions(bananasp_env):
    soil_manager = bananasp_env['soil']

    weather_manager = bananasp_env['weathert']
    pldate = bananasp_env['planting_date']

    def layer_properties(layer):
        layerprts = soil_manager.get_son(layer)
        smn = soil_manager.get_initial_smn(weather_manager, 
                    pldate,
                    layerprts['son'],
                    layerprts['clay'], 
                    layer_depth_cm=soil_manager.depths[layer][1] - soil_manager.depths[layer][0])
                        
        return layerprts, smn


    layer_1, smn1 = layer_properties(0)
    layer_2, smn2 = layer_properties(1)


    return {
        'wsol1': layer_1['Wsol'], 'wsol2': layer_2['Wsol'], 
        'son': (layer_1['son']+ layer_2['son'])/2, 'smn_depth1': smn1, 'smn_depth2': smn2
    }


def generate_mat_parameters(mu: float, sigma: float, nban: int, seed: int = None) -> np.ndarray:
    """
    Directly generates biological parameters for N mats.

    Parameters
    ----------
    mu : float
        Mean of the log-normal distribution (in log space).
    sigma : float
        Standard deviation of the log-normal distribution (in log space).
    nban : int
        Number of mats (samples) to generate.
    seed : int, optional
        Optional random seed for reproducibility.

    Returns
    -------
    np.ndarray
        A numpy array of generated parameters for the mats.
    """
    if seed is not None:
        np.random.seed(seed)
    
    return lognorm.rvs(s=sigma, scale=np.exp(mu), size=nban)

def generate_lognorm_pool(mu: float, sigma: float, size: int = 1000) -> np.ndarray:
    """
    Generates the log-normal pools exactly as R's qlnorm does.

    Parameters
    ----------
    mu : float
        Mean of the log-normal distribution (in log space).
    sigma : float
        Standard deviation of the log-normal distribution (in log space).
    size : int, optional
        Number of probabilities to evaluate, by default 1000.

    Returns
    -------
    np.ndarray
        A numpy array containing the values of the log-normal percent point function.
    """
    probs = np.arange(1, size) / size
    return lognorm.ppf(probs, s=sigma, scale=np.exp(mu))



class BananaMat_cycles(PlantParameters):
    """
    Represents a banana mat managing multiple plant cycles and soil state.
    """
    def __init__(self, mat_id: int, density: float, pool_sdd: np.ndarray, init_soil_parameters: Dict[str, float]):
        """
        Initialize the banana mat cycles manager.

        Parameters
        ----------
        mat_id : int
            Identifier for the mat.
        density : float
            Planting density (number of plants per hectare).
        pool_sdd : np.ndarray
            Thermal time interval between planting/emergence and sucker emergence.
            Stochastically defined with a lognormal distribution.
        init_soil_parameters : Dict[str, float]
            Initial soil parameters for the mat.
        """
        self.mat_id = mat_id
        self.surface_area = 10000.0 / density # m2 per mat
        
        self.cycles: List[BananaCycle] = [BananaCycle(cycle_id=1, sdd_pss=1755.0)]
        self.soil = BANANASoilMat(mat_id, **init_soil_parameters)
        self.ferti = BANANAFerti()
        
        self.pool_sdd = pool_sdd
        self.penal = int(np.random.choice(generate_lognorm_pool(1.0, 0.43)))
        
        
    def update_mat(self, week: int, temperature: float, radiation: float, rain: float, et: float, is_fertilizer_applied: bool, of_amount: float, minf_amount: float) -> None:
        """
        Update the state of the mat for a given week based on environmental conditions and management practices.

        Parameters
        ----------
        week : int
            The current week of the simulation.
        temperature : float
            Average or accumulated temperature for the week.
        radiation : float
            Solar radiation for the week.
        rain : float
            Rainfall amount for the week.
        et : float
            Evapotranspiration for the week.
        is_fertilizer_applied : bool
            True if organic fertilizer is applied in this week.
        of_amount : float
            Amount of organic fertilizer applied.
        minf_amount : float
            Amount of mineral fertilizer applied.
        """
        r_t = week + 1
        
        if r_t < self.penal:
            for cycle in self.cycles:
                cycle.stress = 0.0
                
        mat_lai = sum(c.laiban for c in self.cycles if c.recolte == 0) # Calculate the total leaf area index of the mat by summing the leaf area index of all cycles that have not been harvested
        
        pari_ban = self.Ea * self.Ec * radiation * (1 - np.exp(-self.kBAN * mat_lai)) # PAR intercepted by the canopy
        
        sum_dNResBAN = 0 # total nitrogen mineralized from banana residues across all cycles in the mat
        sum_dNHumBAN = 0 # total nitrogen humified from banana residues across all cycles in the mat
        sum_dDMBANtot = 0 # total newly formed dry biomass across all cycles in the mat
        
        
        for i, cycle in enumerate(self.cycles):        
            cycle.update_phenology(temperature)
            
            eb_ban = 1.9
            if cycle.sominiflo < 1 and cycle.cycle == 1:
                eb_ban = 1.2
            elif cycle.sominiflo >= 1 and cycle.cycle == 1:
                eb_ban = 1.2
            elif cycle.sominiflo < 1:
                eb_ban = 1.9
            else:
                eb_ban = 2.5
            
            lai_banprod = cycle.laiban / mat_lai if mat_lai > 0 else 0 # Proportion of the mat's leaf area index that belongs to the current cycle
            ## biomass production
            cycle.dDMBANtot = (eb_ban * pari_ban * self.surface_area * (1 - cycle.recolte)) * cycle.stress * lai_banprod # Total newly formed dry biomass in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’ (g DM/week) ## equation 3 Ruillé et al. (2025)            
            sum_dDMBANtot += cycle.dDMBANtot
            
            # allocation to sucker
            cycle.alloc_suc = cycle.psk * cycle.dDMBANtot if cycle.reject >= 1 else 0
            cycle.dDMBAN = cycle.dDMBANtot - cycle.alloc_suc
            #allocfromPM
            if cycle.reject >= 1 and i + 1 < len(self.cycles):
                self.cycles[i+1].received_biomass = cycle.alloc_suc
            
            cycle.update_biomass_and_allocation(temperature, self.surface_area)
            cycle.calculate_mineralN_fromBANresidues()
            
            sum_dNResBAN += cycle.dNRESBAN
            sum_dNHumBAN += cycle.dNhumBAN
        
        new_cycles = []
        for cycle in self.cycles:
            if cycle.reject == 1 and not cycle.reject_triggered:
                cycle.reject_triggered = True
                new_sdd = np.random.choice(self.pool_sdd)
                new_cycles.append(BananaCycle(cycle.cycle + 1, new_sdd))
        
        self.cycles.extend(new_cycles)
        total_biomass = sum(c.ban_biomass for c in self.cycles)
        pnBAN = (4.78 * math.pow(total_biomass,-0.13))/100 if total_biomass > 0 else 0
        dNBAN = sum_dDMBANtot * pnBAN
        
        dNBAN_1 = dNBAN * self.ZrBAN1
        dNBAN_2 = dNBAN * self.ZrBAN2
    
        self.ferti.apply_fertilizer(is_fertilizer_applied, of_amount, self.residue_c_yield, self.bm_decomr)        
        
        # Water balance
        kc = -0.0487 * mat_lai**2 + 0.3925 * mat_lai + 0.4235
        
        etr = et * kc
        et1 = etr * self.ZrBAN1
        et2 = etr * self.ZrBAN2
                
        wal1 = max(0, rain - et1 - (self.soil.SW1 - self.soil.wsol1))
        wal = max(0, wal1 - et2 - (self.soil.SW2 - self.soil.wsol2))
        
        self.soil.wsol1 = max(0, self.soil.wsol1 + rain - et1 - wal1)
        self.soil.wsol2 = max(0, self.soil.wsol2 + wal1 - et2 - wal)
        
        # leaching
        nl1 = self.soil.SMN1 * (1 -np.exp(-self.soil.kl1 * wal1 / self.soil.SW1))
        nal = self.soil.SMN2 + nl1
        nl = max(0, nal * (1-np.exp(-self.soil.kl2 * wal / self.soil.SW2)))
        
        # soil organic matter mineralization 
        
        mos = self.soil.SON * (-np.exp(-self.Ksom1 * (week + 1)) + np.exp(-self.Ksom1 * week))
        
        self.soil.SON = self.soil.SON - mos + sum_dNHumBAN + self.ferti.dNhumOF
        
        # uptake
        
        uban1 = min(dNBAN_1, self.soil.SMN1)
        uban2 = min(dNBAN_2, self.soil.SMN2)
        uban = uban1 + uban2
        
        #
        self.soil.SMN1 = max(0.0, self.soil.SMN1 + mos - nl1 + minf_amount - uban1 + sum_dNResBAN + self.ferti.dNRESOF)
        self.soil.SMN2 = max(0.0, self.soil.SMN2 - nl + nl1 - uban2)        
        self.soil.SMN = self.soil.SMN1 + self.soil.SMN2

        
        mat_stress = uban / dNBAN if sum_dDMBANtot > 0 and dNBAN > 0 else 1.0

        for cycle in self.cycles:
            cycle.stress = mat_stress
            
            
class BANANAField:
    """
    Simulates a full banana field consisting of multiple mats.

    Parameters
    ----------
    nban : int, optional
        Number of banana mats to simulate in the field, by default 40.
    density : int, optional
        Planting density (mats per hectare), by default 1300.0.

    Attributes
    ----------
    nban : int
        Number of mats in the field.
    pool_sdd : np.ndarray
        Pool of possible sum-of-degree-days generated for sucker emergence.
    mats : List[BananaMat]
        List of BananaMat objects representing the field.
    """
    def __init__(self, nban: int = 40, density: float = 1300.0, init_soil_parameters: Dict[str, float] = None):
        if init_soil_parameters is None:
            init_soil_parameters = {}
        self.nban = nban
        self.density = density   

        self.pool_sdd: np.ndarray = generate_lognorm_pool(7.102693, 0.1240221)
        self.flowering_delay_weeks = generate_lognorm_pool(1.0, 0.43) # Allows you to create the week offset (flowering), just for the first cycle 
        self.mats: List[BananaMat_cycles] = [BananaMat_cycles(i, density, self.pool_sdd, init_soil_parameters) for i in range(nban)]
    
    
    def simulate(
        self,
        nb_weeks: int,
        weather_data: List[Dict[str, float]],
        ferti_schedule: List[Dict[str, Any]]
        ) -> List[Dict[str, float]]:
        """
        Runs the simulation for a specified number of weeks.

        Parameters
        ----------
        nb_weeks : int
            Number of weeks to simulate.
        weather_data : List[Dict[str, float]]
            List of dictionaries containing weather data (temp, rad, rain, et) per week.
        ferti_schedule : List[Dict[str, Any]]
            List of dictionaries defining fertilizer applications per week.

        Returns
        -------
        List[Dict[str, float]]
            A history (list of daily logs) over the simulated weeks containing averages of SMN and Biomass.
        """
        history = []

        for week in range(nb_weeks):
            w = weather_data[week]
            f = ferti_schedule[week]

            total_smn:float = 0.0
            total_biomass_g:float = 0.0
            total_fruit_g: float = 0.0

            for mat in self.mats:
                #delay_f = int(np.random.choice(self.flowering_delay_weeks))
                #if week<int(delay_f): mat.stress = 0
                mat.update_mat(
                        week = week, temperature = w['dtt'],
                        radiation = w['srad'], rain = w['rain'],
                        et = w['etr'], is_fertilizer_applied = f['application'], of_amount = f['q_org'], 
                        minf_amount = f['min_f'])
                
                total_smn += mat.soil.SMN
                total_biomass_g += sum(c.ban_biomass for c in mat.cycles)
                total_fruit_g += sum(c.bun_biomass for c in mat.cycles)

            avg_biomass_g_per_mat = total_biomass_g / self.nban
            avg_fruit_g_per_mat  = total_fruit_g / self.nban

            history.append({
                'Week': week,
                'Avg_SMN_kg_ha': total_smn / self.nban,
                'Avg_Bioamass_g_mat': avg_biomass_g_per_mat,
                'Avg_Fruit_g_mat': avg_fruit_g_per_mat
            })

        return history 
class PyBananaN(ModelBase):
    def __init__(self, path):
        super().__init__(path)
        self._enviroments = {}

    def set_up_environment(self, env_name = None, **kwargs):
        if env_name is None:
            env_name = 'env_{}'.format(len(self._enviroments)+1)
        
        if self._enviroments.get(env_name) is None:
            self._enviroments[env_name] = {k:v for k, v in kwargs.items()}
        else:
            self._enviroments[env_name].update({k:v for k, v in kwargs.items()})
        
    def reset_enviroments(self):
        self._enviroments = {}

    def set_up_crop(self, starting_date, cycle_duration_weeks, total_planting_windows = None, planting_density = None, verbose = True): ## TODO CROP PARAMETERS
        """
        Sets up the environmental and crop data for the simulation cycle.
        
        Parameters:
        -----------
        starting_date : str or datetime
            The date the crop is planted (t=0).
        time_step_weeks : int
            The moving window size / calculation step (e.g., 1 for weekly, 2 for bi-weekly).
        cycle_duration_weeks : int
            The total length of the crop cycle in weeks.
        verbose : bool
            If True, prints progress messages.
        """
        
        if len(self._process_paths) == 0: self.find_envworking_paths(file_ext='csv')
        
        for env_path in self._process_paths:
            dtstarting_date = check_date(starting_date)
            
            dates = pd.date_range(start=dtstarting_date, periods=total_planting_windows, freq = '7D')
            
            self.set_up_environment(
                env_name=env_path,
                planting_date=starting_date, 
                simulation_dates=dates,
                cycle_duration_weeks=cycle_duration_weeks,
                total_planting_windows = total_planting_windows,
                planting_density = planting_density
            )

            if verbose: print("crop data set up for {}".format(env_path))

    def organize_enviromental_data(self, verbose = True) -> None:
        """
        Organize the environment for the simulation.

        Parameters
        ----------

        planting_date : str
            Initial sowing date in 'YYYY-MM-DD' format

        """
        if len(self._process_paths) == 0: self.find_envworking_paths(file_ext='csv')
        for env_path in self._process_paths:
            dem_path = os.path.join(env_path, 'bananademtable.csv')
            soil_path = os.path.join(env_path, 'bananasoiltable.csv')
            weather_path = os.path.join(env_path, 'bananaweathertable.csv')

            weathert, soilt = set_up_weather_soil_parameters(soil_path =soil_path, dem_path = dem_path, weather_path = weather_path)

            self.set_up_environment(env_name = env_path, soil = soilt, weathert = weathert)

            if verbose: print("weather and soil data organized for {}".format(env_path))
    
    def from_datacube_to_files(self,
        xrdata: xarray.Dataset = None,
        xrdata_path: str = None,
        data_source: str = 'climate',
        target_crs: str = 'EPSG:4326',
        group_by: Optional[str] = None,
        group_by_layer: Optional[np.ndarray] = None,
        group_codes: Optional[dict] = None,
        outputpath: Optional[str] = None,
        pixel_scale: bool = False
    ) -> None:
        """
        Converts a datacube to specific files for the model.

        Parameters
        ----------
        xrdata : xarray.Dataset, optional
                Input xarray Dataset to be summarized. Required if `xrdata_path` is not provided.
        xrdata_path : str, optional
                Path to a NetCDF file containing the dataset. Used if `xrdata` is not passed.
        data_source : str, optional
            Type of data source ('climate', 'dem', or 'soil'), by default 'climate'.
        target_crs : str, optional
            Target coordinate reference system, by default 'EPSG:4326'.
        group_by : str, optional
            Grouping parameter for the data, by default None.
        group_by_layer : Optional[np.ndarray], optional
            Array with the categories data for grouping
        group_codes : dict, optional
            Group codes for classification, by default None.
        outputpath : str, optional
            Output path for the generated files, by default None.
        """
        
        outputpath = outputpath if outputpath else 'tmp'
        
        if data_source == 'climate':
            weatherprocessor = BANANAWeatherTable(xrdata, xrdata_path)
            weatherprocessor(depth_var_name = 'date', group_by= group_by, group_by_layer = group_by_layer, outputpath = outputpath, codes=group_codes, target_crs=target_crs, pixel_scale=pixel_scale)
            
        if data_source == 'dem':
            demprocessor = BANANADEMTable(xrdata, xrdata_path)
            demprocessor(group_by= group_by, group_by_layer = group_by_layer, outputpath = outputpath, codes=group_codes, target_crs=target_crs, pixel_scale=pixel_scale)

        if data_source == 'soil':        
            soilprocessor = BANANASoilTable(xrdata, xrdata_path)
            soilprocessor(depth_var_name = 'depth', group_by= group_by, group_by_layer = group_by_layer, outputpath = outputpath, codes=group_codes, target_crs=target_crs, pixel_scale = pixel_scale)


    def simulate_scenario(self, env_name: str, nban: int = 40): 
        """
        Simulate a single scenario for a given environment.

        Parameters
        ----------
        env_name : str
            Environment name.
        nban : int, optional
            Number of banana plants, by default 40.
        density : float, optional
            Planting density, by default 1300.0.

        Returns
        -------
        pd.DataFrame
            DataFrame with simulation results.
        """
        density = self._enviroments[env_name].get('planting_density', 1300)
        nbweeks = self._enviroments[env_name]['cycle_duration_weeks']
        fertischedule = self._enviroments[env_name].get('ferti_schedule', None)
        init_soil_parameters = soil_initial_conditions(self._enviroments[env_name])

        weather_weekly_data = banana_cycle_weekly_weather(self._enviroments[env_name], total_weeks=nbweeks)

        ferti_schedule = banana_fertilizer_schedule(fertischedule, nbweeks=nbweeks)

        model = BANANAField(nban=nban, density=density, init_soil_parameters=init_soil_parameters)
        history = model.simulate(nb_weeks = nbweeks, weather_data = weather_weekly_data, ferti_schedule = ferti_schedule)

        return history
    
    def _check_paths_existance(self, env_name: str) -> None:
        """
        Check if the paths for the given environment exist.
        """
        if not os.path.exists(self._enviroments[env_name]['soil'].path):return False
        if not os.path.exists(self._enviroments[env_name]['weathert'].path):return False
        return True
    
    def run_traitment(self, treatment_id, env_name, n_years = None):
        
        reporter = BananaNReporter()
        pltdate = self._enviroments[env_name]['simulation_dates'][treatment_id]
        pltdate = check_date(pltdate)
        min_year = pltdate.year
        
        weather_instance = self._enviroments[env_name]['weathert']
        density = self._enviroments[env_name]['planting_density']
        
        years = range(min_year,np.nanmax(weather_instance.weather.year)-1) if n_years is None else range(
            min_year,min_year+n_years)
        
        for year in years:
            try:
                i_year_sowing_date = '{}-{}'.format(year,pltdate.strftime('%m-%d'))
                self._enviroments[env_name]['planting_date'] = i_year_sowing_date
                sim_results = pd.DataFrame(self.simulate_scenario(env_name))
                if sim_results is None: continue
                sim_results['TRNO'] = treatment_id
                sim_results['TOTBAN'] = (sim_results['Avg_Bioamass_g_mat'] / 1000) * density
                sim_results['fruit_yield'] = (sim_results['Avg_Fruit_g_mat'] / 1000) * density
                sim_results['sowing_date'] = i_year_sowing_date
                best_yield_week = sim_results['TOTBAN'].argmax()
                i_year_sowing_date_dt = datetime.strptime(i_year_sowing_date, "%Y-%m-%d")
                harvesting_date = (i_year_sowing_date_dt + timedelta(days=best_yield_week.tolist()*7)).strftime("%Y-%m-%d")

                reporter.update_report(
                            {'crop': 'banana',
                        'TRNO': treatment_id,               
                        'longitude': weather_instance.station.longitude,
                        'latitude': weather_instance.station.latitude,
                        'altitude': weather_instance.station.altitude,
                        'sowing_date': i_year_sowing_date,
                        'harvesting_date': harvesting_date,
                        'week': sim_results.Week.values[best_yield_week],
                        'biomass': sim_results['TOTBAN'].values[best_yield_week],
                        'fruit_yield': sim_results['fruit_yield'].values[best_yield_week],
                        'Avg_Bioamass_g_mat': sim_results['Avg_Bioamass_g_mat'].values[best_yield_week],
                        'Avg_Fruit_g_mat':sim_results['Avg_Fruit_g_mat'].values[best_yield_week]}
                )
            except:
                pass
        
        reporter.save_reporter(path= env_name, fn = f'output_{treatment_id}.csv')
        reporter.clear_report()
        
    def simulate_planting_dates(self, env_name: str, n_windows:int ):
        """
        Simulate multiple planting dates for a given environment.

        Parameters
        ----------
        env_name : str
            Environment name.
        n_windows : int
            Number of planting dates to simulate.
        **kwargs :
            Additional arguments to pass to `simulate_scenario`.

        Returns
        -------
        pd.DataFrame
            DataFrame with simulation results.
        """
        planting_date = self._enviroments[env_name]['planting_date']
        dates = pd.date_range(start=planting_date, periods=n_windows, freq = '7D')
        
        weather_instance = self._enviroments[env_name]['weathert']
        density = self._enviroments[env_name]['planting_density']
        if not self._check_paths_existance(env_name): return False

        lastdate = weather_instance.weather.values[-1,0]
        reporter = BananaNReporter()

        for i, plt_date in enumerate(dates[:-1]):
            nbweeks = self._enviroments[env_name]['cycle_duration_weeks']
            
            if plt_date + timedelta(weeks=nbweeks)>= lastdate: break
            
            sim_results = pd.DataFrame(self.simulate_scenario(env_name))
            if sim_results is None: continue
            sim_results['TRNO'] = i
            sim_results['TOTBAN'] = (sim_results['Avg_Bioamass_g_mat'] / 1000) * density
            sim_results['fruit_yield'] = (sim_results['Avg_Fruit_g_mat'] / 1000) * density
            sim_results['sowing_date'] = plt_date.strftime("%Y-%m-%d")
            # harvest is when the best yield is achieved (not fixed for each planting date)
            best_yield_week = sim_results['TOTBAN'].argmax()
            
            harvesting_date = (plt_date + timedelta(days=best_yield_week.tolist()*7)).strftime("%Y-%m-%d")
            nrows = sim_results['TOTBAN'].values.shape[0]
            for j in range(nrows):
                reporter.update_report(
                            {'crop': 'banana',
                        'TRNO': i,               
                        'longitude': weather_instance.station.longitude,
                        'latitude': weather_instance.station.latitude,
                        'altitude': weather_instance.station.altitude,
                        'sowing_date': plt_date.strftime("%Y-%m-%d"),
                        'harvesting_date': harvesting_date,
                        'week': sim_results.Week.values[j],
                        'biomass': sim_results['TOTBAN'].values[j],
                        'fruit_yield': sim_results['fruit_yield'].values[j],
                        'Avg_Bioamass_g_mat': sim_results['Avg_Bioamass_g_mat'].values[j],
                        'Avg_Fruit_g_mat':sim_results['Avg_Fruit_g_mat'].values[j]}
                )
            
            reporter.save_reporter(path= env_name, fn = f'output_{i}.csv')
            reporter.clear_report()
        
        return True
    
    def run_scenarios(self, env_name, n_years: int = None, n_cores: int = 0):
        
        total_planting_windows = self._enviroments[env_name].get('total_planting_windows',1)
        if n_cores > 0:
            with tqdm(total=total_planting_windows) as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                    future_to_tr ={executor.submit(self.run_traitment, trn, env_name, n_years): (trn) for trn in range(total_planting_windows)}

                    for future in concurrent.futures.as_completed(future_to_tr):
                        idpx = future_to_tr[future]
                        try:
                            future.result()
                            
                        except Exception as exc:
                                print(f"Request for treatment {idpx} generated an exception: {exc}")
                        pbar.update(1)
                    
        else:
            for trn in tqdm(range(total_planting_windows)):
                self.run_traitment(trn, env_name, n_years)
                
    def run(self, nBan = 40, n_cores: int = 1, weekly_simulation:bool = True) -> None:
        """
        Execute the Banana_N
        """
        
        completed = {}
        if weekly_simulation:

            for env_name in tqdm(self._process_paths):
                if not self.simulate_planting_dates(env_name, n_windows=self._enviroments[env_name]['cycle_duration_weeks'], nban = nBan): 
                    continue
                self._enviroments[env_name]['reporter'] = BananaNOutputData(env_name)
                self._enviroments[env_name]['reporter'].output_data().to_csv(os.path.join(env_name, 'output.csv'), index = False)
                completed[env_name] = True

        else:

            for env_name in self._process_paths:
                print(env_name)
                self.run_scenarios(env_name, n_cores=n_cores)
                self._enviroments[env_name]['reporter'] = BananaNOutputData(env_name)
                pd.DataFrame(self._enviroments[env_name]['reporter'].output_data()).to_csv(os.path.join(env_name, 'output.csv'), index = False)
                completed[env_name] = True
        
        return completed    
            