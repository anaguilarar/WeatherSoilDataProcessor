# Python implementation of the SIMPLE crop model by T Moon, GHPF Lab of SNU.
# Zhao C, Liu B, Xiao L, Hoogenboom G, Boote KJ, Kassie BT, Pavan W, Shelia V, Kim KS, Hernandez-Ochoa IM et al. (2019) A SIMPLE crop model. Eur J Agron 104:97-106

import os
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import numba as nb
import pandas as pd
import xarray
from tqdm import tqdm

from ..utils.model_base import ModelBase
from ..utils.u_soil import find_soil_textural_class
from ..utils.output_transforms import yield_data_summarized

from .crop import Crop
from .weather import Weather, Station
from .soil import ARID
from .output import SimpleModelReporter, SimpleModelOutputData
from .files_export import SMWeather, SMDEM, SMSoil

import matplotlib.pyplot as plt
import copy


def run_treatment( weather_instance, crop, arid, treatment_id: int, treatment_dates, n_years:int = None, co2:int = 381, cycle_days:int = 750, output_path = None):
        """Run a treatment simulation"""
        reporter = SimpleModelReporter()
        date = treatment_dates[treatment_id]
        
        min_year = date.year
        years = range(min_year,np.nanmax(weather_instance.weather.year)-1) if n_years is None else range(
            min_year,min_year+n_years)
        
        for year in years:
            #self._initialize()
            i_year_sowing_date = '{}-{}'.format(year,date.strftime('%m-%d'))
            #start_from = np.searchsorted(weather.weather.DATE.values, pd.to_datetime(i_year_sowing_date))
            
            start_from = np.where(weather_instance.weather.DATE.values == pd.to_datetime(i_year_sowing_date))[0][0].astype(int)
            
            upto = cycle_days+start_from if cycle_days else start_from+cycle_days
            
            if upto>(weather_instance.weather.shape[0]):
                upto = weather_instance.weather.shape[0]
            
            outputs = SimpleModel()._simulate_growing_season(weather=weather_instance.weather.copy().iloc[start_from:upto].reset_index(), arid_values= arid.calculate_range(start_from,upto), crop=crop, co2=co2)
            i_year_harvesting_date = (datetime.strptime(i_year_sowing_date,'%Y-%m-%d') + timedelta(days = outputs['maturity_day'])).strftime('%Y-%m-%d')
            reporter.update_report(
                {'crop': crop.crop,
                                    'TRNO': treatment_id,
                                    'soil_texture': arid.texture,
                                    'longitude': weather_instance.station.longitude,
                                    'latitude': weather_instance.station.latitude,
                                    'altitude': weather_instance.station.altitude,
                                    'co2': co2,
                                    'sowing_date': i_year_sowing_date,
                                    'harvesting_date': i_year_harvesting_date,
                                    'cum_temp': outputs['tt'][-1],
                                    'madurity_day': outputs['maturity_day'],
                                    'biomass': outputs['dbiomass_cum'][-1],
                                    'crop_yield': outputs['crop_yield'][-1]}
            )

        reporter.save_reporter(path= output_path, fn = f'output_{treatment_id}.csv')
        reporter.clear_report()

class SimpleModel():
        
    def __init__(self):
        self.f_solar_max = 0.95
        self.initial_f_solar = 0.001
        self.gdmaturity_day = 0
        self.senescence_day = 0
        self._planting_dates = None

        
    def initialize(self,  arid, weather, crop):
        self.f_solar_max = 0.95
        self.initial_f_solar = 0.001
        self.gdmaturity_day = 0
        self.senescence_day = 0
        
        self.arid = arid
        self.weather = weather
        self.crop = crop
        self._raw_weather_info = self.weather.weather.copy()
        
        self._weather_dates = self.weather.weather.DATE.values
        self._dates = np.array([str(np.datetime_as_string(da, unit= 'D')) for da in self._weather_dates])
        self._planting_dates = None
        """Initialize model state variables with pre-allocated arrays"""

        #self.arid._initialize()
        # Pre-allocate arrays based on maximum expected days
        max_days = 700 

        self.f_solar_water = np.zeros(max_days, dtype=np.float32)
        self.f_solar = np.zeros(max_days, dtype=np.float32)
        
        self.i_50b = np.zeros(max_days, dtype=np.float32)
        self.weather.weather = self._raw_weather_info
        self.arid.weather.weather = self._raw_weather_info
        
        self.arid._initialize()
        

    
    @staticmethod
    @nb.njit(cache=True)
    def _vectorized_calculations(tmin, tmax, t_base, t_opt, t_heat, t_ext, S_CO2 , co2):
        
        t_mean = tmin * 0.5 + tmax * 0.5
        
        tt = (t_mean - t_base).cumsum()
        
        ## temperature stress
        f_temp = np.zeros_like(t_mean)
        mask = (t_mean >= t_base) & (t_mean < t_opt)
        f_temp[mask] = (t_mean[mask] - t_base) / (t_opt - t_base)
        f_temp[t_mean >= t_opt] = 1.0
        ## heat stress
        f_heat = np.maximum(1- (tmax - t_heat) / (t_ext - t_heat), 0)
        f_heat[tmax <= t_heat] = 1.0
        f_heat[tmax > t_ext] = 0.0
        
        # co2 stress
        f_co2 = 1
        if co2 >= 350 and co2 < 700:
            f_co2 = 1 + S_CO2 * (co2 - 350)
        elif co2 >= 700:
            f_co2 = 1 + S_CO2 * 350
        
        return tt, f_temp, f_heat, f_co2
    
    
    @staticmethod
    @nb.njit(cache=True)
    def compute_i50b(f_heat: np.ndarray, 
                    f_water: np.ndarray,
                    i_50b_initial:float, i_50maxh:float, i_50maxw: float) -> np.ndarray:
        """
        Numba-accelerated I_50B calculation.
        
        Parameters
        ----------
        f_heat : np.ndarray
            Array of heat stress factors
        f_water : np.ndarray
            Array of water stress factors
        crop_params : dict
            Dictionary of crop parameters
            
        Returns
        -------
        np.ndarray
            Array of I_50B values
        """
        n_days = len(f_heat)
        i50b = np.zeros(n_days, dtype=np.float32)
        i50b[0] = i_50b_initial
        
        for i in range(1, n_days):
            d1 = i50b[i-1] + i_50maxh * (1 - f_heat[i-1])
            d2 = i50b[i-1] + i_50maxw * (1 - f_water[i-1])
            i50b[i] = max(max(d1, d2), 0)
            
        return i50b

    @staticmethod
    @nb.njit(cache=True)
    def _calculate_fsolar(f_solar_max, tt, f_solar_water, i_50a, t_sum, i_50b, initial_f_solar = 0.001): ## initial_f_solar = 0.001 from the r implementation
        
        n_days = len(tt)
        f_solar = np.zeros(n_days, dtype=np.float32)
        f_solar[0] = initial_f_solar
        senescence_day = 0
        for i in range(1, n_days):
            fsolar1 = min(1, f_solar_max/(1 + np.exp(-0.01*(tt[i] - i_50a))))
            fsolar2 = min(1, f_solar_max/(1 + np.exp( 0.01*(tt[i] - (t_sum - i_50b[i-1])))))
            f_solar[i] = min(fsolar1, fsolar2) * min(f_solar_water[i-1], 1)
            
            if (f_solar[i] < f_solar[i-1]) and f_solar[i] <= initial_f_solar:
                senescence_day = i
                
        return f_solar, senescence_day
    
    
    def _simulate_growing_season(self, weather, arid_values, crop, co2) -> None:
        """Optimized growing season simulation using vectorized operations"""
        # Slice weather data
        #weather_slice = {
        #    k: v[start_idx:end_idx] for k, v in self.weather_data.items()
        #}

        n_days = len(weather['tmin'].values)
        
        gdmaturity_day = n_days - 1
        #print('tbase', crop.T_base, 'topt', crop.T_opt, 'theat', crop.T_heat, 'text', crop.T_ext, 'sco2', crop.S_CO2)
        # Vectorized calculations
        tt, f_temp, f_heat, f_co2 = self._vectorized_calculations(
            weather['tmin'].values,
            weather['tmax'].values,
            crop.T_base, crop.T_opt, crop.T_heat, crop.T_ext, crop.S_CO2,
            co2
        )
        
        
        # Calculate water stress (assuming arid model can be vectorized)
        #arid_values = self.arid.calculate_range(start_idx, end_idx)
        f_water = 1 - crop.S_water * arid_values
        f_water = np.maximum(f_water, 0)

        # solar water factor 
        f_solarwater = np.where(f_water<0.1, 0.9 + f_water, 1 )
        i_50b = self.compute_i50b(f_heat, f_water, crop.I_50B, crop.I_50maxH, crop.I_50maxW)
        
        f_solar, senescence_day = self._calculate_fsolar(self.f_solar_max, tt, f_solarwater, crop.I_50A, crop.T_sum, i_50b)
        
        biomass_rate = (
            weather['srad'].values *
            f_solar *
            crop.RUE *
            f_co2 *
            f_temp *
            np.minimum(f_heat, f_water)
        )
        if len(np.where(tt >= crop.T_sum))>0:
            gdmaturity_day = int(np.where(tt >= crop.T_sum)[0][0])
        
        
            
        senescence_day = senescence_day if senescence_day > (crop.senescence_day or 0) else crop.senescence_day
        maturity_day = gdmaturity_day if gdmaturity_day < senescence_day else senescence_day
        
        dbiomass = biomass_rate[:maturity_day+1]
        dbiomass_cum = dbiomass.cumsum()
        #self._update_results_parameters(tt, f_water, f_heat, f_temp, f_solarwater, arid_values,f_solar, biomass_rate)
        return { 'f_water': f_water[:maturity_day],
        'f_heat': f_heat[:maturity_day],
        'f_temp': f_temp[:maturity_day],
        'f_solarwater': f_solarwater[:maturity_day],
        'f_solar': f_solar[:maturity_day],
        'dbiomass': dbiomass,
        'i_50b': i_50b[:maturity_day],
        'tt': tt[:maturity_day+1],
        'dbiomass_cum': dbiomass_cum,
        'crop_yield': dbiomass_cum * crop.HI,
        'arid_values': arid_values[:maturity_day],
        'maturity_day': maturity_day
        }
        
    @property
    def planting_dates(self):
        if self._planting_dates is None:
            self._planting_dates = []
            for tp in range(self._planting_windows):
                self._planting_dates.append(self._starting_date + timedelta(days=(7*(tp))))

        return self._planting_dates
    
    def set_up_management(self, sowing_date, total_planting_windows):
        """_summary_

        Args:
            sowing_date (str): staring crop simulation date, date format %Y-%m-%d'
            total_planting_window (_type_): _description_
        """
        self._planting_windows = total_planting_windows
        self._starting_date = datetime.strptime(sowing_date,  '%Y-%m-%d')

        
    def _update_results_parameters(self, tt, f_water, f_heat, f_temp, f_solar_water, arid_values, f_solar, biomass_rate):
        """Update model state variables with new results"""
        
        self.maturity_day = self.gdmaturity_day if self.gdmaturity_day < self.senescence_day else self.senescence_day
        
        self.f_water = f_water[:self.maturity_day]
        self.f_heat = f_heat[:self.maturity_day]
        self.f_temp = f_temp[:self.maturity_day]
        self.f_solar_water = f_solar_water[:self.maturity_day]
        self.f_solar = f_solar[:self.maturity_day]
        self.dbiomass = biomass_rate[:self.maturity_day+1]
        self.i_50b = self.i_50b[:self.maturity_day]
        self.tt = tt[:self.maturity_day+1]
        self.dbiomass_cum = self.dbiomass.cumsum()
        self.crop_yield = self.dbiomass_cum * self.crop.HI
        self.arid_values = arid_values[:self.maturity_day]
    
    
    @staticmethod
    def _find_day_indates(dates, date):
        
        return np.where(dates == date)[0][0].astype(int)
    
    def run_scenarios(self, arid, weather, crop, starting_date: str, total_planting_windows = None, cycle_days = 750, co2 = 381, n_years = None, n_cores = 0, output_path = None):  ## Co2 values from simple model authors implementation conditions from south brazil

        output_path = output_path or ""
        self.set_up_management(sowing_date=starting_date, total_planting_windows=total_planting_windows)
        
        if n_cores > 0:
            with tqdm(total=total_planting_windows) as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                    future_to_tr ={executor.submit(run_treatment, copy.deepcopy(weather), copy.deepcopy(crop), copy.deepcopy(arid), trn, self.planting_dates, n_years, co2, cycle_days, output_path): (trn) for trn in range(total_planting_windows)}

                    for future in concurrent.futures.as_completed(future_to_tr):
                        idpx = future_to_tr[future]
                        try:
                            future.result()
                            
                        except Exception as exc:
                                print(f"Request for treatment {idpx} generated an exception: {exc}")
                        pbar.update(1)
                    
        else:
            for trn in tqdm(range(total_planting_windows)):
                run_treatment(copy.deepcopy(weather), copy.deepcopy(crop), copy.deepcopy(arid), trn, self.planting_dates, n_years, co2, cycle_days, output_path)
        
    
    

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
    soil_params = pd.read_csv(soil_path)
    soil_texture = find_soil_textural_class(soil_params.sand.values[0], soil_params.clay.values[0])
    location_params = pd.read_csv(dem_path)

    station = Station(
        latitude=float(location_params['LAT'].values[0]),
        longitude=float(location_params['LONG'].values[0]),
        altitude=int(location_params['ELEV'].values[0]),
    )
    
    weather_data = Weather(path = weather_path, station = station)
    weather_data.weather.DATE = pd.to_datetime(weather_data.weather.DATE , format="%Y%m%d")
    arid = ARID(weather_data)
    arid.soil_params(soil_texture.lower())

    
    return  weather_data, arid


class PySimpleModel(ModelBase):
    """
    A crop biomass simulation model based on the SIMPLE crop model framework.

    Parameters
    ----------
    arid : ARID
        ARID model instance for water stress calculations
    weather : Weather
        Weather data container
    crop : Crop
        Crop parameters and characteristics

    Attributes
    ----------
    reporter : SimpleModelReporter
        Data collection object for simulation results
    biomass : List[float]
        Daily biomass accumulation (kg/ha)
    yields : List[float]
        Daily yield values (kg/ha)

    References
    ----------
    Zhao C, Liu B, Xiao L, et al. (2019) A SIMPLE crop model. 
    European Journal of Agronomy 104:97-106. 
    https://doi.org/10.1016/j.eja.2019.01.009

    Notes
    -----
    Modified from: 
    https://github.com/breuderink/green-fingers/blob/main/crop_model.py
    """
    
    def __init__(self, path):
        super().__init__(path)
        self._enviroments = {}

    def set_up_enviroment(self, env_name = None, **kwargs):
        if env_name is None:
            env_name = 'env_{}'.format(len(self._enviroments)+1)
        
        if self._enviroments.get(env_name) is None:
            self._enviroments[env_name] = {k:v for k, v in kwargs.items()}
        else:
            self._enviroments[env_name].update({k:v for k, v in kwargs.items()})
        
    def reset_enviroments(self):
        self._enviroments = {}

    def organize_enviromental_data(self, verbose = True) -> None:
        """
        Organize the environment for the simulation.

        Parameters
        ----------
        crop_name : str
            Crop name            
        planting_date : str
            Initial sowing date in 'YYYY-MM-DD' format
        n_week_windows : int
            Number of planting dates to simulate
        co2 : float, optional
            CO2 concentration (ppm), default 381
        arid : ARID
            ARID model instance for water stress calculations
        weather : Weather
            Weather data container

        """
        if len(self._process_paths) == 0: self.find_envworking_paths(file_ext='csv')
        
        for pathiprocess in self._process_paths:
            dem_path = os.path.join(pathiprocess, 'smdem.csv')
            soil_path = os.path.join(pathiprocess, 'smsoil.csv')
            weather_path = os.path.join(pathiprocess, 'smweather.csv')
            
            weather, arid = set_up_weather_soil_parameters(soil_path =soil_path, dem_path = dem_path, weather_path = weather_path)
            self.set_up_enviroment(env_name = pathiprocess,
                                   arid = arid, weather = weather)
            if verbose: print("weather and soil data organized for {}".format(pathiprocess))
    
    def set_up_crop(self, crop_name,  starting_date, n_week_windows, co2 = 381, crop_params_path = None, cultivar_path = None, verbose = True):
        
        if len(self._process_paths) == 0: self.find_envworking_paths(file_ext='csv')
        
        for pathiprocess in self._process_paths:
            crop = Crop(crop = crop_name, crop_params_path=crop_params_path, cultivar_path=cultivar_path)
            crop.set_params()
            self.set_up_enviroment(env_name = pathiprocess,
                                crop = crop, starting_date = starting_date, n_week_window = n_week_windows, co2 = co2)
            if verbose: print("crop data set up for {}".format(pathiprocess))

    def run(self, n_years = None, n_cores = 0):
        """
        Execute the crop model simulation.

        Parameters
        ----------


        n_years : int, optional
            Number of years to simulate, default None

        Returns
        -------
        pd.DataFrame
            Simulation results containing biomass and yield data
        """
        completed = {}
        for env_name, env in self._enviroments.items():
            print(env_name)
            
            model = SimpleModel()
            model.run_scenarios(env['arid'], env['weather'], env['crop'], starting_date = env['starting_date'], total_planting_windows = env['n_week_window'], cycle_days = 750, 
                                co2 = env['co2'], n_years = n_years, n_cores = n_cores, output_path = env_name)
            
            self._enviroments[env_name]['reporter'] = SimpleModelOutputData(env_name)

            pd.DataFrame(self._enviroments[env_name]['reporter'].output_data()).to_csv(os.path.join(env_name, 'output.csv'), index = False)
            completed[env_name] = True
        return completed
        
    def plot_results(self, target_column = 'biomass', date_column = 'sowing_date') -> plt.Figure:
        """
        plot the results of the site specific simluations results
        
        Parameters:
        -----------
        target_column : str, optional
            The target column to plot, there are two options biomass or crop_yield, by default 'biomass'
        date_column : str, optional
            The date column to plot, there are two options sowing_date or harvesting_date, by default 'sowing_date'
        
        Returns:
        --------
        plt.Figure
        """
        if self._enviroments is None:   return None
        titles = {'biomass':'Biomass (kg/ha)', 'crop_yield':'Yield (kg/ha)'}
        f, ax = plt.subplots(ncols=1, nrows=len(self._enviroments), figsize=(25,(8*len(self._enviroments))))
        for i, (env_name, env) in enumerate(self._enviroments.items()):
            
            yield_data = yield_data_summarized(pd.DataFrame(env['reporter'].output_data()),'trno', date_column=date_column,yield_column= target_column, harvest_column='harvesting_date')

            ax[i].plot(yield_data.sowing_date_year_month_day.values, yield_data[target_column].values, c = 'r')
            ax[i].plot(yield_data.sowing_date_year_month_day.values, yield_data.y_lower.values, c ='gray')
            ax[i].plot(yield_data.sowing_date_year_month_day.values, yield_data.y_upper.values, c ='gray')
            ax[i].fill_between(yield_data.sowing_date_year_month_day.values, yield_data.y_upper.values, yield_data.y_lower.values, color="k", alpha=0.15)
            ax[i].set_title(env_name)
            ax[i].set_xlabel('Date')
            if target_column in titles.keys():
                ax[i].set_ylabel(titles[target_column])
            
            ax[i].grid()
            
        return f, ax

    
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
            weatherprocessor = SMWeather(xrdata)
            weatherprocessor(depth_var_name = 'date', group_by= group_by, outputpath = outputpath, codes=group_codes, target_crs=target_crs)
            
        if data_source == 'dem':
            demprocessor = SMDEM(xrdata)
            demprocessor(group_by= group_by, outputpath = outputpath, codes=group_codes, target_crs=target_crs)

        if data_source == 'soil':        
            soilprocessor = SMSoil(xrdata)
            soilprocessor(depth_var_name = 'depth', group_by= group_by, outputpath = outputpath, codes=group_codes, target_crs=target_crs)
