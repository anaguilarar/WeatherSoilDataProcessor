
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import shutil
import concurrent.futures
from tqdm import tqdm
from omegaconf import OmegaConf
import yaml

from spatialdata.soil_data import TEXTURE_CLASSES
from ..dssat.output import DSSATOutputData
from ..spatial_process import SpatialCM
from .model_base import ReporterBase
from .output_transforms import yield_data_summarized, ColumnNames
from .process import model_selection

def fertilization_simulations(model, configuration, application_day, n_value, rm_simulation_folder = False, verbose = False, plantingWindow = 1, element_tooptimize = 'n'):
    configuration.MANAGEMENT.plantingWindow = plantingWindow
    if len(application_day) == 0:
        configuration.MANAGEMENT.fertilizer_schedule = None
        sim_experiment_path = os.path.join(model._process_paths[0],
                                    's_not_fertilizer')
        
    else:
        if element_tooptimize == 'n': 
            npk_schedule = [[i] for i in n_value]
        elif element_tooptimize == 'p':
            npk_schedule = [[0,i,0] for i in n_value]
        elif element_tooptimize == 'k':
            npk_schedule = [[0,0,i] for i in n_value]
            
        configuration.MANAGEMENT.fertilizer_schedule = {'days_after_planting': application_day, 'npk': npk_schedule}
        
        sim_experiment_path = os.path.join(model._process_paths[0],
                                    's'+'_'.join([str(i) for i in application_day]) +'_'+ '_'.join([str(i) for i in n_value]))
        
    model.set_up_crop(crop=configuration.CROP.name, cultivar=configuration.CROP.cultivar)
    model.set_up_management(crop=configuration.CROP.name, cultivar=configuration.CROP.cultivar, verbose = verbose, **configuration.MANAGEMENT)

    model.run(model.crop_code, crop=configuration.CROP.name ,planting_window=plantingWindow,
                            bin_path = configuration.GENERAL_INFO.bin_path, 
                            dssat_path = configuration.GENERAL_INFO.get('dssat_path', None), remove_tmp_folder=rm_simulation_folder, 
                            sim_experiment_path= sim_experiment_path, verbose = verbose)

def check_element_to_optimize(**kwargs):
    
    n_optimization = any(['n' in i for i in kwargs.keys()])
    if n_optimization: return 'n'
    p_optimization = any(['p' in i for i in kwargs.keys()])
    if p_optimization: return 'p'
    k_optimization = any(['k' in i for i in kwargs.keys()])
    if k_optimization: return 'k'
    
    return None

def yield_simulation(model, configuration, application_day, n_value, rm_simulation_folder = False, element_tooptimize = 'n'):
    
        colnames = ColumnNames(model.name)
        date_column = colnames.growth_colnames['date']
        target_column = colnames.growth_colnames['yield']
        harvest_column = colnames.growth_colnames['hdate']
        nitrogen_uptake = colnames._nitrogen_uptake['dssat']['nitrogen_uptake']

        fertilization_simulations(model, configuration, application_day, n_value, rm_simulation_folder=rm_simulation_folder, element_tooptimize = element_tooptimize)
        
        #model_data = update_data_using_path(self._tmp_path, model = self.model.name)
        if model.name == 'dssat':
            model_data = DSSATOutputData(model._process_paths[0])
            model_data = pd.DataFrame(model_data.output_data())
            
        yield_data = yield_data_summarized(model_data,'TRNO', date_column=date_column,yield_column= target_column, 
                                        harvest_column=harvest_column)
        
        n_uptake = yield_data_summarized(model_data,'TRNO', date_column=date_column,yield_column= nitrogen_uptake, 
                                        harvest_column=harvest_column)
        
        env_path = os.path.join(model._process_paths[0])
        filesinenv = os.listdir(env_path)

        for fn in filesinenv:
            if (fn not in ['TR.SOL', 'WTHE0001.WTH', 'crop_configuration.yaml']) and (not os.path.isdir(os.path.join(env_path, fn))):            
                os.remove(os.path.join(env_path, fn))

        return yield_data[target_column].values[0], n_uptake[nitrogen_uptake].values[0]

def yield_predictions_withapplications(min_split_interval = 1, rm_simulation_folder = True, working_path = None, **kwargs):
    
    element_tooptimize = check_element_to_optimize(**kwargs)
    
    assert element_tooptimize is not None, 'Only n p k are available to optimize'
    configuration = OmegaConf.load(os.path.join(working_path, 'crop_configuration.yaml'))
    model_name = configuration.GENERAL_INFO.get('model', None)
    model = model_selection(model_name, working_path)
    model._process_paths = [working_path]
    
    day_values = []
    napp_values = []
    n = 1
    ## get 
    while True:
        day_value = kwargs.get(f"day{n}", None)
        napp_value = kwargs.get(f"{element_tooptimize}{n}", None)
        n+=1
        if day_value:
            day_values.append(int(round(day_value)))
            napp_values.append(int(round((napp_value))))
        else:
            break
    
    total_n = np.sum(napp_values)

    if total_n < 0:
        print(f"Warning: Total {element_tooptimize} applied is zero or negative.")
        return 0
    if len(day_values)>1:
        for i in range(len(day_values)-1):
            # Penalize invalid combinations by returning a very low NUE
            if (day_values[i+1] <= day_values[i] + min_split_interval): return -1e6
    
    current_yield, n_uptake = yield_simulation(model, configuration, day_values, napp_values, 
                                rm_simulation_folder = rm_simulation_folder, element_tooptimize = element_tooptimize)
    
    if total_n == 0:
        nue = -99
    else:
        nue = n_uptake/total_n
    #print(f"Params: day1={day1_int}, n1={n1_int}, day2={day2_int}, n2={n2_int}, day3={day3_int}, n3={n3_int} -> Yield={current_yield:.0f}, Total N={total_n:.0f}, NUE={nue:.2f}")
    # reporter
    if len(day_values)>0:
        daysdict = {f'day_{i+1}': int(d) for i, d in enumerate(day_values)}
        ndict = {f'n_{i+1}': int(n) for i, n in enumerate(napp_values)}
        daysdict.update(ndict)
        
    else:
        daysdict = {'day_1': 0, f'{element_tooptimize}1': 0}
    
    
    daysdict.update({'nue': nue,
                f'total_{element_tooptimize}': total_n,
                'crop_yield': current_yield})
    

    return current_yield
    
class FertilizerBayesian(SpatialCM):
    """
    Nitrogen application optimization using Bayesian methods.

    This class simulates different nitrogen fertilizer application schedules across planting dates,
    optimizing them using Bayesian Optimization.

    Parameters
    ----------
    configuration_dict : dict, optional
        Dictionary containing configuration parameters.
    configuration_path : str, optional
        Path to configuration file.
    """

    def __init__(self, configuration_dict:Dict=None, configuration_path:str = None):
        
        SpatialCM.__init__(self, configuration_dict=configuration_dict, configuration_path=configuration_path)
        
        self.reporter = ReporterBase()
        
        self._application_schedule = {
                'day1': (1, 15),      # Day of first application (e.g., days after planting)
                'n1': (0, 200),      # Amount N1 (kg/ha). Set min N > 0 to avoid division issues & make sense agronomically.
                'day2': (15, 45),     # Day of second application. Overlaps with day1 range, so constraint is needed.
                'n2': (0, 200),       # Amount N2 (kg/ha). Min N > 0.
                'day3': (40, 70),     # Day of second application. Overlaps with day3 range, so constraint is needed.
                'n3': (0, 100),# Constraint: total N <= max_total_n could be added inside nue_target_multi if needed
            }
        
    def set_application_schedule(self, application_schedule: Optional[Dict] = None) -> None:
        """Set the fertilizer application schedule."""
        self.application_schedule = application_schedule or self._application_schedule
        self.clear_reporter()

    def _get_report_keys(self) -> list:
        """Get keys for reporting results.

        Returns
        -------
        list
            List of keys to report including application days, amounts, and yield.
        """
        n_applications = len(list(self.application_schedule.keys()))//2
        ndays = [f'day{i+1}' for i in range(n_applications)]
        nnapps = [f'n{i+1}' for i in range(n_applications)]
        return ndays + nnapps + ['crop_yield']
    
    def clear_reporter(self):
        reporter_keys = self._get_report_keys()
        self.reporter.set_reporter(reporter_keys)
        
    def set_enviromental_information(self, geocode: str) -> None:
        """Set up environmental information for a specific geographic site.

        Parameters
        ----------
        geocode : str
            Site code to retrieve corresponding spatial data.
        """
        roi = self.geo_features.loc[self.geo_features['GEOCODIGO']==str(geocode)]
        
        self.set_up_folders(site = geocode)

        self.create_roi_sp_data(
        roi=roi,
        group_codes=TEXTURE_CLASSES,  # Codes for grouping data by texture
        create_group_splayer=False,
        export_spatial_data=True
        )
        
        self._cm_envs = self.model._process_paths
    
    def create_configuration_file(self, working_dir: str, pdate: str, hdate: str, tp: int) -> str:
        """Create configuration file for a specific time point.

        Parameters
        ----------
        working_dir : str
            Directory where working files are stored.
        pdate : str
            Planting date.
        hdate : str
            Harvesting date.
        tp : int
            Time point.

        Returns
        -------
        str
            Path to the working temporary directory.
        """
        crop_configuration = OmegaConf.to_container(self.config)
        crop_configuration['MANAGEMENT']['planting_date'] = pdate
        crop_configuration['MANAGEMENT']['harvesting_date'] = hdate
        
        working_tmpdir = os.path.join(working_dir, f'tp_{tp}')
        if not os.path.exists(working_tmpdir): os.mkdir(working_tmpdir)
        shutil.copy2(os.path.join(working_dir,
                                'TR.SOL'), 
                    working_tmpdir)
        shutil.copy2(os.path.join(working_dir,
                                'WTHE0001.WTH'), 
                    working_tmpdir)

        crop_configuration['GENERAL_INFO']['working_tmp_dir'] = working_tmpdir
        
        ## export configuration
        with open(os.path.join(working_tmpdir, 'crop_configuration.yaml'), 'w') as file:
            yaml.dump(crop_configuration, file)
        
        return working_tmpdir

    def calculate_yield_pertp(self, tp: int, **kwargs) -> float:
        """Calculate crop yield for a specific time point.

        Parameters
        ----------
        tp : int
            Time point.

        Returns
        -------
        float
            Predicted yield.
        """
        working_path = os.path.join(self.model._process_paths[0],
                    f'tp_{int(tp)}')
        
        return yield_predictions_withapplications(working_path = working_path, **kwargs)
    
    def organize_results(self, optresults: list) -> pd.DataFrame:
        """Organize optimizer results into a DataFrame.

        Parameters
        ----------
        optresults : list
            Results from Bayesian optimizer.

        Returns
        -------
        pd.DataFrame
            DataFrame with input parameters and yield results.
        """
        result_values = []
        reporter_keys = self._get_report_keys()
        for i in range(len(optresults)):
            reportvalues = {k: int(v) for k, v in optresults[i]['params'].items() if k in reporter_keys}
            reportvalues.update({reporter_keys[-1]:optresults[i]['target']})
            result_values.append(reportvalues)
        
        return pd.DataFrame(result_values)
    
    def simulate_time_point(self, tp: int, starting_date: str, ending_date: str, n_iter: int = 10,
                            date_format: str = '%Y-%m-%d', working_dir: Optional[str] = None,
                            init_points: int = 10, removeworking_path_folder: bool = True,
                            verbose: int = 0) -> pd.DataFrame:
        """Run optimization for a specific time point.

        Parameters
        ----------
        tp : int
            Time point index.
        starting_date : str
            Start planting date.
        ending_date : str
            Ending harvest date.
        n_iter : int
            Number of iterations for optimization.
        date_format : str
            Format of date strings.
        working_dir : str, optional
            Directory where files will be saved.
        init_points : int
            Initial points for Bayesian optimization.
        removeworking_path_folder : bool
            Whether to delete temporary working directory.
        verbose : int
            Verbosity level.

        Returns
        -------
        pd.DataFrame
            DataFrame with simulation results.
        """
        working_dir = working_dir or  self.model._process_paths[0]
        new_pdate = datetime.strptime(starting_date,date_format) + timedelta(days=(7*(tp)))
        new_hdate = datetime.strptime(ending_date,date_format) + timedelta(days=(7*(tp)))
        n_applications = len(list(self.application_schedule.keys()))//2
        
        working_tmpdir = self.create_configuration_file(working_dir=working_dir,tp = tp, 
                                pdate=new_pdate.strftime(date_format), hdate=new_hdate.strftime(date_format))
        ## I couldn't find a best way to pass through the function which time point should be focused in
        application_schedule = self.application_schedule.copy()
        application_schedule.update({'tp': (tp, tp+.0001)})
        
        optimizer = BayesianOptimization(
            f=self.calculate_yield_pertp,
            pbounds=application_schedule,
            random_state=123,
            verbose=verbose 
        )
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        
        fn = os.path.join(self._tmp_path, '{}_{}_{}.csv'.format(
            os.path.basename(working_tmpdir),
            os.path.basename(working_dir), n_applications))
        
        if removeworking_path_folder:
            shutil.rmtree(working_tmpdir, ignore_errors=False)
        
        df_results = self.organize_results(optimizer.res)
        df_results.to_csv(fn)
        return df_results

    def simulate_multipletimes(self, env: int, n_iter: int = 10, time_windows: int = 1,
                            start_from: Optional[str] = None, harvesting_date: Optional[str] = None,
                            date_format: str = '%Y-%m-%d', ncores: int = 0, verbose: int = 0) -> None:
        """Run optimization simulations over multiple time points.

        Parameters
        ----------
        env : int
            Environment index.
        n_iter : int
            Number of iterations per time point.
        time_windows : int
            Number of time points to simulate.
        start_from : str, optional
            Planting start date.
        harvesting_date : str, optional
            Final harvesting date.
        date_format : str
            Date format used.
        ncores : int
            Number of threads to use. If 0, runs sequentially.
        verbose : int
            Verbosity level.
        """
        start_from = start_from or self.config.MANAGEMENT.planting_date
        harvesting_date = harvesting_date or self.config.MANAGEMENT.harvesting_date
        rangedata = tqdm(range(time_windows)) if verbose else range(time_windows)
        self.model._process_paths = [self._cm_envs[env]]
        
        if ncores != 0: 
            with tqdm(total=time_windows) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
                    future_to_tr ={executor.submit(self.simulate_time_point, tp, start_from, harvesting_date, n_iter = n_iter,
                                                date_format = date_format): (tp) for tp in range(time_windows)}
                    for future in concurrent.futures.as_completed(future_to_tr):
                        future.result()
                        pbar.update(1)
        else:
            for tp in tqdm(range(time_windows)):
                self.simulate_time_point(tp, start_from, harvesting_date, n_iter = n_iter, date_format=date_format)
            