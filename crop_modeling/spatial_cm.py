import os
import shutil
import warnings
from typing import List

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from .utils.process import model_selection
from .caf.management import fertiOrganizer
from .utils.output_transforms import ColumnNames, update_data_using_path
from .spatial_process import SpatialCM

def caf_coffeeplant_productioncycle(production_years, management_configuration):
    ncycles = management_configuration['n_cycles']
    default_years = management_configuration['cycle_treatment_1']['life_cycle_years']
    if production_years<3:
        warnings.warn(f'The production years must more than 3 years, model will use {default_years}')
        return management_configuration

    if production_years>12:
        warnings.warn(f'Currently the model only support a simulation up to 12 years, model will use {default_years}')
        return management_configuration

    for ncycle in range(1, ncycles+1):
        management_configuration[f'cycle_treatment_{ncycle}']['life_cycle_years'] = production_years

    return management_configuration   


class SpatialCAF(SpatialCM, fertiOrganizer):

    def __init__(self, planting_date, **kwargs):
        SpatialCM.__init__(self,**kwargs)
        self._colnames = ColumnNames(self.model.name)
        fertiOrganizer.__init__(self,planting_date)


    def ferti_days_after_flowering(self, flowering_dates: pd.DataFrame,
                                   days_of_application: List[float], n_amounts: List[float], nitrogen_factor = 1):
        fertilization_dict = {}
        for i in range(len(self.pdates)):
            fl_df= flowering_dates.loc[flowering_dates.n_cycle == i+1]
            fl_days = (fl_df[self._colnames.growth_colnames['date']] - self.pdates[i]).dt.days.values
            total_fldays = len(fl_days)

            updfertiday = np.zeros(len(fl_days)*len(days_of_application), dtype= float)
            updamount = np.zeros(len(fl_days)*len(days_of_application), dtype= float)

            for z, (day, amount) in enumerate(zip(days_of_application, n_amounts)):
                ferti_day = np.array(fl_days) + day
                updfertiday[total_fldays*z:(total_fldays*(z+1))] = ferti_day.tolist()
                updamount[total_fldays*z:(total_fldays*(z+1))] = [amount*nitrogen_factor]*total_fldays

            fertilization_dict[i+1] = [updfertiday.tolist(), updamount.tolist()]

        return fertilization_dict

    def get_flowering_dates(self, duration=None, remove_tmp_folders = True, verbose = True, run_model = True):

        date_colname, fd_colname = self._colnames.growth_colnames['date'], self._colnames.growth_colnames['flowering_date']
        n_cycle, crop_yield = self._colnames.growth_colnames['number_of_cycle'], self._colnames.growth_colnames['yield']
        h_date = self._colnames.growth_colnames['hdate']
        if run_model:
            self.config.MANAGEMENT.fertilization = self.fertilization_schedule([1,2],[0,0])
            if duration is not None:
                self.config.MANAGEMENT = caf_coffeeplant_productioncycle(int(duration), self.config.MANAGEMENT)

            completed_sims = self.run_caf(verbose = verbose)
        else: 
            completed_sims = None
        print(completed_sims, self._tmp_path)
        model_data = update_data_using_path(self._tmp_path, model = self.model.name)
        completed_sims = completed_sims if completed_sims is not None else {k:True for k, v in model_data.items()}

        flowering_dates = {}
        baseline_yields = {}
        
        for k, v in model_data.items():
            flowering_dates_df = v.output_data().sort_values(date_colname)[[n_cycle, fd_colname, date_colname, crop_yield]]
            harvested_yield_df = v.output_data().sort_values(date_colname)[[n_cycle, h_date, date_colname, crop_yield]]
            
            flowering_dates[k] = flowering_dates_df.loc[flowering_dates_df[fd_colname] != 0]
            baseline_yields[k] = harvested_yield_df.loc[harvested_yield_df[h_date] != 0]
            
            
        if remove_tmp_folders:
            for pathfolder in self.model._process_paths:
                for i in range(np.unique(flowering_dates[k][n_cycle].values).shape[0]):
                    shutil.rmtree(os.path.join(pathfolder, f'_{i}'), ignore_errors=False, onerror=None)
                    os.remove(os.path.join(pathfolder, f'output_{i}.csv'))

        return flowering_dates, baseline_yields
    
    def upload_crop_model(self):
        self.model = model_selection(self.config.GENERAL_INFO.get('model', None), self.config.GENERAL_INFO.get('working_path', 'tmp'))
        print('--> CAF model was updaded oupputs will be located in: ', self.model.path)
        
    def run_caf(self, setup_envs = True, verbose = True):
        
        if setup_envs: self.model.find_envworking_paths(self._tmp_path, 'csv')
        
        self.model.set_parameters(file_path = self.config.CROP.parameters)
        params = OmegaConf.to_container(self.config).get('PARAMETERS',None)
        if params is not None:
            self.model.modify_plant_parameters(params, verbose = verbose)
        
        for n_cycle in range(0,self.config.MANAGEMENT.n_cycles):
            nplus1 = n_cycle+1
            cycle_management = OmegaConf.to_container(self.config.MANAGEMENT).get(f'cycle_treatment_{nplus1}',None)
            assert cycle_management, 'configurate management per life cycle'
        
            fert, coffee_prun, tree_prun, tree_thinning = self.model.set_up_management(cycle_management)
            self.model.set_tree_parameters(verbose = verbose, **self.config.MANAGEMENT.TREE)
            self.model.organize_env(
                n_cycle =n_cycle,
                verbose = verbose,
                planting_date=cycle_management['planting_date'], life_cycle_years = cycle_management['life_cycle_years'],
                fert=fert, coffee_prun=coffee_prun, tree_prun=tree_prun, tree_thinning=tree_thinning, dll_path = self.config.GENERAL_INFO.bin_path)
        
        # run the simulation
        return self.model.run(n_cycles = self.config.MANAGEMENT.n_cycles)