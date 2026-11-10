
import os
import shutil
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import warnings


import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ..spatial_process import SpatialCM
from .output_transforms import CAFOutputData, ColumnNames, update_data_using_path
from ..caf.management import prunningOrganizer, fertiOrganizer

from spatialdata.soil_data import TEXTURE_CLASSES

    

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

def run_caf(spatial_cm):
    
    
    spatial_cm.model.set_parameters(file_path = spatial_cm.config.CROP.parameters)
    
    params = OmegaConf.to_container(spatial_cm.config).get('PARAMETERS',None)
    if params is not None:
        spatial_cm.model.modify_plant_parameters(params, verbose = False)
            
    for n_cycle in range(0,spatial_cm.config.MANAGEMENT.n_cycles):
        nplus1 = n_cycle+1
        cycle_management = OmegaConf.to_container(spatial_cm.config.MANAGEMENT).get(f'cycle_treatment_{nplus1}',None)
        
        assert cycle_management, 'configurate management per life cycle'
                   
        fert, coffee_prun, tree_prun, tree_thinning = spatial_cm.model.set_up_management(cycle_management)
        spatial_cm.model.set_tree_parameters( **cycle_management['TREE'], verbose = False)
        spatial_cm.model.organize_env(
            n_cycle =n_cycle,
            verbose = False,
            planting_date=cycle_management['planting_date'], life_cycle_years = cycle_management['life_cycle_years'],
            fert=fert, coffee_prun=coffee_prun, tree_prun=tree_prun, tree_thinning=tree_thinning, dll_path = spatial_cm.config.GENERAL_INFO.bin_path)
    
    # run the simulation
    return spatial_cm.model.run(n_cycles = spatial_cm.config.MANAGEMENT.n_cycles)


class SpatialCAF(fertiOrganizer):

    def __init__(self, spatial_processor, planting_date, **kwargs):

        self.spatial_processor = spatial_processor
        self._colnames = ColumnNames(spatial_processor.model.name)

        super().__init__(planting_date)


    def ferti_days_after_flowering(self, flowering_dates: pd.DataFrame, days_of_application: List[float], n_amounts: List[float], nitrogen_factor = 1):
        fertilization_dict = {}
        for i in range(len(self.pdates)):
            fl_df= flowering_dates.loc[flowering_dates.n_cycle == i+1]
            fl_days = (fl_df[self._colnames.growth_colnames['date']] - self.pdates[i]).dt.days.values
            total_fldays = len(fl_days)

            updfertiday = np.zeros(len(fl_days)*len(days_of_application), dtype= float)
            updamount = np.zeros(len(fl_days)*len(days_of_application), dtype= float)

            for z, (day, amount) in enumerate(zip(days_of_application, n_amounts)):
                ferti_day = np.array(fl_days) + (day*30)
                updfertiday[total_fldays*z:(total_fldays*(z+1))] = ferti_day.tolist()
                updamount[total_fldays*z:(total_fldays*(z+1))] = [amount*nitrogen_factor]*total_fldays

            fertilization_dict[i+1] = [updfertiday.tolist(), updamount.tolist()]

        return fertilization_dict


    def get_flowering_dates(self, duration=None, remove_tmp_folders = True):


        date_colname, fd_colname = self._colnames.growth_colnames['date'], self._colnames.growth_colnames['flowering_date']
        n_cycle, crop_yield = self._colnames.growth_colnames['number_of_cycle'], self._colnames.growth_colnames['yield']
        h_date = self._colnames.growth_colnames['hdate']

        self.spatial_processor.config.MANAGEMENT.fertilization = self.fertilization_schedule([1,2],[0,0])
        if duration is not None:
            self.spatial_processor.config.MANAGEMENT = caf_coffeeplant_productioncycle(int(duration), self.spatial_processor.config.MANAGEMENT)

        completed_sims = run_caf(self.spatial_processor)
        model_data = update_data_using_path(self.spatial_processor._tmp_path, model = self.spatial_processor.model.name)
        completed_sims = completed_sims if completed_sims is not None else {k:True for k, v in model_data.items()}

        flowering_dates = {}
        baseline_yields = {}
        
        for k, v in model_data.items():
            flowering_dates_df = v.output_data().sort_values(date_colname)[[n_cycle, fd_colname, date_colname, crop_yield]]
            harvested_yield_df = v.output_data().sort_values(date_colname)[[n_cycle, h_date, date_colname, crop_yield]]
            
            flowering_dates[k] = flowering_dates_df.loc[flowering_dates_df[fd_colname] != 0]
            baseline_yields[k] = harvested_yield_df.loc[harvested_yield_df[h_date] != 0]
            
        if remove_tmp_folders:
            for pathfolder in self.spatial_processor.model._process_paths:
                for i in range(np.unique(flowering_dates[k][n_cycle].values).shape[0]):
                    #shutil.rmtree(os.path.join(pathfolder, f'_{i}'), ignore_errors=False, onerror=None)
                    os.remove(os.path.join(pathfolder, f'output_{i}.csv'))

        return flowering_dates, baseline_yields


NITROGEN_MODEL_MULTIPLIER = 10

def get_applications_from_kwwargs(**kwargs):
    day_values = []
    napp_values = []
    n = 0
    ## get 
    while True:
        day_value = kwargs.get(f"day{n}", None)
        napp_value = kwargs.get(f"n{n}", None)
        n+=1
        if day_value:
            day_values.append(int(round(day_value)))
            napp_values.append(int(round((napp_value))))
        else:
            break
    
    return day_values, napp_values

def set_up_sim_enviroment(src_path, target_path):
    
    wth_path = os.path.join(src_path, 'cafdem.csv')
    sl_path = os.path.join(src_path, 'cafsoil.csv')
    assert os.path.exists(wth_path) and os.path.exists(sl_path), 'create caf files first'
    shutil.copy2(wth_path, target_path)
    shutil.copy2(sl_path, target_path)
    shutil.copy2(os.path.join(src_path, 'cafweather.csv'), target_path)
    


def fertilization_simulations(spatial_processor, application_day, n_value, nitrogen_factor = 1):
    
    planting_dates = [spatial_processor.config.MANAGEMENT[f'cycle_treatment_{i+1}']['planting_date'] 
        for i in range(spatial_processor.config.MANAGEMENT.n_cycles)]
    
    
    caf_model = SpatialCAF(spatial_processor, planting_dates)
    
    applicationfirst_years = n_value[0]
    application_monthfirst_years = application_day[0]
    
    if len(application_day) == 0:
        spatial_processor.config.MANAGEMENT.fertilization = caf_model.fertilization_schedule([1,2],[0,0])
        sim_experiment_path = os.path.join(spatial_processor.model._process_paths[0],
                                    's_not_fertilizer')
        
    else:
        flowering_dates = pd.DataFrame.from_dict(spatial_processor.config.flowering_days, orient='index')
        
        flowering_dates['HDAT'] = pd.to_datetime(flowering_dates['HDAT'])
        
        ferti_days = caf_model.ferti_days_after_flowering(flowering_dates, application_day[1:], n_value[1:], nitrogen_factor = nitrogen_factor)
        
        for z in range(spatial_processor.config.MANAGEMENT.n_cycles):
            ferti_cycle_schedule = caf_model.create_event_cycle_fert_schedule(z, *ferti_days[z+1])         
            spatial_processor.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization'] = ferti_cycle_schedule
            
        
        if applicationfirst_years >0:
            
            for z in range(spatial_processor.config.MANAGEMENT.n_cycles):
                cycle_event = {
                    'years': spatial_processor.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization']['years'],
                    'dayofyear': spatial_processor.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization']['dayofyear'],
                    'N_amount': spatial_processor.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization']['N_amount']
                }
                
                for m in range(2):
                    
                    cycle_event['years'].append(caf_model.pdates[z].year+m)
                    cycle_event['dayofyear'].append(application_monthfirst_years*30)
                    cycle_event['N_amount'].append(applicationfirst_years*nitrogen_factor)
                cycle_event['n_fertilization_per_year'] = 1
                
                spatial_processor.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization'] = cycle_event
                
                    
        sim_experiment_path = os.path.join(spatial_processor.model._process_paths[0],
                                    's'+'_'.join([str(i) for i in application_day]) +'_'+ '_'.join([str(i) for i in n_value]))
        if not os.path.exists(sim_experiment_path): os.mkdir(sim_experiment_path)

    
    set_up_sim_enviroment(spatial_processor.model._process_paths[0], sim_experiment_path)
    
    spatial_processor.model._process_paths = [sim_experiment_path]
    
    completed_sims = run_caf(spatial_processor)
    
    return completed_sims


def yield_predictions_withapplications(working_path = None, rm_simulation_folder = True, **kwargs):

    # This function now accepts a SpatialCM object instead of creating a new one.
    cm_sp = SpatialCM(configuration_path = os.path.join(working_path,'crop_configuration.yaml'), load_env_data = False)

    cm_sp._tmp_path = cm_sp.config.run_path
    day_values, napp_values = get_applications_from_kwwargs(**kwargs)

    # The check for negative N is still useful here.
    total_n = np.sum(napp_values) * NITROGEN_MODEL_MULTIPLIER
    if total_n < 0:
        print("Warning: Total N applied is zero or negative. Setting yield to 0.")
        return 0
    # if len(day_values)>1:
    #     for i in range(len(day_values)-1):
    #         # Penalize invalid combinations by returning a very low NUE
    #         if (day_values[i+1] <= day_values[i] + min_split_interval): return -1e6

    # The working path is now managed by the cm_sp object.
    cm_sp.model._process_paths = [working_path]

    return yield_simulation(cm_sp, day_values, napp_values,
                                    rm_simulation_folder = rm_simulation_folder)
    

def set_enviromental_information(cm_spatial_processor, geocode: str) -> None:
        """Set up environmental information for a specific geographic site.

        Parameters
        ----------
        geocode : str
            Site code to retrieve corresponding spatial data.
        """
        from spatialdata.utils import read_compressed_xarray

        roi = cm_spatial_processor.geo_features.loc[cm_spatial_processor.geo_features['GEOCODIGO']==str(geocode)]
        roi_name = roi[cm_spatial_processor.config.SPATIAL_INFO.feature_name].values[0]
        cm_spatial_processor.set_up_folders(site = roi_name)
        
        village_info = cm_spatial_processor.config.SPATIAL_INFO.get('villages_folderpath', '')
        datauploaded = False
        if not os.path.exists(cm_spatial_processor._dem_tmppath) and os.path.exists(os.path.join(village_info, str(geocode))):
            datauploaded = True
            for dataset in ['weather', 'soil', 'dem']:
                    fn = os.path.basename(cm_spatial_processor.__dict__[f'_{dataset}_tmppath'])
                    out_path = os.path.join(cm_spatial_processor.config.SPATIAL_INFO.villages_folderpath, str(geocode), fn.replace('.nc','.zip'))
                    shutil.copy2(out_path, cm_spatial_processor._tmp_path)
                    
                    read_compressed_xarray(os.path.join(cm_spatial_processor._tmp_path, fn.replace('.nc','.zip')), cm_spatial_processor._tmp_path)
    	            
                    print(f'{fn} loaded')
        

        cm_spatial_processor.create_roi_sp_data(
        roi=roi,
        group_codes=TEXTURE_CLASSES,  # Codes for grouping data by texture
        export_spatial_data = not datauploaded
        )
        
    
def main():
    
    config_path = f'model_configurations/crop_model_configuration_coffee.yaml'
    
    
    config = OmegaConf.load(config_path)
    
    results_path = config.GENERAL_INFO.working_path
    
    geocode = '030106'
    cultivar_id = 'sun'
    
    cm_sp = SpatialCM(configuration_dict = config, load_env_data= False)
    
    roi = cm_sp.geo_features.loc[cm_sp.geo_features['GEOCODIGO']==str(geocode)]
    ## prunnning
    prunning_fraction = cm_sp.config.MANAGEMENT.get('prunning_fraction', 0)
    prunning_years = cm_sp.config.MANAGEMENT.get('prunning_years', [1])
    prunning_doys = cm_sp.config.MANAGEMENT.get('prunning_day_of_the_year', [310])
    
    set_enviromental_information(cm_sp, geocode=geocode)
    cm_sp.model.find_envworking_paths(cm_sp._tmp_path, 'csv')
    ## planting date
    planting_dates = [cm_sp.config.MANAGEMENT[f'cycle_treatment_{i+1}']['planting_date']
        for i in range(cm_sp.config.MANAGEMENT.n_cycles)]
    
    pr_schedule = None
    if prunning_fraction != 0:
        prunning_schedule = prunningOrganizer(planting_dates)
        pr_schedule = prunning_schedule.prunning_schedule(years=prunning_years, days_ofthe_year=prunning_doys, prunning_fraction=prunning_fraction)
        for i in range(len(planting_dates)):
            cm_sp.config.MANAGEMENT[f'cycle_treatment_{i+1}']['coffee_prunning'] = pr_schedule[f'cycle_treatment_{i+1}']

    # 
    caf_model = SpatialCAF(cm_sp, planting_dates)
    flowering_dates, bsl_yields = caf_model.get_flowering_dates()
    print(flowering_dates)
    
    ferti_days = caf_model.ferti_days_after_flowering(flowering_dates, [], n_value[1:], nitrogen_factor = nitrogen_factor)
        
    for z in range(spatial_processor.config.MANAGEMENT.n_cycles):
        ferti_cycle_schedule = caf_model.create_event_cycle_fert_schedule(z, *ferti_days[z+1])         
        spatial_processor.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization'] = ferti_cycle_schedule
            