import os
import sys
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import geopandas as gpd
from omegaconf import OmegaConf
from tqdm import tqdm
import concurrent.futures
import shutil
import copy


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# vendored Py_DSSATTools imports itself as top-level "DSSATTools", so its
# folder must be importable when the package is not pip-installed
sys.path.insert(1, os.path.join(REPO, "Py_DSSATTools"))
from crop_modeling.utils.ferti_optimization import FertilizerBayesian  # noqa: E402
from crop_modeling.utils.output_transforms import yield_data_summarized, ColumnNames, DSSATOutputData, two_digit_format  # noqa: E402


CULTIVARS = {"bean": "IB0006", "maize": "IB0060"}

def get_dssat_ouput_asdf(processor, target = 'yield',  envid=0):
    
    colnames = ColumnNames(processor.name)
    date_column = colnames.growth_colnames['date']
    
    target_column = colnames.growth_colnames[target]
    harvest_column = colnames.growth_colnames['hdate']
    if target == 'nitrogen_uptake':
        target_column = colnames._nitrogen_uptake['dssat']['nitrogen_uptake']

    #model_data = update_data_using_path(self._tmp_path, model = self.model.name)
    if processor.name == 'dssat':
        model_data = DSSATOutputData(processor._process_paths[envid])
        model_data = pd.DataFrame(model_data.output_data())
        model_data = model_data.loc[model_data[target_column] > 0]
        
    return yield_data_summarized(model_data,'TRNO', date_column=date_column,yield_column= target_column, 
                                    harvest_column=harvest_column)


def concat_simulation_outputs(geocode:str, config:dict,  dates:np.ndarray,  date_format:str = '%Y-%m-%d'):
    
    working_dir = config.GENERAL_INFO.get('working_path')
    tps = config.MANAGEMENT.plantingWindow
    napp = config.BAYESIAN.get('n_applications',3)

    env_paths = [i for i in os.listdir(os.path.join(working_dir,geocode)) if os.path.isdir(os.path.join(working_dir,geocode,i))]
    fn_template  = 'tp_{tp}_{soil_text}_'+str(napp)+'.csv'

    pdlist= []
    for j in range(len(env_paths)):
        data_list = []
        for tp in range(tps):
            ## replace file in the template
            fn = fn_template.format(tp = tp, soil_text = env_paths[j])
            if not os.path.exists(os.path.join(working_dir, geocode, fn)): continue
            #data = read_file(os.path.join(working_dir, geocode, fn))
            data = pd.read_csv(os.path.join(working_dir, geocode, fn))
            data['total_nitrogen']  = np.sum([data[k] for k in config.BAYESIAN.schedules[f'app_{napp}'].keys() if 'day' not in k], axis = 0)
            # filter 0 yields
            data = data.loc[data['crop_yield']>0]
            data['date'] = datetime.strptime(dates[tp],date_format)
            data['month_day'] = ['{}-{}'.format(two_digit_format(month), two_digit_format(day)) for month, day in zip(data['date'].dt.month, data['date'].dt.day)]
            data['date'] = data['date'].apply(lambda x: datetime.strftime(x, date_format))
            data_list.append(data)
            # export data
        if len(data_list)<=1: continue
        pdlist.append(pd.concat(data_list).reset_index()) #[pd.concat(data_list).reset_index().columns[2:]].to_json(os.path.join(outputpath,f'nitrogen_results_{env_paths[j]}.json'), orient='records')

    return pdlist

def run_dssat(spatial_cm):
    
    spatial_cm.model.find_envworking_paths(spatial_cm._tmp_path, 'WTH')
    # Set up crop files
    spatial_cm.model.set_up_crop(crop=spatial_cm.crop, cultivar=spatial_cm.cultivar)
    # Set up management files
    spatial_cm.model.set_up_management(crop=spatial_cm.crop, cultivar=spatial_cm.cultivar, **spatial_cm.config.MANAGEMENT)
    # run the simulation
    
    dssath_path = spatial_cm.config.GENERAL_INFO.get('dssat_path', None)
    return spatial_cm.model.run(spatial_cm.model.crop_code, crop=spatial_cm.crop,planting_window=spatial_cm.config.MANAGEMENT.plantingWindow,
                                ncores = spatial_cm.config.GENERAL_INFO.ncores,
                                bin_path = spatial_cm.config.GENERAL_INFO.bin_path, dssat_path = dssath_path, remove_tmp_folder=False)
    
        

def create_non_fertilizer_scenario(geocode, config, load_env_data = False):
    config = copy.deepcopy(config)
    config.GENERAL_INFO.working_path = 'runs'
    config.MANAGEMENT.fertilizer_schedule = {'days_after_planting': None,
                                        'npk': None}
    
    cm_optimizer = FertilizerBayesian(config, load_env_data = load_env_data)

    cm_optimizer.set_enviromental_information(geocode)
    run_dssat(cm_optimizer)
    dfbl_list = []
    for envid in range(len(cm_optimizer.model._process_paths)):
        yield_data = get_dssat_ouput_asdf(cm_optimizer.model,envid= envid)
        baseline_data = pd.DataFrame()
        baseline_data['crop_yield'] = yield_data['HWAH'].values
        baseline_data['date'] = yield_data['pdat_year_month_day'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
        baseline_data['month_day'] = yield_data['month_day'].values
        napp = config.BAYESIAN.get('n_applications',3)
        for k in config.BAYESIAN.schedules[f'app_{napp}'].keys(): baseline_data[k] = 0
        baseline_data['total_nitrogen'] = 0
        
        dfbl_list.append(baseline_data)
        
    return dfbl_list

def run_optimizer(geocode, crop, config = None):

    config_path = f'model_configurations/crop_model_configuration_{crop}.yaml'
    if config is None:
        config = OmegaConf.load(config_path)
    else:
        config = copy.deepcopy(config) 
    fert_optimizer = FertilizerBayesian(config, load_env_data = False)
    #try:
    fert_optimizer.set_enviromental_information(geocode)
    #except:
    #    return False
    napp = fert_optimizer.config.BAYESIAN.get('n_applications',3)
    total_nitrogen = fert_optimizer.config.BAYESIAN.schedules[f'app_{napp}']
    fert_optimizer.set_application_schedule(total_nitrogen)

    for env in range(len(fert_optimizer._cm_envs)):
        fert_optimizer.simulate_multipletimes(
            env = env, 
            time_windows= fert_optimizer.config.MANAGEMENT.plantingWindow, 
            n_iter=fert_optimizer.config.BAYESIAN.n_iter, 
            ncores=0)
    
    return True

def run_n_optimization(roi_name, crop):
    
    config_path = f'model_configurations/crop_model_configuration_{crop}.yaml'
    config = OmegaConf.load(config_path)
    output_path = config.BAYESIAN.get('output_path', f'{crop}_optimization')
    sp_output_path = os.path.join(output_path, roi_name)
    if os.path.exists(sp_output_path):
        if len(os.listdir(sp_output_path))>0: return None
    
    
    optimizerresults = run_optimizer(roi_name, crop, config)
    if not optimizerresults: return None
    
    os.makedirs(sp_output_path)
    
    
    nonferti_ouputs = create_non_fertilizer_scenario(roi_name, config)
    dates = nonferti_ouputs[0].date.values[::1]
    bayesian_apps = concat_simulation_outputs(roi_name, config, dates,  date_format = '%Y-%m-%d')

    
    sim_path = os.path.join(config.GENERAL_INFO.working_path, roi_name)
    env_paths = [i for i in os.listdir(sim_path) if os.path.isdir(os.path.join(sim_path,i))]

    if len(bayesian_apps) == len(nonferti_ouputs):
        for i in range(len(nonferti_ouputs)):
            #print(os.path.join(sp_output_path,f'nitrogen_results_{env_paths[i]}_{crop}_{config.CROP.cultivar}.json'))
            pd.concat([nonferti_ouputs[i], bayesian_apps[i][nonferti_ouputs[i].columns]]).to_json(
                os.path.join(sp_output_path,f'nitrogen_results_{env_paths[i]}_{crop}_{config.CROP.cultivar}.json'), orient='records')
    
    try:
        shutil.rmtree(sim_path, ignore_errors=True)
    except:
        return None

def main():

    p = argparse.ArgumentParser(description=__doc__, formatter_class= argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required = True, help = "target crop, currently maize and beans")
    p.add_argument("--cultivar", default = None)
    p.add_argument("--output", default = "runs")
    
    parsedargs = p.parse_args()
    
    
    crop = parsedargs.crop
    cultivar = parsedargs.cultivar
    
        
    assert crop in ["bean", "maize"], "No other crop is allowed"
    
    if cultivar is None:
        cultivar = CULTIVARS[crop]
    
    print(crop, cultivar)
    config_path = f'D:/OneDrive - CGIAR/scripts/SueloDeHonduras/WeatherSoilDataProcessor/model_configurations/crop_model_configuration_{crop}.yaml'
    config = OmegaConf.load(config_path)
    print(config.SPATIAL_INFO.villages_folderpath)
    rois = gpd.read_file(config.SPATIAL_INFO.geospatial_path)
    roi_names = set()
    config.CROP.cultivar = cultivar
    
    for i in rois[config.SPATIAL_INFO.feature_name].values:
        roi_names.add(i)
    #roi_names = np.unique(roi_names)
    
    roi_names = list(roi_names)
    print(len(roi_names))
    #run_n_optimization(roi_name, crop)

    with tqdm(total=len(roi_names)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
            future_to_tr ={executor.submit(run_n_optimization, roi_names[idx], crop): (idx
                                      ) for idx in range(10,len(roi_names))}
            for future in concurrent.futures.as_completed(future_to_tr):
                idpx = future_to_tr[future]
                try:
                    future.result()
                
                except Exception as exc:
                    print(f"Request for treatment {idpx} generated an exception: {exc}")
                    pbar.update(1)
    
    
    # print(3*(len(roi_names)//4), 4*(len(roi_names)//5))
    # run
    # for idx in tqdm(range(1000, 1010)):
    #     #if not os.path.exists(os.path.join(config.GENERAL_INFO.working_path, str(roi_names[idx]))):
    #         #try:
    #     run_n_optimization(str(roi_names[idx]), crop)
    #         #except:
    #         #    continue

if __name__ == '__main__':
    main()
    
    