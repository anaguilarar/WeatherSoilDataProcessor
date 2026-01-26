
import os
import shutil

from datetime import datetime
from typing import List
import json
import logging

import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from crop_modeling.utils.output_transforms import (update_data_using_path, export_data_ascsv, 
                                                   coffee_yield_data_summarized, ONI_DATA, identify_enso_events)
from crop_modeling.caf.management import CAFManagement
from crop_modeling.caf.output import CAFOutputData
from crop_modeling.spatial_cm import SpatialCAF
from crop_modeling.caf.files_export import CAFSoil
from crop_modeling.utils.process import summarize_datacube_as_df

from spatialdata.utils import read_compressed_xarray

from spatialdata.soil_data import TEXTURE_CLASSES

def set_up_sim_environment(src_path, target_path):
    
    wth_path = os.path.join(src_path, 'cafdem.csv')
    sl_path = os.path.join(src_path, 'cafsoil.csv')
    assert os.path.exists(wth_path) and os.path.exists(sl_path), 'create caf files first'
    shutil.copy2(wth_path, target_path)
    shutil.copy2(sl_path, target_path)
    shutil.copy2(os.path.join(src_path, 'cafweather.csv'), target_path)

def extract_soil_data(spatial_processor, soil_variables, dimension_name: str, depth: str, group_by:str = None, project_to:str = "EPSG:4326",
                    pixel_scale: bool = False, group_codes: List = None
                    ):
    
    if not os.path.exists(spatial_processor._soil_tmppath) and os.path.exists(spatial_processor._weather_tmppath): return None
    cafsoil = CAFSoil(xrdata_path = spatial_processor._soil_tmppath)
    group_bylayer = spatial_processor._get_group_layer(project_to = None) ## avoid exporting the group layer again
    dfdata = summarize_datacube_as_df(xrdata_path=spatial_processor._soil_tmppath, dimension_name= dimension_name, group_by = group_by, 
                                    group_by_layer = group_bylayer,
                                    project_to=  project_to, pixel_scale = pixel_scale)
    dfdata = cafsoil.process_data(dfdata)
    
    variablestoexport = soil_variables
    
    if group_by is not None:
        if group_codes is not None:
            dfdata[group_by] = dfdata[group_by].apply(lambda x: group_codes[int(x)].replace(" ",""))
            
        variablestoexport = [group_by]+soil_variables
        
    dfdata = dfdata.loc[dfdata[dimension_name] == depth][variablestoexport]
        
    return dfdata

def month_year(dates: pd.Series):
    months = dates.dt.month
    years = dates.dt.year

    encodeddates = []
    for m,y in zip(months, years):
        mstr = str(m) if m >= 10 else '0{}'.format(m)
        encodeddates.append(f'{mstr}_{y}')
    
    return encodeddates

def fertilization_set_up(application_month, n_ammount, n_years, planting_year):

    ferti_event = {
        'years': [],
        'dayofyear': [],
        'N_amount': [],
    }
    
    for m in range(n_years):
        ferti_event['years'].append(planting_year + m)
        ferti_event['dayofyear'].append(application_month*30)
        ferti_event['N_amount'].append(n_ammount)
        
    return ferti_event
    

def organize_caf_fertilization_from_app(fert_input: List):
    if len(fert_input) == 0: return None, None
    if not fert_input[0]['coffee_monthsaftertransplant'].isnumeric(): return None, None
    fert_item = fert_input
    days_list = []
    nammount_list = []
    for fert_item in fert_input:
        
        to_days = int(fert_item['coffee_monthsaftertransplant'])
        amount = float(fert_item['coffee_cantidadkgha'])
        val_list = []
        for element in ['n', 'p',  'k']:
            value = min(float(fert_item.get(f'coffee_{element}',0)),100)
            total = round((value/100) * amount, 3)
            val_list.append(total)
        
        n,_,_ = val_list
        days_list.append(to_days); nammount_list.append(n)
    return days_list, nammount_list

## input params

def main_caf_function(data_dict, config, messages = None):
    
    tmp_path =  data_dict["resultpath"]
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(tmp_path, 'run.log'))

    crop = data_dict["crop"].lower()
    cultivar_id = data_dict["variety"].lower()
    duration = 7 if data_dict["duration"] == '' else int(data_dict["duration"]) if crop == 'coffee' else ""
    geocode = data_dict['aldea']

    fertilizationdata = json.loads(data_dict["fertilizationdata"]) if data_dict["fertilizationdata"] != "" else []
    app_days, n_amounts = organize_caf_fertilization_from_app(fertilizationdata)

    village_info = config.SPATIAL_INFO.get('villages_folderpath', '')

    ## organize management

    dict_organizer = CAFManagement()
    planting_dates = dict_organizer.planting_dates_from_aperiod(config.MANAGEMENT.starting_date, '2024-12-31', config.MANAGEMENT.n_cycles, coffe_plant_duration=duration)
    management_dict = dict_organizer.create_config_template(planting_dates, starting_date=config.MANAGEMENT.starting_date)

    management_dict['TREE']['species_name'] = cultivar_id
    if cultivar_id.lower() == 'sun': management_dict['TREE']['tree_density'] = 0
    else: management_dict['TREE']['tree_density'] = None

    ## define model

    cm_sp = SpatialCAF(configuration_dict = config, load_env_data= False, planting_date = planting_dates)
    cm_sp.upload_crop_model()
        
    cm_sp.config.MANAGEMENT = management_dict

    ## create spatial data
    logging.info(messages[1])
    roi = cm_sp.geo_features.loc[cm_sp.geo_features['GEOCODIGO']==str(geocode)]
    roi_name = roi[cm_sp.config.SPATIAL_INFO.feature_name].values[0]
    cm_sp.set_up_folders(site = roi_name)

    datauploaded = False
    if not os.path.exists(cm_sp._dem_tmppath) and os.path.exists(os.path.join(village_info, str(geocode))):
        datauploaded = True
        for dataset in ['weather', 'soil', 'dem']:
            fn = os.path.basename(cm_sp.__dict__[f'_{dataset}_tmppath'])
            out_path = os.path.join(village_info, str(geocode), fn.replace('.nc','.zip'))
            shutil.copy2(out_path, cm_sp._tmp_path)
            read_compressed_xarray(os.path.join(cm_sp._tmp_path, fn.replace('.nc','.zip')), cm_sp._tmp_path)
            
    workingpath = cm_sp.create_roi_sp_data(
        roi=roi,
        group_codes=TEXTURE_CLASSES,  # Codes for grouping data by texture
        export_spatial_data=not datauploaded
    )
    soildata = extract_soil_data(cm_sp, soil_variables=["phh2o","som"], dimension_name='depth', depth='15-30', group_by='texture', group_codes=TEXTURE_CLASSES)
    n_ammount = 0
    ## get flowering dates if there is any nitrogen application
    flowering_dates = None
    logging.info(messages[2])
    if app_days is not None:
        flowering_dates, _ = cm_sp.get_flowering_dates(duration=duration, remove_tmp_folders = True, verbose= False)
        ## assuming that therie is not a big difference between localities with different 
        flowering_dates_df = flowering_dates[list(flowering_dates.keys())[0]]
        n_ammount = np.sum(n_amounts)
        ferti_days_percycle = cm_sp.ferti_days_after_flowering(flowering_dates_df, app_days, n_amounts)
        for z in range(cm_sp.config.MANAGEMENT.n_cycles):
            print('---> cycle: ', z)
            ferti_cycle_schedule = cm_sp.create_event_cycle_fert_schedule(z, *ferti_days_percycle[z+1])         
            cm_sp.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization'] = ferti_cycle_schedule
            print(cm_sp.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization'])
    
    logging.info(messages[3])
    completed_sims = cm_sp.run_caf(verbose=False)
    model_data = update_data_using_path(cm_sp._tmp_path, model = cm_sp.model.name)
    completed_sims = completed_sims if completed_sims is not None else {k:True for k, v in model_data.items()}
    logging.info(messages[4])
    export_data_ascsv(completed_sims, model_data, cm_sp.crop.lower(), cm_sp._tmp_path, cm_sp.model.name, weather_variables2export = ['date', 'tmin', 'tmax', 'rain'])

    ## organize results
    soildata['total_nitrogen'] = n_ammount
    fn = os.path.join(cm_sp._tmp_path,f'soil_and_nitrogen_values_{crop}.json')
    soildata.to_json(fn, orient='records')
    logging.info(messages[5])
    
    potential_yield_path = os.path.join(workingpath,f'{crop.lower()}_potential_yield.csv')
    outputmodel_data = pd.read_csv(potential_yield_path).dropna()
    groupby_colname = 'texture'
    ouput_columns = cm_sp._colnames
    group_dates_by = ouput_columns.growth_colnames['number_of_cycle']
    date_col_name = ouput_columns.growth_colnames['date']
    target_col_name = ouput_columns.growth_colnames['yield']
    fl_column = ouput_columns.growth_colnames['flowering_date']
    harvest_col_name = ouput_columns.growth_colnames.get('hdate', None)
    weather_date_col_name = ouput_columns.weather_columns['date']

    outputmodel_data = pd.read_csv(potential_yield_path).dropna()
    # weather
    
    weather = pd.read_csv(os.path.join(workingpath,'weather.csv'))

    weather_variables = weather.columns[1:]
    pos_weathervar_oi = [list(weather.columns).index(col) for col in weather_variables]
    weather_variables

    for texture_class, subset in outputmodel_data.groupby([groupby_colname]):
        yield_data = coffee_yield_data_summarized(subset, date_column = date_col_name, yield_column=target_col_name, n_cycle_column=group_dates_by)
        #yield_data.to_csv(os.path.join(workingpath, f'yield_data_{texture_class[0]}.csv'))
        yield_data.loc[~ np.isnan(yield_data.daytoflower)][['date','period']].to_csv(os.path.join(workingpath,f'flowering_dates_{texture_class[0]}.csv'))
        datastacked = []
        
        # add enso data
        for i, cycledata in yield_data.groupby([group_dates_by]):
            dates = cycledata[date_col_name].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
            year_month = np.unique([[int(d[3:]), int(d[:2])] for d in month_year(dates)], axis = 0)
            classification_df = identify_enso_events(ONI_DATA, year_month)
            classification_df[group_dates_by] = i[0]
            datastacked.append(classification_df)

        dates = yield_data[date_col_name].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        classification_df = pd.concat(datastacked)
        classification_df['month_year'] = ['_'.join([str(m) if m>9 else f'0{m}', str(y)]) for y, m in zip(classification_df.Year.values, classification_df.Month.values)]
        yield_data['month_year'] = month_year(dates)

        pltdata = []
        for i in yield_data[group_dates_by].unique():
            cycledata = yield_data.loc[yield_data[group_dates_by] == i]
            classification_cycle = classification_df.loc[classification_df[group_dates_by] == i]
            cycledata = cycledata.reset_index().merge(classification_cycle.reset_index()[['month_year', 'Season', 'Phase']], 
                                                        on = 'month_year', how = 'left')
            pltdata.append(cycledata[[date_col_name, target_col_name, fl_column, group_dates_by, harvest_col_name, groupby_colname, 
                                    "date","nyear_month_day","daytoflower","period","month_year","Season","Phase"]])
        pd.concat(pltdata).to_csv(os.path.join(workingpath, f'yield_data_{texture_class[0]}.csv'))
        
        weather_percycle = []
        ## create weather pattern
        for z, (group_id, data) in enumerate(yield_data.groupby([group_dates_by])):
            pdate = data.HDAT.values[0]
            hdate = data.HDAT.values[-1]
            weather_summarized = weather.values[np.logical_and(weather['DATE']>=pdate,weather['DATE']<=hdate)][:,pos_weathervar_oi].swapaxes(0,1)
            weather_df = pd.DataFrame(weather_summarized.T, columns= weather_variables)
            weather_df['month_year'] = month_year(data.HDAT.apply(lambda x: datetime.strptime(x, '%Y-%m-%d')))
            
            weather_summarized = weather_df.groupby('month_year').agg({'RAIN': 'sum', 'TMIN': 'mean', 'TMAX': 'mean'}).reset_index()
            year_month = np.unique([[int(i[3:]), int(i[:2])] for i in weather_df['month_year'].values], axis = 0)
            dfoni = identify_enso_events(ONI_DATA, year_month)

            dfoni['month_year'] = dfoni.apply(lambda x: '{}_{}'.format(x.Month if int(x.Month)>9 else f'0{x.Month}', int(x.Year)), axis = 1)
            weather_summarized = weather_summarized.reset_index().merge(dfoni.reset_index()[['month_year', 'Season', 'Phase']], 
                                                                on = 'month_year', how = 'left')

            ## FLOWERING
            fl_dt = [datetime.strptime(x, '%Y-%m-%d') for x in data.loc[data[fl_column] == 1].HDAT.values ]
            fl_df = pd.DataFrame({"month_year": month_year(pd.Series(fl_dt)), fl_column: 1})
            weather_summarized = weather_summarized.reset_index().merge(fl_df.reset_index()[['month_year', fl_column]], 
                                                                on = 'month_year', how = 'left')

            weather_summarized['dummy_date'] = weather_summarized.month_year.apply(lambda x: datetime.strptime( '{}-{}-01'.format((int(x.split('_')[1])-(int(pdate[:4])))+2000, x.split('_')[0]), '%Y-%m-%d'))
            weather_summarized = weather_summarized.sort_values(['dummy_date'])
            weather_summarized[group_dates_by] = z+1
            weather_percycle.append(weather_summarized)
            
        pd.concat(weather_percycle).to_csv(os.path.join(workingpath, f'weatherbytreatment_{texture_class[0]}.csv'))

    orig_cm_enviroments = cm_sp.model._process_paths

    ## add recomendations
    nsimluation_path = config.NITROGEN_OPT.get('simulations_output_dir', None)
    geocode_n_outputs= os.path.join(nsimluation_path, str(geocode))
    
    ## ENSO adition
    datastacked = []
    for i in np.unique(outputmodel_data[group_dates_by]):
        cycledata = outputmodel_data.loc[outputmodel_data[group_dates_by] == i]

        dates = cycledata[date_col_name].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        year_month = np.unique([[int(i[3:]), int(i[:2])] for i in month_year(dates)], axis = 0)
        # Run Classification
        classification_df = identify_enso_events(ONI_DATA, year_month)
        classification_df[group_dates_by] = i
        datastacked.append(classification_df)

    dates = outputmodel_data.HDAT.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    classification_df = pd.concat(datastacked)
    classification_df['month_year'] = ['_'.join([str(m) if m>9 else f'0{m}', str(y)]) for y, m in zip(classification_df.Year.values, classification_df.Month.values)]
    outputmodel_data['month_year'] = month_year(dates)

    pltdata = []
    for i in outputmodel_data.n_cycle.unique():
        cycledata = outputmodel_data.loc[outputmodel_data.n_cycle == i]
        classification_cycle = classification_df.loc[classification_df.n_cycle == i]
        pltdata.append(cycledata.reset_index().merge(classification_cycle.reset_index()[['month_year', 'Season', 'Phase']], 
                                                    on = 'month_year', how = 'left'))
        
    if os.path.exists(geocode_n_outputs):
        
        if flowering_dates is None:
            print('---------------------------> get get_flowering_dates')
            flowering_dates, baseline_yields = cm_sp.get_flowering_dates(duration=duration, remove_tmp_folders = True, verbose= False, run_model=False)
            ## assuming that therie is not a big difference between localities with different 
            flowering_dates_df = flowering_dates[list(flowering_dates.keys())[0]]
            
        if not os.path.exists(os.path.join(cm_sp._tmp_path, 'best_practices')): os.mkdir(os.path.join(cm_sp._tmp_path, 'best_practices'))
        cm_sp.model._process_paths = [os.path.join(cm_sp._tmp_path, 'best_practices')]
        for cm_env in orig_cm_enviroments:
            texture_class = os.path.basename(cm_env)
            n_fn = os.path.join(nsimluation_path, str(geocode), f'nitrogen_results_{texture_class}_coffee_sun.json')
            if not os.path.exists(n_fn): continue
            
            n_sim_data = pd.read_json(n_fn)
            yield_data = n_sim_data['crop_yield'].values
            maxy = np.nanmax(yield_data)
            miny = np.nanmin(yield_data)
            nyield_data = (yield_data-miny)/(maxy - miny)

            nyield_filtered = nyield_data[nyield_data<0.9]

            threshold_value = float((np.sort(nyield_filtered)[-1] * (maxy - miny))+miny)
            best_subset = n_sim_data[n_sim_data['crop_yield'].values >= threshold_value]
            minval_nit = best_subset.sort_values(['nit_ayear3'])['nit_ayear3'].values[0]
            best_subset = best_subset[best_subset['nit_ayear3'] == minval_nit]
            best_subset = best_subset.sort_values(['crop_yield'],ascending= False).iloc[0:1]
            n_value = [int(best_subset['n0'].values[0]), int(best_subset['n1'].values[0]), int(best_subset['n2'].values[0])]

            application_day = [int(best_subset['month0'].values[0]), int(best_subset['month1'].values[0]), int(best_subset['month2'].values[0])]
            set_up_sim_environment(cm_env, cm_sp.model._process_paths[0])
            
            ferti_days = cm_sp.ferti_days_after_flowering(flowering_dates_df, application_day[1:], n_value[1:], nitrogen_factor = 1)
            for z in range(cm_sp.config.MANAGEMENT.n_cycles):
                ferti_cycle_schedule = cm_sp.create_event_cycle_fert_schedule(z, *ferti_days[z+1])         
                cm_sp.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization'] = ferti_cycle_schedule
                
            
            for z in range(cm_sp.config.MANAGEMENT.n_cycles):
                ferti_event = fertilization_set_up(int(application_day[0]), int(n_value[0]), 2, int(pd.to_datetime(planting_dates)[z].year))
                stablishment_schedule = {
                    'years': cm_sp.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization']['years'] + ferti_event['years'],
                    'dayofyear': cm_sp.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization']['dayofyear'] + ferti_event['dayofyear'],
                    'N_amount': cm_sp.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization']['N_amount'] + ferti_event['N_amount'],
                    'n_fertilization_per_year': 1
                    
                }

                cm_sp.config.MANAGEMENT[f'cycle_treatment_{z+1}']['fertilization'] = stablishment_schedule
                
            cm_sp.run_caf(setup_envs=False, verbose=False)
            output_data = {'best_practices':CAFOutputData(cm_sp.model._process_paths[0])}
            output_df = export_data_ascsv({'best_practices': True}, output_data, cm_sp.crop.lower(), cm_sp._tmp_path, cm_sp.model.name, weather_variables2export = ['date', 'tmin', 'tmax', 'rain'], export_data=False)
            
            for txt_class, subset in output_df.groupby([groupby_colname]):
                print(os.path.join(workingpath, f'{txt_class[0]}_yield_data_{texture_class}.csv'))
                yield_data = coffee_yield_data_summarized(subset, date_column = date_col_name, yield_column=target_col_name, n_cycle_column=group_dates_by, harvest_column=harvest_col_name)
                print(yield_data)
                yield_data.to_csv(os.path.join(workingpath, f'{txt_class[0]}_yield_data_{texture_class}.csv'))
    return workingpath
        
if __name__ == "__main__":
    MESSAGES = {
    1: "extracting_spatial_information",
    2: "setting_up_crop_model_files",
    3: "running_crop_model",
    4: "exporting_output_files",
    5: "organizing_data_for_visualization",
    6: "process_completed"
    }
    
    params_input = '{"resultpath": "runs/","flagweather": "1","id_user": "13","crop": "coffee","variety": "sun","aldea": "151910","latitud": "14.798048","longitud": "-87.288973","duration": "","fertilizationdata": "[{\\"coffee_monthsaftertransplant\\":\\"\\",\\"coffee_fuente\\":\\"\\",\\"coffee_n\\":\\"\\",\\"coffee_p\\":\\"\\",\\"coffee_k\\":\\"\\",\\"coffee_cantidadkgha\\":\\"\\"}]"}'
    data_dict = json.loads(params_input)
    #params_input = '{"resultpath": "runs/","flagweather": "1","id_user": "13","crop": "coffee","variety": "sun","aldea": "031501","latitud": "14.798048","longitud": "-87.288973","duration": "","fertilizationdata": "[{\\"coffee_monthsaftertransplant\\":\\"10\\",\\"coffee_fuente\\":\\"5\\",\\"coffee_n\\":\\"12\\",\\"coffee_p\\":\\"24\\",\\"coffee_k\\":\\"12\\",\\"coffee_cantidadkgha\\":\\"100\\"},{\\"coffee_monthsaftertransplant\\":\\"20\\",\\"coffee_fuente\\":\\"2\\",\\"coffee_n\\":\\"18\\",\\"coffee_p\\":\\"46\\",\\"coffee_k\\":\\"0\\",\\"coffee_cantidadkgha\\":\\"50\\"}]"}'
    config_path = f'model_configurations/crop_model_configuration_coffee.yaml'
    crop_configuration = OmegaConf.load(config_path)

    crop_configuration.GENERAL_INFO.working_path = f'runs'
    
    workingpath = main_caf_function(data_dict, config=crop_configuration, messages = MESSAGES)
    print(workingpath)
