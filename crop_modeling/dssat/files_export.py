from .soil import DSSATSoil_fromSOILGRIDS, DSSATSoil_base
from .weather import DSSAT_Weather
from .files_reading import values_type_inffstyle
from ..utils.process import check_percentage

import numpy as np
import os
import concurrent.futures

from datetime import datetime

import fortranformat as ff

from ._base import section_indices

def check_coords(path: str, long, lat, country) -> None:
    fmt = '1X,A11,1X,A11,1X,F7.3,1X,F8.3,A24'
    _, formattypes = values_type_inffstyle(fmt)
    
    lines = DSSATSoil_base.open_file(path)
    infoindices = list(section_indices(lines, pattern='@'))
    
    if lines[infoindices[0]+1].startswith('        '):
        #lilines = [i for i in lines[infoindices[0]+1].split(' ') if i!= '']
        lilines = ['Tempor', country[:3], lat, long, 'Unclassified\n']
        vals = [formattypes[i](val) for i, val in enumerate(lilines)]
        lines[infoindices[0]+1] = ff.FortranRecordWriter(fmt).write(vals)
        # = f' Tempor      {country[:3]}           {lat}  {long} Unclassified\n'    
        with open(path, 'w') as file:
            for line in lines:
                file.write(f"{line}")
                
def from_weather_to_dssat(weather_df, group_by = None, date_name ='date', 
                        params_df_names = None,refht = 2, outputpath = None, outputfn = None, 
                        codes = None, date_format = '%Y%m%d', pixel_scale = False,
                        ncores = 5,
                        sub_working_path = None):
    
    sub_working_path = 'dssatenv' if sub_working_path is None else sub_working_path
    
    def file_for_dssat_env(df_subset, id_group):
        if codes is not None:
            id_group = int(id_group) if isinstance(id_group, float) else id_group
            outputpathgroup = os.path.join(outputpath,'{}'.format(codes[id_group].replace(' ','')))
        else:
            outputpathgroup = os.path.join(outputpath,'{}'.format(id_group))  

        create_weather_file(df_subset, outputpathgroup)
        
    def create_weather_file(subset, output_path):
        # tmax must be higher than tmin
        if not all(subset.TMIN <= subset.TMAX):
            subset.loc[(subset.TMIN > subset.TMAX),"TMAX"] = subset.loc[(subset.TMIN > subset.TMAX),"TMIN"]+1
        
        weatherdata = DSSAT_Weather(subset.reset_index().dropna(), parmasdssat, refht = refht)
        weatherdata._name = outputfn if outputfn is not None else weatherdata._name
        
        if not os.path.exists(output_path): os.mkdir(output_path)
        weatherdata.write(output_path)
            
        
    params_df_names = params_df_names if params_df_names is not None else  {
        "DATE": date_name,
        "TMIN": "tmin",
        "SRAD": "srad",
        "RAIN": "precipitation",
        "TMAX": "tmax",
        "LON": "x",
        "LAT": "y"
    }
    ## check date
    
    if not isinstance(weather_df[date_name].values[0], np.datetime64):
        weather_df[date_name] = weather_df[date_name].map(lambda x: datetime.strptime(x, date_format))
        
    changenames = {c:k for k,c in params_df_names.items()}

    weather_df = weather_df.rename(columns = changenames)
    
    parmasdssat = {k:k for k,v in params_df_names.items()}
    
    if group_by:
        for i in np.unique(weather_df[group_by].values):
            file_for_dssat_env(weather_df.loc[weather_df[group_by] == i], i)
            
    elif pixel_scale:
        group_by = 'pixel'
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
            future_to_tr ={executor.submit(file_for_dssat_env, weather_df.loc[weather_df[group_by] == weather_df[group_by].values[i]],
                                        weather_df[group_by].values[i]): (
                i) for i in range(np.unique(weather_df[group_by].values).shape[0])}
            for future in concurrent.futures.as_completed(future_to_tr):
                tr = future_to_tr[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Request for treatment {tr} generated an exception: {exc}")
                        
    else:
        create_weather_file(weather_df, os.path.join(outputpath,sub_working_path))


def from_soil_to_dssat(df, group_by = None, depth_name ='depth', 
                        country = None,site = None, outputpath = None, outputfn = None, 
                        codes = None, soil_id = None, pixel_scale = False, ncores = 5,
                        sub_working_path = None,verbose = True) -> None:
    
    def file_for_dssat_env(df_subset, id_group):
        if codes is not None:
            id_group = int(id_group) if isinstance(id_group, float) else id_group
            outputpathgroup = os.path.join(outputpath,'{}'.format(codes[id_group].replace(' ','')))
        else:
            outputpathgroup = os.path.join(outputpath,'{}'.format(id_group))  

        create_soil_file(df_subset, outputpathgroup)
        
    def create_soil_file(subset, output_path, verbose = True):

        sand, clay = None, None
        sand = float(check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].sand.values)))
        clay = float(check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].clay.values)))
        if verbose: print(f"sand {sand} clay {clay}")
        if not np.isnan(clay) and not np.isnan(sand):
            
            long = np.round(np.nanmean(subset.x.values), 3)
            lat = np.round(np.nanmean(subset.y.values), 3)
            
            ddsat_soilgrid = DSSATSoil_fromSOILGRIDS(long = long, lat = lat, sand = sand, clay = clay, country = country,
                                                    site = site, id = soil_id)
            
            ddsat_soilgrid.add_soil_layers_from_df(subset)
            if not os.path.exists(output_path): os.mkdir(output_path)
            fn = os.path.join(output_path, 'TR.SOL')
            ddsat_soilgrid.write(fn)
            
            check_coords(fn, long=long, lat= lat, country = country)
        else:
            print("It must have at least sand and clay values")
    
    sub_working_path = 'dssatenv' if sub_working_path is None else sub_working_path
    
    firstdepth = np.unique(df[depth_name].values).tolist()
    firstdepth.sort()
    if group_by:
        for i in np.unique(df[group_by].values):
            file_for_dssat_env(df.loc[df[group_by] == i], i)
            
    elif pixel_scale:
        group_by = 'pixel'
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
            future_to_tr ={executor.submit(file_for_dssat_env, df.loc[df[group_by] == df[group_by].values[i]], df[group_by].values[i]): (
                i) for i in range(np.unique(df[group_by].values).shape[0])}
            for future in concurrent.futures.as_completed(future_to_tr):
                tr = future_to_tr[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Request for treatment {tr} generated an exception: {exc}")
                                        
    
    else:
        create_soil_file(df, os.path.join(outputpath,sub_working_path), verbose= verbose)
                
        
        
                
