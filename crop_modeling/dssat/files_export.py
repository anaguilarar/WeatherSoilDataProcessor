from .soil import DSSATSoil_fromSOILGRIDS, DSSATSoil_base
from .weather import DSSAT_Weather
from .files_reading import values_type_inffstyle

import numpy as np
import os

from datetime import datetime
from ..utils.process import check_percentage
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
                
def from_weather_to_dssat(weather_df, group_by: str = 'group', date_name ='date', 
                        params_df_names = None,refht = 2, outputpath = None, outputfn = None, codes = None, date_format = '%Y%m%d'):
    
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
        uniquegroups = np.unique(weather_df[group_by].values)
        for i in uniquegroups:
            subset = weather_df.loc[weather_df[group_by] == i]
            # tmax must be higher than tmin
            if not all(subset.TMIN <= subset.TMAX):
                subset.loc[(subset.TMIN > subset.TMAX),"TMAX"] = subset.loc[(subset.TMIN > subset.TMAX),"TMIN"]+1
            
            weatherdata = DSSAT_Weather(subset.reset_index().dropna(), parmasdssat, refht = refht)
            weatherdata._name = outputfn if outputfn is not None else weatherdata._name
            
            if codes is not None:
                #weatherdata._name = weatherdata._name#+'{}'.format(codes[i].replace(' ',''))
                outputpathgroup = os.path.join(outputpath,'{}'.format(codes[i].replace(' ','')))
            
            else:
                #weatherdata._name = weatherdata._name+'{}'.format(i)
                outputpathgroup = os.path.join(outputpath,'_{}'.format(i))
            
            if not os.path.exists(outputpathgroup):
                os.mkdir(outputpathgroup)
            weatherdata.write(outputpathgroup)
    

def from_soil_to_dssat(df, group_by: str = 'group', depth_name ='depth', 
                        country = None,site = None, outputpath = None, outputfn = None, 
                        codes = None, soil_id = None) -> None:

    
    uniquegroups = np.unique(df[group_by].values)
    firstdepth = np.unique(df[depth_name].values).tolist()
    firstdepth.sort()
    
    for i in uniquegroups:
        
        subset = df.loc[df[group_by] == i]
        sand, clay = None, None
        sand = float(check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].sand.values)))
        clay = float(check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].clay.values)))
        print(f"sand {sand} clay {clay}")
        if not np.isnan(clay) and not np.isnan(sand):
            
            long = np.round(np.nanmean(subset.x.values), 3)
            lat = np.round(np.nanmean(subset.y.values), 3)
            
            ddsat_soilgrid = DSSATSoil_fromSOILGRIDS(long = long, lat = lat, sand = sand, clay = clay, country = country,
                                                    site = site, id = soil_id)
            
            ddsat_soilgrid.add_soil_layers_from_df(subset)

            if codes is not None:
                i = int(i) if isinstance(i, float) else i
                outputfngroup = outputfn+'{}'.format(codes[i].replace(' ',''))
                outputpathgroup = os.path.join(outputpath,'{}'.format(codes[i].replace(' ','')))
            else:
                outputfngroup = outputfn+'{}'.format(i)
                outputpathgroup = os.path.join(outputpath,'{}'.format(i))    
            if not os.path.exists(outputpathgroup):
                os.mkdir(outputpathgroup)

            fn = os.path.join(outputpathgroup, 'TR.SOL')
            ddsat_soilgrid.write(fn)
            
            check_coords(fn, long=long, lat= lat, country = country)
        else:
            print("It must have at least sand and clay values")
            
