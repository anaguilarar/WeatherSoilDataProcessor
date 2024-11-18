
import numpy as np
from .dssat.weather import DSSAT_Weather
from .dssat.soil import DSSATSoil_fromSOILGRIDS

from tqdm import tqdm
import os
import pandas as pd
import concurrent.futures

def get_variables_to_summarize(xrdata, indexname):
     ## add coords

    datavars = {i:'mean' for i in xrdata.data_vars.keys()}
    for i in xrdata.sizes.keys():
        if indexname == i:
            continue
        datavars.update({i:'mean'})
        datavars.update({i:'mean'})

    return datavars


def check_weatherxr_scales(xrdata, tmax_colname = 'tmax', tmin_colname = 'tmin', srad_colname = 'srad'):

    if tmax_colname in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[tmax_colname].values)>273.15:
        xrdata[tmax_colname] -= 273.15
    if tmin_colname in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[tmin_colname].values)>273.15:
        xrdata[tmin_colname] -= 273.15
    if srad_colname in list(xrdata.data_vars.keys()) and np.nanmax(xrdata[srad_colname].values)>1000000:
        xrdata[srad_colname] /= 1000000
    return xrdata


def summarize_dataframe(date, xrdata, variables, groupby = None):
        ddf = xrdata.to_dataframe().reset_index().dropna()
        if groupby:
            ddf = ddf.groupby([groupby], dropna = True).agg(variables).reset_index()
        else:
            ddf['tmp'] = 0
            groupby= 'tmp'
            ddf = ddf.dropna().reset_index()
        ddf['date'] = date

        return ddf

def from_weather_to_dssat(xrdata, groupby: str = None, date_name ='date', 
                          params_df_names = None,refht = 2, outputpath = None, outputfn = None, codes = None, ncores = 0):
    
    params_df_names = params_df_names if params_df_names is not None else  {
        "DATE": date_name,
        "TMIN": "tmin",
        "SRAD": "srad",
        "RAIN": "precipitation",
        "TMAX": "tmax",
        "LON": "x",
        "LAT": "y"
    }
    weatherdatavars = {}
    weatherdatavars = get_variables_to_summarize(xrdata, date_name)
    if groupby:
        weatherdatavars.pop(groupby)

    changenames = {c:k for k,c in params_df_names.items()}
    

    xrdata = check_weatherxr_scales(xrdata)
    #weather_df = xrdata.to_dataframe().reset_index().dropna()
    
    ddf = xrdata.to_dataframe().reset_index().dropna()
    weather_df = ddf.groupby([groupby, date_name], dropna = True).agg(weatherdatavars).reset_index()

    weather_df = weather_df.rename(columns = changenames)
    
    parmasdssat = {k:k for k,v in params_df_names.items()}
    
    if groupby:
        uniquegroups = np.unique(weather_df[groupby].values)
        for i in uniquegroups:
            subset = weather_df.loc[weather_df[groupby] == i]
            if not all(subset.TMIN <= subset.TMAX):
                subset.loc[(subset.TMIN > subset.TMAX),"TMAX"] = subset.loc[(subset.TMIN > subset.TMAX),"TMIN"]+1
            weatherdata = DSSAT_Weather(subset.reset_index().dropna(), parmasdssat, refht = refht)
            weatherdata._name = outputfn if outputfn is not None else weatherdata._name
            if codes is not None:
                weatherdata._name = weatherdata._name+'{}'.format(codes[i].replace(' ',''))
                outputpathgroup = os.path.join(outputpath,'{}'.format(codes[i].replace(' ','')))
            
            else:
                weatherdata._name = weatherdata._name+'{}'.format(i)
                outputpathgroup = os.path.join(outputpath,'_{}'.format(i))
            
            if not os.path.exists(outputpathgroup):
                os.mkdir(outputpathgroup)
            weatherdata.write(outputpathgroup)
    
def check_percentage(value):
    return value *0.1 if np.max(value)>100 else value


def from_soil_to_dssat(xrdata, groupby: str = None, depth_name ='depth', 
                        country = None,site = None, outputpath = None, outputfn = None, 
                        codes = None, soil_id = None):
    
    soildatavars = get_variables_to_summarize(xrdata, depth_name)
    soildf = xrdata.to_dataframe().reset_index().dropna()

    if groupby:
        soildatavars.pop(groupby)
        ddf = soildf.groupby([groupby, depth_name], dropna = True).agg(soildatavars).reset_index()
    else:
        ddf = soildf
        ddf['tmp'] = 0
        groupby= 'tmp'
        ddf = ddf.groupby([groupby,depth_name], dropna = True).agg(soildatavars).reset_index()
    

    uniquegroups = np.unique(ddf[groupby].values)
    firstdepth = np.unique(soildf[depth_name].values).tolist()
    firstdepth.sort()

    for i in uniquegroups:
        subset = ddf.loc[ddf[groupby] == i]
        sand, clay = None, None
        sand = check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].sand.values))
        clay = check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].clay.values))
        print(f"sand {sand} clay {clay}")
        if not np.isnan(clay) and not np.isnan(sand):
            long = subset.loc[subset[depth_name] == firstdepth[0]].x.values.mean()
            lat = subset.loc[subset[depth_name] == firstdepth[0]].y.values.mean()
        
            ddsat_soilgrid = DSSATSoil_fromSOILGRIDS(long = long, lat = lat, sand = sand, clay = clay, country = country,
                                                    site = site, id = soil_id)
            
            ddsat_soilgrid.add_soil_layers_from_df(subset)

            if codes is not None:
                outputfngroup = outputfn+'{}'.format(codes[i].replace(' ',''))
                outputpathgroup = os.path.join(outputpath,'{}'.format(codes[i].replace(' ','')))
            else:
                outputfngroup = outputfn+'{}'.format(i)
                outputpathgroup = os.path.join(outputpath,'{}'.format(i))    
            if not os.path.exists(outputpathgroup):
                os.mkdir(outputpathgroup)

            fn = os.path.join(outputpathgroup, outputfngroup+'.SOL')
            ddsat_soilgrid.write(fn)
        else:
            print("It must have at least sand and clay values")
    return ddf