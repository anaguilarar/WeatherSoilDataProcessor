from rosetta import SoilData, rosetta
import numpy as np
from .dssat_transform import DSSATSoil_fromSOILGRIDS, DSSAT_Weather

from tqdm import tqdm
import os
import pandas as pd

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


def find_soil_textural_class(sand,clay):
    """
    Function that returns the USDA-NRCS soil textural class given 
    the percent sand and clay.
    
    Parameters:
    sand (float, integer): Sand content as a percentage  
    clay (float, integer): Clay content as a percentage
    
    Returns:
    string: One of the 12 soil textural classes
    
    Authors:
    Andres Patrignani
    
    Date created:
    12 Jan 2024
    
    Source:
    E. Benham and R.J. Ahrens, W.D. 2009. 
    Clarification of Soil Texture Class Boundaries. 
    Nettleton National Soil Survey Center, USDA-NRCS, Lincoln, Nebraska.
    adapted: https://soilwater.github.io/pynotes-agriscience/exercises/soil_textural_class.html
    """
    
    if not isinstance(sand, (int, float, np.int64)):
        raise TypeError(f"Input type {type(sand)} is not valid.")

    try:
        # Determine silt content
        silt = 100 - sand - clay
        
        if sand + clay > 100:
            raise Exception('Inputs add over 100%')
        elif sand < 0 or clay < 0:
            raise Exception('One or more inputs are negative')
            
    except ValueError as e:
        return f"Invalid input: {e}"
    
    # Classification rules
    if silt + 1.5*clay < 15:
        textural_class = 'sand'

    elif silt + 1.5*clay >= 15 and silt + 2*clay < 30:
        textural_class = 'loamy sand'

    elif (clay >= 7 and clay < 20 and sand > 52 and silt + 2*clay >= 30) or (clay < 7 and silt < 50 and silt + 2*clay >= 30):
        textural_class = 'sandy loam'

    elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
        textural_class = 'loam'

    elif (silt >= 50 and clay >= 12 and clay < 27) or (silt >= 50 and silt < 80 and clay < 12):
        textural_class = 'silt loam'

    elif silt >= 80 and clay < 12:
        textural_class = 'silt'

    elif clay >= 20 and clay < 35 and silt < 28 and sand > 45:
        textural_class = 'sandy clay loam'

    elif clay >= 27 and clay < 40 and sand > 20 and sand <= 45:
        textural_class = 'clay loam'

    elif clay >= 27 and clay < 40 and sand <= 20:
        textural_class = 'silty clay loam'

    elif clay >= 35 and sand > 45:
        textural_class = 'sandy clay'

    elif clay >= 40 and silt >= 40:
        textural_class = 'silty clay'

    elif clay >= 40 and sand <= 45 and silt < 40:
        textural_class = 'clay'

    else:
        textural_class = 'unknown' # in case we failed to catch any errors earlier

    return textural_class


def slu1(clay:float, sand:float) -> float:
    """
    Calculate Stage 1 evaporation limit (slu), in mm.

    Parameters
    ----------
    clay : float
        Percentage of clay in the soil.
    sand : float
        Percentage of sand in the soil.

    Returns
    -------
    float
        The Stage 1 evaporation limit in millimeters.

    Raises
    ------
    ValueError
        If the clay or sand percentage is not in the range of 0 to 100.
    """
    if not (0 <= clay <= 100):
        raise ValueError("Clay percentage must be between 0 and 100.")
    if not (0 <= sand <= 100):
        raise ValueError("Sand percentage must be between 0 and 100.")
    

    if sand>=80:
        lu = 20-0.15*sand
    elif clay>=50:
        lu = 11 - 0.06*clay
    else:
        lu = 8-0.08*clay
    return lu

## 'Sat. hydraulic conductivity, macropore, cm h-1'
def calculate_sks(sand: float = None, 
                  silt: float = None, 
                  clay: float = None, 
                  bulk_density: float = None, 
                  field_capacity: float = None, 
                  permanent_wp: float = None) -> float:
    """
    Calculate Saturated Hydraulic Conductivity (Ks) in cm/h based on soil properties.

    Parameters
    ----------
    sand : float, optional
        Percentage of sand in the soil (0-100).
    silt : float, optional
        Percentage of silt in the soil (0-100).
    clay : float, optional
        Percentage of clay in the soil (0-100).
    bulk_density : float, optional
        Bulk density of the soil in g/cm³.
    field_capacity : float, optional
        Field capacity of the soil in cm³/cm³.
    permanent_wp : float, optional
        Permanent wilting point of the soil in cm³/cm³.

    Returns
    -------
    float
        Estimated saturated hydraulic conductivity (Ks) in cm/h.

    Raises
    ------
    ValueError
        If not enough information is provided to calculate sand, silt, or clay percentages.
    """
    
    if sand is None and silt is not None and clay is not None:
        sand = 100 - silt - clay
    elif silt is None and sand is not None and clay is not None:
        silt = 100 - sand - clay 
    elif clay is None and sand is not None and silt is not None:
        clay = 100 - sand - silt  

    elif sand is None or silt is None or clay is None:
        raise ValueError("At least two of sand, silt, or clay percentages must be provided.")
    
    datinfo =  [i for i in [sand, silt,clay, bulk_density, field_capacity, permanent_wp] if i]
    
    soil_data = SoilData.from_array(
                [datinfo]
            )
    
    vangenuchten_pars, _, _ = rosetta(3, soil_data)

    ks = (10**vangenuchten_pars[0][-1]) # Convert log(Ks) to Ks in cm/day / 24

    ks = ks / 24  # Convert cm/day to cm/hour
    return ks




def from_weather_to_dssat(xrdata, groupby: str = None, date_name ='date', 
                          params_df_names = None,refht = 2, outputpath = None, outputfn = None, codes = None):
    
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
    dfdataper_date = []
    weather_df = xrdata.to_dataframe().reset_index()

    for d in tqdm(np.unique(weather_df[date_name].values)):
        ddf = weather_df.loc[weather_df[date_name] == d]

        if groupby:
            ddf = ddf.groupby([groupby], dropna = True).agg(weatherdatavars).reset_index()
        else:
            ddf['tmp'] = 0
            groupby= 'tmp'
            ddf = weather_df.dropna().reset_index()
        ddf['date'] = d
        dfdataper_date.append(ddf)

    weather_df = pd.concat(dfdataper_date)
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
                        country = None,site = None, outputpath = None, outputfn = None, codes = None):
    
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
        
        sand = check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].sand.values))
        clay = check_percentage(np.nanmean(subset.loc[subset[depth_name]  == firstdepth[0]].clay.values))
        print(f"sand {sand} clay {clay}")
        if not np.isnan(clay) and not np.isnan(sand):
            long = subset.loc[subset[depth_name] == firstdepth[0]].x.values.mean()
            lat = subset.loc[subset[depth_name] == firstdepth[0]].y.values.mean()
        
            ddsat_soilgrid = DSSATSoil_fromSOILGRIDS(long = long, lat = lat, sand = sand, clay = clay, country = country, site = site)
            
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