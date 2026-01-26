import pandas as pd
import os
import numpy as np
from calendar import monthrange
from datetime import datetime, timedelta
from tqdm import tqdm
import xarray
import concurrent.futures

from ..caf.output import CAFOutputData
from ..dssat.output import DSSATOutputData
from ..simple_model.output import SimpleModelOutputData

monthstring = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ONI_DATA = pd.read_fwf('https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt')


threemonth_dict = {
    'DJF': [12,1,2],
    'JFM': [1,2,3],
    'FMA': [2,3,4],
    'MAM': [3,4,5],
    'AMJ': [4,5,6],
    'MJJ': [5,6,7],
    'JJA': [6,7,8],
    'JAS': [7,8,9],
    'ASO': [8,9,10],
    'SON': [9,10,11],
    'OND': [10,11,12],
    'NDJ': [11,12,1],
}

class ColumnNames():
    """
    check columns names for each model type
    """
    _weather_column = {
            'dssat': {'date':'DATE',
                    'tmax':'TMAX',
                    'tmin': 'TMIN',
                    'srad': 'SRAD',
                    'rain':'RAIN'},
            
            'simple_model': {'date':'DATE',
                    'tmax':'tmax',
                    'tmin': 'tmin',
                    'srad': 'srad',
                    'rain': 'rain'},
            
            'caf': {'date': 'DATE',
                    'tmax':'TMAX',
                    'tmin': 'TMIN',
                    'srad': 'GR',
                    'rain':'RAIN'},
            }
    
    _growth_column = {
            'dssat': {'date':'PDAT',
                    'mdate':'MDAT',
                'hdate':'HDAT',
                'yield':'HWAH',
                'number_of_cycle': 'TRNO'},
            
            'simple_model': {'date':'sowing_date',
                'hdate':'harvesting_date',
                'yield':'crop_yield',
                'number_of_cycle': 'TRNO'},
            
            'caf': {'date': 'HDAT',
                'yield':'harvDM_f_hay',
                'flowering_date': 'DayFl',
                'number_of_cycle': 'n_cycle',
                'hdate': 'DayHarv(1)'
            }
        }
    
    _nitrogen_uptake = {
        'dssat': {'nitrogen_uptake': 'NUCM',
                  'nitrogen_applied': 'NICM'}
    }
    
    def __init__(self, model_name):
        self.name = model_name
        assert model_name in self.avail_models, 'check models name'
    
    @property
    def avail_models(self):
        return list(self._weather_column.keys())
    
    @property    
    def weather_columns(self):        
        return self._weather_column[self.name]
    
    @property
    def growth_colnames(self):
        return self._growth_column[self.name]
    
    def change_weathertomodelstyle(self, rename_to = 'dssat', rename_whichcolumns = None):
        refcolumns = self._weather_column[rename_to]
        rename_whichcolumns = rename_whichcolumns or list(refcolumns.keys())
        src_columns = self.weather_columns
        return {src_columns[v]:refcolumns[v] for v in rename_whichcolumns}
        

def update_data_using_path(path, model = 'dssat'):

    model_class = {'dssat': DSSATOutputData,
                'caf': CAFOutputData,
                'simple_model': SimpleModelOutputData}
    
    assert model in list(model_class.keys()), f"please check model's name it must be {list(model_class.keys())}"

    groupclasses = [
        i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))
    ]

    return {
        groupclasses[i]: model_class[model](os.path.join(path, groupclasses[i]))
        for i in range(len(groupclasses))
    }
    
def export_weather_by_years(weather_data, path):
   
    years = np.unique(weather_data["DATE"].dt.year)
    for year in years:
        wsubset = weather_data.loc[weather_data["DATE"].dt.year == year]
        wsubset.to_csv(os.path.join(path, f'weather_{year}.csv'), index =False )
       
    with open(os.path.join(path, f'weather_years.txt'),'wb') as fn:
        fn.write(', '.join([str(i) for i in years]).encode())


def export_data_ascsv(processed_sims, output_data, crop, tmp_path, model_name, weather_variables2export = ['date', 'tmin', 'tmax', 'rain', 'srad'], group_by = 'texture', export_data = True):
    model_columns = ColumnNames(model_name)
    completedgroups = [k for k,v in processed_sims.items() if v]
    # export weather data
    weather_data = output_data[completedgroups[0]].weather_data()
    if model_name != 'dssat':
        # change simple model column names
        newcolnames = model_columns.change_weathertomodelstyle(rename_whichcolumns=weather_variables2export, rename_to='dssat')
        weather_data = weather_data.rename(columns = newcolnames)
        columnames = [v for v in newcolnames.values()]  
    else:
        columnames = [v for v in model_columns.weather_columns.values()] 
    weather_data = weather_data[columnames]
    if export_data:
        weather_data.to_csv(os.path.join(tmp_path, 'weather.csv'), index =False)
        export_weather_by_years(weather_data, tmp_path)
    
    # export yield data
    date_colname, y_colname = model_columns.growth_colnames['date'], model_columns.growth_colnames['yield']
    potentialyield_data = []
    for gval in completedgroups:
        dftmp = output_data[gval].output_data().sort_values(date_colname)
        if crop != 'coffee': dftmp = dftmp.loc[dftmp[y_colname] > 0]
        dftmp[group_by] = gval
        potentialyield_data.append(dftmp)
 
    columnames = [v for k, v in model_columns.growth_colnames.items()]
    if export_data:
        pd.concat(potentialyield_data)[columnames + [group_by]].to_csv(os.path.join(tmp_path, f'{crop}_potential_yield.csv'))
    else:
        return pd.concat(potentialyield_data)[columnames + [group_by]]
        
    

def oni_season(dates: pd.DatetimeIndex) -> list:
    """
    Determines the ONI season classification for given dates.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Array of dates.

    Returns
    -------
    list
        List of ONI classifications ('La Niña', 'El Niño', or 'Normal') for each date.
    """

    yr = dates.dt.year.values
    month = dates.dt.month.values
    onivalue = []
    for idx in range(month.shape[0]):
        subset_year, subset_month = yr[idx], month[idx]
        oniseason = list(threemonth_dict.keys())[subset_month-1]
        oniyear = ONI_DATA.loc[ONI_DATA.YR == subset_year]
        anomaly = oniyear.loc[oniyear.SEAS == oniseason]['ANOM'].values[0]
        if anomaly<=-0.5: oni = 'La Niña'
        elif anomaly>=0.5: oni = 'El Niño'
        else: oni = 'Normal'
        onivalue.append(oni)
        
    return onivalue

def identify_enso_events(oni_data, target_dates):
    """
    Classifies a list of [Year, Month] pairs based on official ONI event rules.
    """
    # 1. Prepare ONI Data
    df = pd.DataFrame(oni_data)
    
    # Identify qualifying periods (anomalies >= 0.5 or <= -0.5)
    df['is_nino_thresh'] = False
    df.loc[df['ANOM'] >= 0.5, 'is_nino_thresh'] = True
    df['is_nina_thresh'] = False
    df.loc[df['ANOM'] <= -0.5, 'is_nina_thresh'] = True
    
    
    # Function to find streaks of at least 5 consecutive True values
    def get_event_mask(series, min_len=5):
        # Group consecutive identical values
        groups = (series != series.shift()).cumsum()
        # Count sizes and filter for those that meet the streak requirement
        streak_counts = series.groupby(groups).transform('count')
        return series & (streak_counts >= min_len)

    df['Official_Nino'] = get_event_mask(df['is_nino_thresh'])
    df['Official_Nina'] = get_event_mask(df['is_nina_thresh'])
    
    # 2. Map Month to ONI Season Label
    # Seasons are centered on the target month (e.g., March = FMA)
    month_to_season = {
        1: 'DJF', 2: 'JFM', 3: 'FMA', 4: 'MAM', 5: 'AMJ', 6: 'MJJ',
        7: 'JJA', 8: 'JAS', 9: 'ASO', 10: 'SON', 11: 'OND', 12: 'NDJ'
    }
    
    results = []
    for yr, mo in target_dates:
        season_label = month_to_season[mo]
        # Match year and season label in the ONI table
        match = df[(df['YR'] == yr) & (df['SEAS'] == season_label)]
        
        if match.empty:
            phase = "Data Missing"
        elif match.iloc[0]['Official_Nino']:
            phase = "El Niño"
        elif match.iloc[0]['Official_Nina']:
            phase = "La Niña"
        else:
            phase = "Neutral"
        
        results.append([yr, mo, season_label, phase])
        
    return pd.DataFrame(results, columns=['Year', 'Month', 'Season', 'Phase'])

def add_oni_season_to_yield_data(yield_data: pd.DataFrame, group_by = 'TRNO', date_column:str = 'PDAT', yield_column: str = 'HWAM', harvest_column:str = 'HDAT'):
    """
    Adds ONI classification to yield data and summarizes historical, La Niña, and El Niño seasons.

    Parameters
    ----------
    yield_data : pd.DataFrame
        Dataframe containing yield data.
    group_by : str, optional
        Column to group by, by default 'TRNO'.
    date_column : str, optional
        Planting date column, by default 'PDAT'.
    yield_column : str, optional
        Yield column, by default 'HWAM'.
    harvest_column : str, optional
        Harvest date column, by default 'HDAT'.

    Returns
    -------
    pd.DataFrame
        Dataframe with historical ONI classifications added.
    """
    data = yield_data.copy()
    data['oni'] = oni_season(data[date_column])
    
    historical = yield_data_summarized(data, group_by, date_column=date_column, harvest_column=harvest_column,yield_column=yield_column)
    ninasubset = data.loc[data.oni == 'La Niña'].reset_index()
    if 'level_0' in ninasubset.columns:
        ninasubset = ninasubset.drop(columns='level_0')
    historical_nina = yield_data_summarized(ninasubset, 
                                            group_by, date_column=date_column, harvest_column=harvest_column,yield_column=yield_column)
    ninosubset = data.loc[data.oni == 'El Niño'].reset_index()
    if 'level_0' in ninosubset.columns:
        ninosubset = ninosubset.drop(columns='level_0')
    historical_nino = yield_data_summarized(ninosubset, 
                                            group_by, date_column=date_column, harvest_column=harvest_column,yield_column=yield_column)
    
    historical_nina['month_day'] = historical['month_day'][:historical_nina.shape[0]]
    historical_nina[f'{date_column.lower()}_year_month_day'] = historical[f'{date_column.lower()}_year_month_day'][:historical_nina.shape[0]]
    historical_nina[f'{harvest_column.lower()}_year_month_day'] = historical[f'{harvest_column.lower()}_year_month_day'][:historical_nina.shape[0]]
    historical_nino['month_day'] = historical['month_day'][:historical_nino.shape[0]]
    historical_nino[f'{date_column.lower()}_year_month_day'] = historical[f'{date_column.lower()}_year_month_day'][:historical_nino.shape[0]]
    historical_nino[f'{harvest_column.lower()}_year_month_day'] = historical[f'{harvest_column.lower()}_year_month_day'][:historical_nino.shape[0]]
    
    historical['oni'] = 'Histórico'
    historical_nina['oni'] = 'La Niña'
    historical_nino['oni'] = 'El Niño'

    df = pd.concat([historical,historical_nino,historical_nina])
    
    return df


def from_doy_to_date(doy, ref_year):
    return pd.to_datetime(f'{ref_year}-01-01') + pd.to_timedelta(doy - 1, unit='d')

def two_digit_format(value):
    return f'0{value}' if value < 10 else str(value)


def get_dummy_crop_cycle_dates(crop_data, group_by, date_column, harvest_column=None, initial_year = None):
    init_year = initial_year or 2000
    doy_prev = 0
    dates = {}
    for i in np.sort(np.unique(crop_data[group_by])):

        subsettrno = crop_data.loc[crop_data[group_by] == i].reset_index()
        doy = int(np.mean(subsettrno[date_column].dt.day_of_year))
        if harvest_column is not None:
            crop_cycle_days = int(np.mean(subsettrno[harvest_column] - subsettrno[date_column]).days)
        else:
            crop_cycle_days = 0
        if doy<(doy_prev-10):
            init_year+=1
        pdat = datetime.strptime(f'{init_year}-{doy}', "%Y-%j")
        hdat = pdat + timedelta(days=crop_cycle_days)
        dates[i] = [pdat, hdat]
        doy_prev= doy
    df = pd.DataFrame(dates).T.reset_index()
    if harvest_column is not None:
        return df.rename(columns={'index':group_by, 0: f'{date_column.lower()}_year_month_day', 
                            1: f'{harvest_column.lower()}_year_month_day'}).sort_values([group_by]).reset_index()
    else:
        return df.rename(columns={'index':group_by, 0: f'{date_column.lower()}_year_month_day'}).sort_values([group_by]).reset_index()
    
def summarize_dates_bygroup(yield_data, group_by:str, date_column:str = 'PDAT', harvest_column:str = 'HDAT', refplanting_year = None, refharvesting_year = None):
    
    datasummarised = get_dummy_crop_cycle_dates(yield_data, group_by=group_by, 
                                                date_column=date_column, harvest_column=harvest_column, initial_year=refplanting_year)
    
    datasummarised['month_day'] = ['{}-{}'.format(
        two_digit_format(month),
        two_digit_format(day),
        ) for month, day in zip(datasummarised[f'{date_column.lower()}_year_month_day'].dt.month,
                                datasummarised[f'{date_column.lower()}_year_month_day'].dt.day)]
    
    return datasummarised


def yield_data_summarized(yield_data, groupp_by = 'TRNO', date_column:str = 'PDAT', 
                        yield_column: str = 'HWAM', harvest_column = 'HDAT', refplanting_year = None, refharvesting_year = None):
    data = yield_data.copy()
    #get dates
    historical = summarize_dates_bygroup(data, groupp_by, date_column=date_column, harvest_column=harvest_column,
                                        refplanting_year = refplanting_year,refharvesting_year= refharvesting_year)
    
    #get averaged yield
    summarizedyield_data = data.groupby([groupp_by]).aggregate({yield_column:'mean'}).reset_index().sort_values([groupp_by])[yield_column]
    historical[yield_column] = summarizedyield_data
    
    #add variance
    std_val = data.groupby([groupp_by]).aggregate({yield_column:'std'}).reset_index().sort_values([groupp_by])[yield_column]
    historical["y_upper"] = summarizedyield_data + std_val
    historical["y_lower"] = summarizedyield_data - std_val
    
    return historical


def coffee_yield_data_summarized(yield_data, date_column:str = 'HDAT', yield_column: str = 'harvDM_f_hay', n_cycle_column = 'n_cycle', harvest_column = None):
    
    datatoconcatenate = []
    for n_cycle, subset_cycle in yield_data.groupby([n_cycle_column]):
        if isinstance(subset_cycle[date_column].values[0], str):
            subset_cycle['date'] = subset_cycle[date_column].apply(lambda x:  datetime.strptime(x, '%Y-%m-%d'))
        else:
            subset_cycle['date'] = subset_cycle[date_column]
            
        years = subset_cycle['date'].dt.year
        start_year, end_year = np.min(years), np.max(years)
        
        subset_cycle['nyear_month_day'] = ['{}-{}-{}'.format(
            '200{}'.format(year-start_year) if (year-start_year)<=9 else '20{}'.format(year-start_year),
            '0{}'.format(month) if month<=9 else month,
            '0{}'.format(day) if day<=9 else day
            ) for year, month,day in zip(subset_cycle['date'].dt.year,
                                        subset_cycle['date'].dt.month, subset_cycle['date'].dt.day)]
        
        daytoflower = subset_cycle['DayFl']*np.max(subset_cycle[yield_column])//2
        daytoflower[daytoflower == 0] = np.nan
        subset_cycle['daytoflower'] = daytoflower
        if harvest_column:
            harvestday = subset_cycle[harvest_column]*np.max(subset_cycle[yield_column])//2
            harvestday[harvestday == 0] = np.nan
            subset_cycle['hdat'] = harvestday
        subset_cycle['period'] = f'{start_year} - {end_year}'
        datatoconcatenate.append(subset_cycle)
            
    return pd.concat(datatoconcatenate)

## spatial pixeldata
def spatial_dummy_crop_cycle_dates(dates_grouped_by_treatment, initial_year = None):
    date_column = 'PDAT'
    init_year = initial_year or 2000
    doy_prev = 0
    dates = {}
    #dates_grouped_by_treatment = dim_dates.reshape(dim_dates.shape[0]//plantingWindow, plantingWindow).T
    for i in range(dates_grouped_by_treatment.shape[0]):
        subsettrno = dates_grouped_by_treatment[i].copy()

        doy = int(np.mean(pd.Series(subsettrno).dt.day_of_year))

        if doy<(doy_prev-10):
            init_year+=1
        pdat = datetime.strptime(f'{init_year}-{doy}', "%Y-%j")
        dates[i] = [pdat]
        doy_prev= doy
    df = pd.DataFrame(dates).T.reset_index()

    return df.rename(columns={'index':'TRNO', 0: f'{date_column.lower()}_year_month_day'}).reset_index()[f'{date_column.lower()}_year_month_day'].values


def summarize_spatial_yields_by_time_window(xrdata, plantingWindow, target_dim_name = 'HWAH', date_dim_name = 'date', initial_year:int = 2000):
    
    dim_dates = xrdata[date_dim_name].values
    
    dates_grouped_by_treatment = dim_dates.reshape(dim_dates.shape[0]//plantingWindow, plantingWindow).T
    newdates = spatial_dummy_crop_cycle_dates(dates_grouped_by_treatment, initial_year = initial_year)
    target_vals = [xrdata[target_dim_name].sel(date = dates_grouped_by_treatment[i]).mean(dim = [date_dim_name]).values for i in range(dates_grouped_by_treatment.shape[0])]
    
    historic_spatial_yield = xarray.Dataset(
    {
        "HWAH": (("date","y", "x"), np.array(target_vals)),
    },
    coords={"x": xrdata.x.values, "y": xrdata.y.values, "date": newdates},
    )
    historic_spatial_yield.attrs= xrdata.attrs
    historic_spatial_yield.attrs['count'] = 1
    
    return historic_spatial_yield

## summarize weather
def summarize_event_weather(weather, init_dates: pd.Series, end_dates: pd.Series, weathercol_index):

    minlong = int(((end_dates -  init_dates)/np.timedelta64(1, 'D')).min())

    data = np.zeros((len(init_dates), len(weathercol_index), minlong))
    for i in range(len(init_dates)):
        pdate = init_dates.values[i]
        hdate = end_dates.values[i]
        data[i] = weather.values[np.logical_and(weather['DATE']>=pdate,weather['DATE']<=hdate)][:minlong,weathercol_index].swapaxes(0,1)

    return data.mean(axis=0)

def get_weather_event(yield_data, weather, summarized_dates, group_id, group_by, weathercol_index, date_column='PDAT', harvest_column = 'HDAT'):
    
    subset_values = yield_data.loc[yield_data[group_by] == group_id]
    if subset_values.shape[0]>1:
        weather_summarized = summarize_event_weather(weather, subset_values[date_column],subset_values[harvest_column], weathercol_index=weathercol_index)
        subgroudates = summarized_dates.loc[summarized_dates[group_by] == group_id]
        pdat = subgroudates[f'{date_column.lower()}_year_month_day'].values[0]
        

        event_weather = pd.DataFrame(weather_summarized.swapaxes(0,1), columns=weather.columns[weathercol_index])
        hdat = hdat = pdat + np.timedelta64(event_weather.shape[0], 'D')
        event_weather['DATE'] = pd.date_range(start=pdat,end=hdat)[:event_weather.shape[0]]
        event_weather[group_by] = int(group_id)
        event_weather['month_day'] = subgroudates['month_day'].values[0]
        return event_weather


def get_summarized_weather_by_treatment(yield_data, weather,group_dates_by,  
                                        weather_variables = None, date_column = 'PDAT',
                                        harvest_column = 'HDAT', ncores = 0):
    
    weather_variables = weather_variables or weather.columns[1:]
    pos_weathervar_oi = [list(weather.columns).index(col) for col in weather_variables]

    tr_o_dates = summarize_dates_bygroup(yield_data, group_dates_by, date_column, harvest_column)
    
    events_weather_list = []
    date_groups = np.unique(yield_data[group_dates_by])
    
    if ncores == 0:
        for i in tqdm(range(len(date_groups))):
            event_weather = get_weather_event(yield_data, weather, tr_o_dates, date_groups[i],group_dates_by, pos_weathervar_oi,date_column, harvest_column)
            events_weather_list.append(event_weather)
    else:
        with tqdm(total=len(date_groups)) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
                future_to_day ={executor.submit(get_weather_event, yield_data, weather, tr_o_dates, date_groups[i],group_dates_by,
                                                pos_weathervar_oi,date_column, harvest_column): (i) for i in range(len(date_groups))}
                
                for future in concurrent.futures.as_completed(future_to_day):
                    igroup = future_to_day[future]
                    try:
                        events_weather_list.append(future.result())
                    except Exception as exc:
                        print(f"Request for treatment {igroup} generated an exception: {exc}")
                    pbar.update(1)
    
    return pd.concat([subset for subset in events_weather_list if subset is not None]).sort_values([group_dates_by])

