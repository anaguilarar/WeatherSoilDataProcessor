import pandas as pd
import os
import numpy as np
from math import ceil
from calendar import monthrange
from datetime import datetime
from tqdm import tqdm
import concurrent.futures

monthstring = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

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
    oni_data = pd.read_fwf('https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt')

    yr = dates.dt.year.values
    month = dates.dt.month.values
    onivalue = []
    for idx in range(month.shape[0]):
        subset_year, subset_month = yr[idx], month[idx]
        oniseason = list(threemonth_dict.keys())[subset_month-1]
        oniyear = oni_data.loc[oni_data.YR == subset_year]
        anomaly = oniyear.loc[oniyear.SEAS == oniseason]['ANOM'].values[0]
        if anomaly<=-0.5: oni = 'La Niña'
        elif anomaly>=0.5: oni = 'El Niño'
        else: oni = 'Normal'
        onivalue.append(oni)
        
    return onivalue


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

    historical_nina = yield_data_summarized(data.loc[data.oni == 'La Niña'].reset_index(), 
                                            group_by, date_column=date_column, harvest_column=harvest_column,yield_column=yield_column)
    
    historical_nino = yield_data_summarized(data.loc[data.oni == 'El Niño'].reset_index(), 
                                            group_by, date_column=date_column, harvest_column=harvest_column,yield_column=yield_column)
    
    historical_nina['month_day'] = historical['month_day'][:historical_nina.shape[0]]
    historical_nina['pdat_year_month_day'] = historical['pdat_year_month_day'][:historical_nina.shape[0]]
    historical_nina['hdat_year_month_day'] = historical['hdat_year_month_day'][:historical_nina.shape[0]]
    historical_nino['month_day'] = historical['month_day'][:historical_nino.shape[0]]
    historical_nino['pdat_year_month_day'] = historical['pdat_year_month_day'][:historical_nino.shape[0]]
    historical_nino['hdat_year_month_day'] = historical['hdat_year_month_day'][:historical_nino.shape[0]]
    
    historical['oni'] = 'Histórico'
    historical_nina['oni'] = 'La Niña'
    historical_nino['oni'] = 'El Niño'

    df = pd.concat([historical,historical_nino,historical_nina])
    
    return df


def add_year_and_doy(data:pd.DataFrame, date_column:str, format:str = '%Y-%m-%d') -> pd.DataFrame:
    """
    Adds year and day-of-year (DOY) columns to a dataframe based on a date column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing date information.
    date_column : str
        Name of the date column.
    format : str, optional
        Format of the date column if not in datetime format, by default '%Y-%m-%d'.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with year and DOY columns added.
    """
    
    if not isinstance(data[date_column].values[0], np.datetime64):
        data[date_column] = data[date_column].apply(lambda x: datetime.strptime(x, format))
    
    data[f'{date_column}year'] = data[date_column].dt.year
    data[f'{date_column}doy'] = data[date_column].dt.dayofyear.astype(int)
    return data

def from_doy_to_date(doy, ref_year):
    return pd.to_datetime(f'{ref_year}-01-01') + pd.to_timedelta(doy - 1, unit='d')

def two_digit_format(value):
    return f'0{value}' if value < 10 else str(value)


def convert_year_doy_2date(data:pd.DataFrame, date_column:str, harvest_column:str, refplanting_year:int, refharvesting_year:int, maxups:int = 1):
    """
    Converts year and DOY columns to full date format.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing year and DOY columns.
    date_column : str
        Planting date column.
    harvest_column : str
        Harvest date column.
    refplanting_year : int
        Reference year for planting dates.
    refharvesting_year : int
        Reference year for harvesting dates.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with full date columns added.
    """
    pdat, hdat = [], []
    total_adds =0 
    for i in range(data.shape[0]):
        doy = int(data.iloc[i][f'{date_column}doy'])
        hdatdoy = int(data.iloc[i][f'{harvest_column}doy'])
        if i>0 and doy < int(data.iloc[i-1][f'{date_column}doy']-200):
            if total_adds <= maxups:
                refplanting_year +=1
                total_adds +=1 
        if i>0 and hdatdoy < int(data.iloc[i-1][f'{harvest_column}doy']-200):
            if total_adds <= maxups:
                refharvesting_year +=1
        pdat.append(from_doy_to_date(doy, refplanting_year))
        hdat.append(from_doy_to_date(hdatdoy, refharvesting_year)) 

    data[f'{date_column.lower()}_year_month_day'] = pdat
    data[f'{harvest_column.lower()}_year_month_day'] = hdat
    
    return data

def summarize_dates_bygroup(yield_data, group_by:str, date_column:str = 'PDAT', harvest_column:str = 'HDAT', refplanting_year = None, refharvesting_year = None):
    yield_data = add_year_and_doy(yield_data,date_column)
    if refplanting_year is None:
        refplanting_year = yield_data.loc[yield_data[group_by] == np.unique(yield_data[group_by].values)[0]][f'{date_column}year'].values[0]
    yield_data = add_year_and_doy(yield_data,harvest_column)
    if refharvesting_year is None:
        refharvesting_year = yield_data.loc[yield_data[group_by] == np.unique(yield_data[group_by].values)[0]][f'{harvest_column}year'].values[0]

    datasummarised = yield_data.groupby([group_by]).aggregate({
                        f'{harvest_column}doy': 'mean', f'{date_column}doy': 'mean'}).reset_index(
                            ).sort_values([group_by])
    
    datasummarised = convert_year_doy_2date(datasummarised, date_column, harvest_column, refplanting_year, refharvesting_year)

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


def coffee_yield_data_summarized(yield_data, date_column:str = 'HDAT', yield_column: str = 'harvDM_f_hay', n_cycle_column = 'n_cycle'):
    datatoconcatenate = []
    for n_cycle, subset_cycle in yield_data.groupby([n_cycle_column]):
        
        subset_cycle['date'] = subset_cycle[date_column].apply(lambda x:  datetime.strptime(x, '%Y-%m-%d'))
        years = subset_cycle['date'].dt.year
        start_year, end_year = np.min(years), np.max(years)
        
        subset_cycle['nyear_month_day'] = ['{}-{}-{}'.format(
            '200{}'.format(year-start_year) if (year-start_year)<9 else '00{}'.format(year-start_year),
            '0{}'.format(month) if month<9 else month,
            '0{}'.format(day) if day<9 else day
            ) for year, month,day in zip(subset_cycle['date'].dt.year,
                                        subset_cycle['date'].dt.month, subset_cycle['date'].dt.day)]
        
        daytoflower = subset_cycle['DayFl']*np.max(subset_cycle[yield_column])//2
        daytoflower[daytoflower == 0] = np.nan
        subset_cycle['daytoflower'] = daytoflower
        subset_cycle['period'] = f'{start_year} - {end_year}'
        datatoconcatenate.append(subset_cycle)
            
    return pd.concat(datatoconcatenate)


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
        pdat = subgroudates['pdat_year_month_day'].values[0]
        hdat = subgroudates['hdat_year_month_day'].values[0]

        event_weather = pd.DataFrame(weather_summarized.swapaxes(0,1), columns=weather.columns[weathercol_index])
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

