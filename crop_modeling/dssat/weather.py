
import pandas as pd
import numpy as np
import pandas as pd
import os
from DSSATTools.weather import Weather
from ..utils.u_weather import monthly_amplitude

class DSSAT_Weather(Weather):
    """
    A class representing DSSAT Weather data.

    This class is responsible for processing weather data in the format expected by the DSSAT model.
    It takes a DataFrame, renames columns, calculates long-term average temperature (TAV), 
    and temperature amplitude (AMP), and initializes the Weather class.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing weather data.
    column_names : dict
        Dictionary mapping DSSAT column names to DataFrame column names.
    **kwargs : dict, optional
        Additional parameters to pass to the parent class `Weather`.

    Attributes
    ----------
    _df : pd.DataFrame
        The DataFrame with renamed columns and selected weather data.
    """
    @staticmethod
    def get_dates_from_file(file_path):
        assert file_path.endswith('WTH'), 'the file is not a DSSAT compatible format file'
        with open(file_path, 'r', encoding="utf-8") as fn:
            
            for i, line in enumerate(fn):
                stiped_line = line.strip()
                if stiped_line.startswith('@') and 'DATE' in stiped_line:
                    break
            lines = fn.readlines()
            
            return [lines[j].split(' ')[0] for j in range(i+1, len(lines))]
        
    def __init__(self, df:  pd.DataFrame, column_names: dict, **kwargs):
        """
        Initialize the DSSATWeather class.

        This method renames the columns of the input DataFrame using the `column_names` mapping, 
        calculates the mean longitude and latitude, computes temperature amplitude (AMP) 
        and long-term average temperature (TAV), and initializes the parent Weather class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing weather data.
        column_names : dict
            Dictionary where keys are DSSAT weather parameter names, 
            and values are the corresponding DataFrame column names.
        **kwargs : dict, optional
            Additional parameters passed to the parent class `Weather`.
        """
        # Reverse column_names dictionary
        column_names = {v:k for k, v in column_names.items()}
        self._df = df.rename(columns=column_names)
        lon = self._df['LON'].values.mean()
        lat = self._df['LAT'].values.mean()

        self._df = self._df[[v for k,v in column_names.items()]]
        self._df = self._df.drop(['LON','LAT'],axis=1)
        pars = {i:i for i in self._df.columns}

        amp = self.calculate_temperature_amplitude()
        tav = self.calculate_long_term_average_temp()
    
        super().__init__(self._df, pars, lon = lon, lat = lat,amp = amp, tav =tav, **kwargs)

    def calculate_long_term_average_temp(self, tmax_name: str = 'TMAX', tmin_name: str = 'TMIN') -> float:
        """
        Calculate long-term average temperature (TAV).

        This method computes the average of daily maximum and minimum temperatures.

        Parameters
        ----------
        tmax_name : str, optional
            The column name for maximum temperature. Default is 'TMAX'.
        tmin_name : str, optional
            The column name for minimum temperature. Default is 'TMIN'.

        Returns
        -------
        float
            The long-term average temperature (TAV).
        """
        return ((self._df[tmax_name].values + self._df[tmin_name].values)/2).mean()

    def calculate_temperature_amplitude(self, tmax_name: str = 'TMAX', tmin_name: str = 'TMIN', date_name: str = 'DATE') -> float:
        """
        Calculate temperature amplitude (AMP).

        This method computes the temperature amplitude, which is the half-difference 
        between the maximum and minimum monthly averages of temperature over time.

        Parameters
        ----------
        tmax_name : str, optional
            The column name for maximum temperature. Default is 'TMAX'.
        tmin_name : str, optional
            The column name for minimum temperature. Default is 'TMIN'.
        date_name : str, optional
            The column name for dates. Default is 'DATE'.

        Returns
        -------
        float
            The temperature amplitude (AMP).
        
        Raises
        ------
        AssertionError
            If the `date_name` column is not of type `np.datetime64`.
        """
        assert isinstance(self._df[date_name].values[0], np.datetime64)
        month = self._df[date_name].map(lambda x: str(x.month)).values

        fdgrouped = self._df[[tmax_name, tmin_name]].groupby(month)
        
        ampvals = fdgrouped.apply(monthly_amplitude).avgm.values
        amp = (ampvals.max()-ampvals.min())/2

        return float(amp)



