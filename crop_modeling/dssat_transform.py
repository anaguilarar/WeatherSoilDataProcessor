
from .soil_funs import find_soil_textural_class, calculate_sks, slu1
import pandas as pd
from DSSATTools.weather import Weather
from DSSATTools.soil import SoilProfile, SoilLayer

import numpy as np


class DSSATSoil_fromSOILGRIDS(SoilProfile):
    """
    DSSAT Soil profile class built from SoilGrids data.

    This class is used to construct soil profiles for DSSAT from SoilGrids data. 
    It provides methods for mapping soil data, calculating hydraulic conductivity, 
    and setting general soil and location parameters.
    
    Attributes
    ----------
    GENERAL_DESCRIPTION : dict
        Descriptions for each soil parameter.
    SOILGRID_MAPPING : dict
        Mapping of SoilGrid parameters to DSSAT parameters and conversion factors.
    ALBEDOS : tuple
        Albedo values for various texture classes.
    RUNOFF_CURVES : tuple
        Runoff curve values for various texture classes.
    TEXTURE_CLASSES : tuple
        Textural classes of the soil.
    SUM_TEXTURE_CLASSES : tuple
        DSSAT textural classes for summary purposes.
    """

    GENERAL_DESCRIPTION = {
        'SLLL': 'Lower limit of plant extractable soil water, cm3 cm-3',
        'SDUL': 'Drained upper limit, cm3 cm-3',
        'SSAT': 'Upper limit, saturated, cm3 cm-3',
        'SRGF': 'Root growth factor, soil only, 0.0 to 1.0',
        'SBDM': 'Bulk density, moist, g cm-3',
        'SLOC': 'Organic carbon, %',
        'SLCL': 'Clay (<0.002 mm), %',
        'SLSI': 'Silt (0.05 to 0.002 mm), %',
        'SLCF': 'Coarse fraction (>2 mm), %',
        'SLNI': 'Total nitrogen, %',
        'SLHW': 'Soil pH',
        'SMHB': 'pH in buffer determination method, code',
        'SCEC': 'Cation exchange capacity, cmol kg-1',
        'SADC': 'Anion adsorption coefficient, cm3 (H2O) g [soil]-1',
        'SSKS': 'Saturated hydraulic conductivity, macropore, cm h-1'
    }

    SOILGRID_MAPPING = {
        'clay': ['SLCL', 0.01],
        'silt': ['SLSI', 0.01],
        'nitrogen': ['SLNI', 0.01],
        'bdod': ['SBDM', 0.01],
        'cfvo': ['SLCF', 0.01],
        'phh2o': ['SLHW', 0.1],
        'soc': ['SLOC', 0.01],
        'cec': ['SCEC', 0.1],
        'wv1500': ['SLLL', 0.001],
        'wv0033': ['SDUL', 0.001],
        'wv0010': ['SSAT', 0.001]
    }

    ALBEDOS = (0.12, 0.12, 0.13, 0.13, 0.12, 0.13, 0.13, 0.14, 0.13, 0.13, 0.16, 0.19, 0.13)
    RUNOFF_CURVES = (73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 68.0, 73.0, 68.0, 68.0, 73.0)
    TEXTURE_CLASSES = ("clay", "silty clay", "sandy clay", "clay loam", "silty clay loam", "sandy clay loam", "loam", 
                    "silty loam", "sandy loam", "silt", "loamy sand", "sand", "unknown")
    
    SUM_TEXTURE_CLASSES = ("C", "SIC", "SC", "CL", "SICL", "SCL", "L", "SIL", "SL", "SI", "LS", "S", "unknown")

    @staticmethod
    def get_init_from_texture(sand: float, clay: float):
        """
        Find the soil texture class based on sand and clay content.

        Parameters
        ----------
        sand : float
            Sand content as a percentage.
        clay : float
            Clay content as a percentage.

        Returns
        -------
        str
            The soil texture class.
        """
        return find_soil_textural_class(sand, clay)
    
    def soilgrid_dict(self, dict_names  = None):
        """
        SoilGrid convention compared with DSSAT.
        
        Parameters
        ----------
        dict_names : dict, optional
            A dictionary mapping SoilGrid variables to DSSAT variables. If not provided, the default mapping is used.

        Returns
        -------
        dict
            The mapping dictionary used for converting SoilGrid data to DSSAT format.
        """

        if dict_names:
            self._dict_names = dict_names
        else:
            self._dict_names = {'clay': ['SLCL', 0.1],
         'silt': ['SLSI', 0.1],
         'nitrogen': ['SLNI', 0.01],
         'bdod': ['SBDM', 0.01],
         'cfvo': ['SLCF', 0.01],
         'phh2o': ['SLHW', 0.1],
         'soc': ['SLOC', 0.01],
         'cec': ['SCEC', 0.1],
         'wv1500': ['SLLL',0.001],
         'wv0033': ['SDUL',0.001],
         'wv0010': ['SSAT',0.001]}
        
        return self._dict_names

    def from_df_to_dssat_list(self, df: pd.DataFrame, depth_col_name: str = 'depth') -> list:
        """
        Convert soil data from a DataFrame to a DSSAT-compatible format.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing soil data.
        depth_col_name : str, optional
            The name of the column containing depth information. Default is 'depth'.

        Returns
        -------
        list
            A list of soil layers in DSSAT-compatible format.
        """
        colnames = df.columns
        soillayers = []
        for row in range(df.shape[0]):
            dict_dssat = {}
            for col in list(self._dict_names.keys()):
                kdssat, fac = self._dict_names[col]
                if col not in colnames:
                    dict_dssat[kdssat]  = None
                else:
                    #if np.max(df[col].values[row])>1/fac:
                    val = fac*df[col].values[row]
                    #else:
                    #    val = df[col].values[row]
                        
                    dict_dssat[kdssat]  = val

            dict_dssat['SSKS'] = calculate_sks(
                            silt = dict_dssat['SLSI'], 
                            clay = dict_dssat['SLCL'], 
                            bulk_density = dict_dssat.get('SBDM',None), 
                            field_capacity = dict_dssat.get('SDUL',None),
                            permanent_wp = dict_dssat.get('SLLL',None))

            depth = df[depth_col_name].values[row]
            depth = int(depth.split('-')[1]) if isinstance(depth, str) else depth

            soillayers.append((depth, dict_dssat))
        return soillayers
    
    def add_soil_layers_from_df(self, df: pd.DataFrame, depth_col_name: str = 'depth') -> list:
        """
        Add soil layers from a DataFrame and convert them to DSSAT format.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing soil data.
        depth_col_name : str, optional
            The column name for depth. Default is 'depth'.

        Returns
        -------
        list            List of soil layers added to the profile.
        """
        #soil.SoilLayer(20, {'SLCL': 50, 'SLSI': 45}),
        listvals = self.from_df_to_dssat_list(df, depth_col_name= depth_col_name)
        ## sort depths 
        sortedbydepth = np.argsort([depth for depth,val in listvals])
        listvals = [listvals[i] for i in sortedbydepth]
        soillayers = [SoilLayer(d, v) for d,v in listvals]
        for layer in soillayers: self.add_layer(layer)
        
        return listvals
            
    def soil_general_soilline_parameters(self):
        """
        Set general soil parameters based on soil texture.

        This method sets parameters like albedo, root growth factor, runoff curve, 
        and pH buffer method based on the soil's texture class.
        """
        self._texture = self.get_init_from_texture(self._sand, self._clay)

        self._pos_text = self.TEXTURE_CLASSES.index(self._texture)

        self.SALB = self.ALBEDOS[self._pos_text]

        self.SLU1 = slu1(self._clay, self._sand)

        #self.SLDR = 0.5
        self.SLRO = self.RUNOFF_CURVES[self._pos_text]
        self.SLNF = 1.
        self.SLPF =0.92
        self.SMHB,  self.SMPX,  self.SMKE = "IB001", "IB001", "IB001"

    def set_general_location_parameters(self, **kwargs):
        """
        Set general location parameters such as description, country, site, latitude, and longitude.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional location parameters including 'description', 'country', 'site', 'lat', and 'lon'.
        """
        self.description = kwargs.get('description',None)
        self.description = "ISRIC V2 {} {}".format(self.SUM_TEXTURE_CLASSES[self._pos_text],
                                                                self._texture) if self.description is None else self.description
        
        self.country = kwargs.get('country', 'COL')
        self.site = kwargs.get('site', 'CAL')
        self.lat = kwargs.get('lat', -99)
        self.lon = kwargs.get('lon', -99)

        self.id = kwargs.get('id', None)
        self.id = f'{self.country}-{self.site}'.replace(' ', '') if self.id is None else self.id


    def __init__(self, **kwargs) -> None:
        """
        Initialize the DSSAT soil profile from SoilGrids data.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional soil profile parameters like sand, clay, and silt content.
        """

        self._sand = kwargs.get('sand',None)
        self._clay = kwargs.get('clay',None)
        self._silt = kwargs.get('silt',None)

        super().__init__(pars = {})
        self.soil_general_soilline_parameters()
        self.soilgrid_dict()
        self.set_general_location_parameters(**kwargs)


def monthly_amplitude(c):
    d = {}
    d['avgm'] = ((c.iloc[:,0] + c.iloc[:,1])/2).mean()
    return pd.Series(d, index = ['avgm'])

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



