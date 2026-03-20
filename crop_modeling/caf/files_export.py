import pandas as pd
from ..utils.model_base import TableDataTransformer

# CAFWeather class implementation
class CAFWeather(TableDataTransformer):
    @property
    def params_df_names(self):
        return {
            "year": "year",
            "doy": "doy",
            "GR": "srad",
            "TMIN": "tmin",
            "TMAX": "tmax",
            "VP": "vp",
            "WN": "wn",
            "RAIN": "precipitation",
        }

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        date = pd.to_datetime(df["date"], format="%Y%m%d")
        df["year"] = date.dt.year
        df["doy"] = date.dt.dayofyear

        # Ensure TMAX is not less than TMIN
        if not all(df.tmin <= df.tmax):
            df.loc[(df.tmin > df.tmax),"tmax"] = df.loc[(df.tmin > df.tmax),"tmin"]+1
            
        return df
    
class CAFDEM(TableDataTransformer):
    @property
    def params_df_names(self):
        return {
            "LONG": "x",
            "LAT": "y",
            "ELEV": "dem",
            "SLOPE": "slope",
        }

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    

class CAFSoil(TableDataTransformer):
    @property
    def params_df_names(self):
        return {
            "DEPTH": "depth",
            "LONG": "x",
            "LAT": "y",
            "SOC": "soc",
            "CSOM0": "som",
            "clay": "clay",
            "silt": "silt"
        }

    @property
    def soilgrid_scaling_factors(self):
        return {
            "clay": 0.1, # to convert to percentage
            "silt": 0.1, # to convert to percentage
            "nitrogen": 0.01, # cg/kg to g/kg
            "bdod": 0.01,
            "cfvo": 0.01,
            "phh2o": 0.1,
            "soc": 0.01,
            "cec": 0.1,
            "wv1500": 0.001,
            "wv0033": 0.001,
            "wv0010": 0.001,
        }

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, factor in self.soilgrid_scaling_factors.items():
            if col in df.columns:
                df[col] *= factor
        df["som"] = df["soc"].values * (100 / 55)  # Convert SOC to SOM ## https://www.nature.com/articles/s41598-022-05476-5 som is 55 soc
        
        return df
    