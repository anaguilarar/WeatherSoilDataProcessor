from ..utils.model_base import TableDataTransformer

import pandas as pd

# CAFWeather class implementation
class SMWeather(TableDataTransformer):
    @property
    def params_df_names(self):
        return {
            "DATE": "date",
            "year": "year",
            "julian_day": "doy",
            "srad": "srad",
            "tmin": "tmin",
            "tmax": "tmax",
            "vapour_pressure": "vp",
            "wind_speed": "wn",
            "rain": "precipitation",
        }

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        date = pd.to_datetime(df["date"], format="%Y%m%d")
        df["year"] = date.dt.year
        df["doy"] = date.dt.dayofyear

        # Ensure TMAX is not less than TMIN
        if not all(df.tmin <= df.tmax):
            df.loc[(df.tmin > df.tmax),"tmax"] = df.loc[(df.tmin > df.tmax),"tmin"]+1
            
        return df
    
class SMDEM(TableDataTransformer):
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
    

class SMSoil(TableDataTransformer):
    @property
    def params_df_names(self):
        return {
            "DEPTH": "depth",
            "LONG": "x",
            "LAT": "y",
            "SOC": "soc",
            "CSOM0": "som",
            "clay": "clay",
            "sand": "sand"
        }

    @property
    def soilgrid_scaling_factors(self):
        return {
            "clay": 0.1,
            "silt": 0.1,
            "sand": 0.1,
            "nitrogen": 0.01,
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