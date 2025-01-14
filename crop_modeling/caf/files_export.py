import pandas as pd
import numpy as np
import os
import xarray

from abc import ABC, abstractmethod
from typing import Optional, Dict

from ..utils.process import summarize_datacube_as_df

# Define the abstract base class
class CAFSP_transformer(ABC):
    """
    Abstract base class for transforming xarray datasets into CAF data format.

    Subclasses must implement the specific functionality required for their data type.
    """

    def __init__(self, xrdata: xarray.Dataset):
        """
        Initialize the CAFSP_transformer instance.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The input dataset to be processed.
        """
        self.xrdata = xrdata

    @property
    @abstractmethod
    def params_df_names(self) -> Dict[str, str]:
        """Dictionary mapping target column names to their input names."""
        pass

    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and transform the data specific to the subclass."""
        pass

    def __call__(
        self,
        group_by: Optional[str] = None,
        depth_var_name: Optional[str] = None,
        codes: Optional[Dict[int, str]] = None,
        target_crs: str = "EPSG:4326",
        outputpath: str = None,
    ):
        """
        Process and export data as CSV files grouped by the specified attribute.

        Parameters
        ----------
        group_by : str, optional
            Column to group the data by. If None, data will be grouped into a single group.
        depth_var_name : str, optional
            Name of the depth dimension, by default None.
        codes : dict, optional
            Mapping of group indices to names for output directories, by default None.
        target_crs : str, optional
            Target coordinate reference system, by default "EPSG:4326".
        outputpath : str
            Path to save the output CSV files.
        """
        dfdata = summarize_datacube_as_df(
            self.xrdata, dimension_name=depth_var_name, group_by=group_by, project_to=target_crs
        )
        
        if not group_by:
            dfdata["group"] = "0"
            group_by = "group"

        # Process data specific to the subclass
        dfdata = self.process_data(dfdata)

        # Rename columns based on `params_df_names`
        dfdata = dfdata.rename(columns={v: k for k, v in self.params_df_names.items()})

        # Export grouped data
        unique_groups = np.unique(dfdata[group_by].values)
        for group_id in unique_groups:
            subset = dfdata[dfdata[group_by] == group_id]

            group_dir = (
                os.path.join(outputpath, codes[group_id].replace(" ", ""))
                if codes is not None
                else os.path.join(outputpath, f"_{group_id}")
            )
            if not os.path.exists(group_dir): os.mkdir(group_dir)
            fn = os.path.join(group_dir, f"{self.__class__.__name__.lower()}.csv")
            subset[list(self.params_df_names.keys())].to_csv(
                fn, index=False
            )
            
# CAFWeather class implementation
class CAFWeather(CAFSP_transformer):
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
    
class CAFDEM(CAFSP_transformer):
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
    

class CAFSoil(CAFSP_transformer):
    @property
    def params_df_names(self):
        return {
            "DEPTH": "depth",
            "LONG": "x",
            "LAT": "y",
            "SOC": "soc",
            "CSOM0": "som",
        }

    @property
    def soilgrid_scaling_factors(self):
        return {
            "clay": 0.1,
            "silt": 0.1,
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
    