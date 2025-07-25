import os
import glob
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import xarray

from .process import summarize_datacube_as_df


def loadjson(fn):
    """
    Load JSON data from a file.

    Parameters
    ----------
    fn : str
        Filename of the JSON file to load.

    Returns
    -------
    dict or None
        Dictionary containing the loaded JSON data.
        Returns None if the file does not exist.
    """
    
    if os.path.exists(fn):
        with open(fn, "rb") as fn:
            reporter = json.load(fn)
    else:
        reporter = None
    return reporter


class ModelBase:
    """
    Base class to handle model initialization for DSSAT and CAF models.
    """
    def __init__(self, working_path: str):
        self.path = working_path
        
        self._process_paths: List[str] = []
        
        if not os.path.exists(working_path):
            os.makedirs(working_path)
    
    @abstractmethod
    def run(self):
        """Placeholder for model-specific initialization."""
        raise NotImplementedError("Subclasses must implement the 'run' method.")
    
    @abstractmethod
    def from_datacube_to_files(self):
        """Placeholder for model-specific initialization."""
        raise NotImplementedError("Subclasses must implement the 'from_datacube_to_files' method.")
    
    
    @abstractmethod
    def set_up_management(self):
        """Placeholder for model-specific initialization."""
        raise NotImplementedError("Subclasses must implement the 'set_up_management' method.")
    
    def find_envworking_paths(self, path = None, file_ext = 'WTH'):
        path = self.path if path is None else path
        folders = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path,i))]
        list_files = []
        for folder in folders: 
            pathsin = glob.glob(os.path.join(path, folder)+'/*.{}*'.format(file_ext)) 
            if pathsin: list_files.append(pathsin[0])

        self._process_paths = [os.path.dirname(fn) for fn in list_files]
        return self._process_paths
    
class BaseOutputData(ABC):
    """
    Abstract base class for handling the models outputs
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the object with a path.

        Parameters
        ----------
        path : str
            Path to the directory containing model files.
        """
        self.data = {}
        self.path = path


    @property
    @abstractmethod
    def extent_files(self) -> dict:
        """
        Define the file extensions or patterns for different data types.

        Returns
        -------
        dict
            Mapping of data type to file extensions or patterns.
        """
        pass

    def get_files(self, data_type: str):
        """
        Retrieve files of the specified type.

        Parameters
        ----------
        data_type : str
            The type of data to fetch files for ("climate", "soil", "output").

        Returns
        -------
        list
            List of file paths matching the specified type.
        """

        fns = glob.glob(self.path + f"/*{self.extent_files[data_type]}*")
        assert len(fns) > 0, f"No files were found for {data_type}"
        return fns
    
    @abstractmethod
    def output_data(self, year: int = None) -> pd.DataFrame:
        """
        Extract and process output data.

        Parameters
        ----------
        year : int, optional
            Filter data for a specific year.

        Returns
        -------
        pd.DataFrame
            Processed output data.
        """
        pass

    @abstractmethod
    def weather_data(self, year: int = None) -> pd.DataFrame:
        """
        Extract and process weather data.

        Parameters
        ----------
        year : int, optional
            Filter data for a specific year.

        Returns
        -------
        pd.DataFrame
            Processed weather data.
        """
        pass

    @abstractmethod
    def soil_data(self) -> pd.DataFrame:
        """
        Extract and process soil data.

        Returns
        -------
        pd.DataFrame
            Processed soil data.
        """
        pass

class ReporterBase():
    def __init__(self):
        self._report = None
        self._report_keys = []
        
    
    @property
    def report(self) -> dict:
        return self._report
        
    def set_reporter(self, checkpoint_keys):
        
        reporter = {}
        for keyname in checkpoint_keys:
            reporter.update({keyname: []})
        self._report = reporter
        self._update_keys(reporter)
        return reporter
    

    
    def _update_keys(self, reporter_dict):
        self._report_keys = [keyarg for keyarg in reporter_dict.keys()]
    
    def update_report(self, new_entry):    
        
        """
        Update the reporter with a new entry.

        Parameters
        ----------
        new_entry : dict
            A dictionary containing the new entry to add. Keys in this dictionary should match 
            those in the _reporter_keys attribute.

        Raises
        ------
        ValueError
            If the keys in the new_entry do not match the _reporter_keys.

        Returns
        -------
        None
        """
        
        if not all(key in self._report_keys for key in new_entry):
            raise ValueError("Keys in the new entry do not match the reporter keys.")
        
        for k in list(self._report_keys):
            self._report[k].append(new_entry[k])   
    
    
    def save_reporter(self, path: str, fn:str, suffix = '.json'):
        if fn.endswith('.json'):
            json_object = json.dumps(self.report, indent=4)
            with open(os.path.join(path, fn), "w") as outfile:
                outfile.write(json_object)
        elif fn.endswith('.csv'):
            df = pd.DataFrame(self.report)
            df.to_csv(os.path.join(path, fn), index = False)
            
    def load_reporter(self, path, verbose = True):
        reporter = loadjson(path)
        if reporter is None:
            self.set_reporter([''])
            if verbose: print('No data was found')
        else:
            if verbose: print('load')
        self._update_keys(reporter)
        self._report = reporter

# Define the abstract base class
class TableDataTransformer(ABC):
    """
    Abstract base class for transforming xarray datasets into table data format.

    Subclasses must implement the specific functionality required for their data type.
    """

    def __init__(self, xrdata: xarray.Dataset = None, xrdata_path:str = None):
        """
        Initialize the CAFSP_transformer instance.

        Parameters
        ----------
        xrdata : xarray.Dataset, optional
                Input xarray Dataset to be processed. Required if `xrdata_path` is not provided.
            xrdata_path : str, optional
                Path to a NetCDF file containing the dataset. Used if `xrdata` is not passed.

        """
        self.xrdata = xrdata
        self.xrdata_path = xrdata_path

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
        group_by_layer: Optional[np.ndarray] = None,
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
        group_by_layer : np.ndarray, optional
            a numpy array indicating the group category for each spatial location
        
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
            xrdata = self.xrdata, xrdata_path= self.xrdata_path, dimension_name=depth_var_name, group_by=group_by,group_by_layer = group_by_layer,  project_to=target_crs
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
            
            