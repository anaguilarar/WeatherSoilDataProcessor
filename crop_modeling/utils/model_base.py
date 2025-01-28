import os

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import glob

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
