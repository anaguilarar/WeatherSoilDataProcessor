import os

from abc import ABC, abstractmethod
from typing import List
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

        #list_files = glob.glob(self._tmp_path+'/*.{}*'.format('SOL'))
        self._process_paths = [os.path.dirname(fn) for fn in list_files]
        return self._process_paths
    
    
