import abc 
from typing import List
import os
from .files_reading import delimitate_header_indices, section_indices
import pandas as pd

import os
import re

def is_float_regex(value):
    return bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+$', value))

def coords_from_soil_file(soil_path):
    DSSATFiles.open_file(soil_path)

    lines = DSSATFiles.open_file(soil_path)
    for i in DSSATFiles.get_section_indices(lines, '@SITE'):
        lstripped = lines[i+1].strip()
        datainline = [strid for strid in lstripped.split(' ') if strid != '']
        if is_float_regex(datainline[2]) and is_float_regex(datainline[3]):
            lat, long = float(datainline[2]), float(datainline[3])
        else: 
            lat, long = None, None
            
    return lat, long


class DSSATFiles(): 
    @staticmethod
    def open_file(path: str) -> List[str]:
        """
        Reads a text file and returns its lines as a list.

        Parameters
        ----------
        path : str
            Path to the DSSAT soil file.

        Returns
        -------
        List[str]
            A list of lines in the file.

        Raises
        ------
        AssertionError
            If the file does not exist at the specified path.
        """
        
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf-8") as fn:
            lines = fn.readlines()
            
        return lines
    @staticmethod
    def get_section_indices(line, pattern):
        return section_indices(line, pattern)
    
    @staticmethod
    def extract_table_segment(path, pattern):
        
        lines = DSSATFiles.open_file(path)
        section_id = list(DSSATFiles.get_section_indices(lines, pattern= pattern))[0]
        section_header_str = lines[section_id]
        data_rows = []
        for i, line in enumerate(lines[section_id+1:]):
            stiped_line = line.strip()
            data_rows.append([strid for strid in stiped_line.split(' ') if strid != ''])
        header_indices = delimitate_header_indices(section_header_str)
        
        header_names = [section_header_str[i:j].strip()
                    for i, j in header_indices]

        return pd.DataFrame(data=data_rows, columns=header_names[1:])
        