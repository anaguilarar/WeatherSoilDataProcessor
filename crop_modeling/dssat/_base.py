import abc 
from typing import List
import os
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


def delimitate_header_indices(section_header_str):
    start_indices = [0]+[i for i, character in enumerate(section_header_str)
                            if character == ' ' and section_header_str[i+1] != ' ']
    end_indices = start_indices[1:] + [len(section_header_str)+20]
    return list(zip(start_indices, end_indices))

def section_indices(lines, pattern = '@'):
    #with open(path, 'r', encoding="utf-8") as fn:
    for i, line in enumerate(lines):
        stiped_line = line.strip()
        if stiped_line.startswith(pattern):
            yield i
            
def section_to_df(lines, header_idx = 0):
    
    data_rows = []
    for i, line in enumerate(lines[header_idx+1:]):
        stiped_line = line.strip()
        data_rows.append([strid for strid in stiped_line.split(' ') if strid != ''])
        
    header_indices = delimitate_header_indices(lines[header_idx])
    
    header_names = [lines[header_idx][i:j].strip()
                for i, j in header_indices]
    df = pd.DataFrame(data=data_rows, columns=header_names
                        ) if len(data_rows[0]) == len(header_names) else pd.DataFrame(
                            data=data_rows, columns=header_names[1:])
    return df


def list_subsection_as_df(section_lines, header_indices):
    dfsection = []
    for i in range(len(header_indices)):
        if i == len(header_indices)-1:
            dfsection.append(section_to_df(section_lines[header_indices[i]:]))
        else:
            dfsection.append(section_to_df(section_lines[header_indices[i]:header_indices[i+1]]))
            
    return dfsection

def section_chunk(lines, section_pattern):
    section_pos = list(section_indices(lines, pattern=section_pattern))
    section_init = section_pos[0]+1
    for i, line in enumerate(lines[section_init:]):
        if line =='\n':
            section_end = i+ section_init
            break
    
    return lines[section_init:section_end]

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
    
    @staticmethod
    def extract_section_asdf(path, section_name, sub_section_pattern):
        
        lines = DSSATFiles.open_file(path)
        
        section_lines = section_chunk(lines, section_name)
        header_inds = list(section_indices(section_lines, pattern=sub_section_pattern))
        dflist = list_subsection_as_df(section_lines, header_indices =header_inds)
        return dflist