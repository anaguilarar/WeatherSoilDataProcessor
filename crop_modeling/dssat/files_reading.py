
## taken from https://github.com/AgroClimaticTools/dssat-pylib/blob/main/dssatpylib/util_read_dssat_out.py

from typing import List, Optional, Dict, Union
import pandas as pd
import numpy as np
from datetime import timedelta
from ._base import delimitate_header_indices, DSSATFiles,is_float_regex, section_chunk

import fortranformat as ff


def remove_header_lines(section_list):
    return [section_list[0][0]] + [line[-1] for line in section_list]

def flat_list_of_list(listvalues):
    return [val for l1 in listvalues for val in l1]

def _iterate_until(string, pattern):
    for j,val in enumerate(string):
        if val == pattern: break
    return  j + 1 

def values_type_inffstyle(rowff_style):
    count = 0
    iteration = 0
    seppos = []
    typedata = []
    while count < (len(rowff_style)):
        char = rowff_style[count]
        if is_float_regex(char):
            for z,val in enumerate(rowff_style[count:]):
                    if not is_float_regex(val): break
            nextchars = rowff_style[(z+count):(z+count+1)]
            if nextchars == 'X': count +=2 
            if nextchars == '(': 
                n = int(rowff_style[count:(z+count)])
                j = _iterate_until(rowff_style[count:], ")")
                seppos.append(rowff_style[count:(j+count)])
                if 'A' in rowff_style[count:(j+count)]: typedata.append(n*[str])
                elif 'F' in rowff_style[count:(j+count)]: typedata.append(n*[float])
                elif 'I' in rowff_style[count:(j+count)]: typedata.append(n*[int])
                count +=(j)
        else:
            if char == ',':
                count+=1
                continue
            j = _iterate_until(rowff_style[count:], ",")
            seppos.append(rowff_style[count:(j+count)])
            if 'I' in rowff_style[count:(j+count)]:
                typedata.append([int])
            elif 'A' in rowff_style[count:(j+count)]:
                typedata.append([str])
            elif 'F' in rowff_style[count:(j+count)]:
                typedata.append([float])
            count +=j
        iteration +=1
        assert iteration<100, 'Something wrong with the style {}'.format(seppos)
    typedata = flat_list_of_list(typedata)
    return seppos, typedata


def join_row_using_header_indices(section_header_str, section_line_str, row_to_replace):
    
    header_indices = delimitate_header_indices(section_header_str)

    stline =[None]*len(header_indices) 
    for z, (i, j) in enumerate(header_indices):
        stline[z] = ' '*len(section_header_str)

        if z == 0:
            stline[z] = section_line_str[:(j-i)]
        elif section_header_str[i:j].count(' ')>0:
            posini = [pos for pos, char in enumerate(section_header_str[i:j]) if char == ' ']
            stline[z] = ' '* (posini[0]+1) + row_to_replace[z][:(j-i)] + ' '* abs((len(row_to_replace[z])+1)-(j-i))  
        else:
            stline[z] = row_to_replace[z][:(j-i)] 

    return ''.join(stline)+'\n'

def getting_line_inoutputfile(header, line):
    """
    this only works assuming that the variables are not space separated, and the only one is the initial planting
    """
    
    dataline = [i for i in line.split(' ') if i != '']
    newline = [' ']
    count = 0
    while len(newline) < (len(header)) and count < len(dataline):
        if dataline[count].startswith('Initial'):
            newline.append((dataline[count] + ' ' + dataline[count + 1]).strip())
            count = count + 2
            continue
        elif dataline[count].startswith('Planting'):
            newline.append((dataline[count] + ' '+ dataline[count + 1] + ' ' + dataline[count + 2]).strip())
            count = count + 3
            continue
        else:
            newline.append(dataline[count].strip())
            count +=1
    
    return newline

def df_to_dssat_string_format(df, header_style, row_style):
    
    header_line = ff.FortranRecordWriter(header_style).write(
                    tuple(df.columns.values))+'\n'
    _, data_row_type = values_type_inffstyle(row_style)
    rowline = []
    for j in range(df.shape[0]):
        subsetr = []
        assert len(tuple(df.values[j])) == len(data_row_type)
        for i, val in enumerate(tuple(df.values[j])):
            
            subsetr.append(data_row_type[i](val))
        rowline.append(ff.FortranRecordWriter(row_style).write(subsetr)+'\n')
    
    return [header_line]+ rowline

class DSSATFileModifier(DSSATFiles):
    """
    A class to modify sections of DSSAT input files with new information.
    """

    def __init__(self, path: str = None, section_names: Dict[str, str] = None):
        """
        Initialize the DSSAT file modifier.

        Parameters
        ----------
        path : str
            Path to the DSSAT input file.
        section_names : Dict[str, str]
            Dictionary mapping section names to their patterns.
        """
        self.file_path = path
        self.section_names = section_names
        if self.file_path:
            self.lines = self.open_file(self.file_path)

    def _modify_lines(self, new_df: List[pd.DataFrame], section_name:str, sub_section_pattern:str):
        section_lines = section_chunk(self.lines, section_name)
        section_indxs = list(self.get_section_indices(section_lines, pattern=sub_section_pattern))
        newc =section_lines.copy()
        for i in range(len(section_indxs)):
            
            if i != len(section_indxs)-1:
                for j in range(len(section_lines[section_indxs[i]+1:section_indxs[i+1]])):
                    newc[j+(i)+1] = join_row_using_header_indices(section_lines[section_indxs[i]], section_lines[section_indxs[i]+1], new_df[i].values[0])
            else:
                for j in range(len(section_lines[section_indxs[i]+1:])):
                    
                    newc[j+(section_indxs[i]+1)] = join_row_using_header_indices(section_lines[section_indxs[i]], section_lines[section_indxs[i]+1], new_df[i].values[j])
                    
        return newc

    def _write_file(self, lines: List[str]) -> None:
        """Write modified lines back to the DSSAT file."""
        with open(self.file_path, 'w') as file:
            file.writelines(lines)

    def _extract_section(self, section_name: str) -> List[str]:
        """
        Extract lines of a specific section based on its name.

        Parameters
        ----------
        section_name : str
            Name of the section to extract.

        Returns
        -------
        List[str]
            Lines of the section.
        """
        section_pattern = self.section_names[section_name]
        section_start = next(i for i, line in enumerate(self.lines) if section_pattern in line) + 1
        section_end = next((i for i, line in enumerate(self.lines[section_start:]) if line == '\n'), len(self.lines)) + section_start
        return self.lines[section_start:section_end]

    def write_df_asff(
        self,
        df_list: str,
        header_ffstyle:str, row_ffstyle:str
    ) -> None:
        """
        Modify a section of the DSSAT file using a list of DataFrame changes.

        Parameters
        ----------
        section_name : str
            Name of the section to modify.

        """
        return(df_to_dssat_string_format(df_list,header_ffstyle,row_ffstyle))

    def _lines_to_df(self, lines: List[str], header_idx: int) -> pd.DataFrame:
        """
        Convert section lines into a DataFrame.

        Parameters
        ----------
        lines : List[str]
            Section lines.
        header_idx : int
            Index of the header line.

        Returns
        -------
        pd.DataFrame
            DataFrame representation of the section.
        """
        header = lines[header_idx].split()
        data = [line.split() for line in lines[header_idx + 1:] if line.strip()]
        return pd.DataFrame(data, columns=header)

    def _df_to_lines(self, df: pd.DataFrame, header: str) -> List[str]:
        """
        Convert a DataFrame back into section lines.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to convert.
        header : str
            Original header line of the section.

        Returns
        -------
        List[str]
            Converted lines.
        """
        lines = [header]
        for _, row in df.iterrows():
            line = ' '.join(str(value).ljust(len(col)) for col, value in zip(df.columns, row.values))
            lines.append(line)
        return lines

