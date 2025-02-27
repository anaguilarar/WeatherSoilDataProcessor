from ._base import DSSATFiles, is_float_regex
from .files_reading import delimitate_header_indices, getting_line_inoutputfile
from ..utils.model_base import BaseOutputData

import os
import numpy as np
import pandas as pd

from datetime import datetime

def update_dssat_data_using_path(path):


    groupclasses = [
        i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))
    ]

    return {
        groupclasses[i]: DSSATOutputData(os.path.join(path, groupclasses[i]))
        for i in range(len(groupclasses))
    }



class DSSATOutput(DSSATFiles):

    def __init__(self, path) -> None:        
        assert path.endswith('.OUT'), 'the file must be and dssat output file format'
        self.lines = self.open_file(path)
    
    def read_output_file_aspddf(self, inittable = '!IDENTIFIERS'):
        #section_id = list(section_indices(self.lines, pattern= inittable))[0]+1
        
        section_id = list(self.get_section_indices(self.lines, pattern= inittable))[0]+1
        section_header_str = self.lines[section_id]
        header_indices = delimitate_header_indices(section_header_str)
        data_rows = []
        header_names = [section_header_str[i:j].strip()
                    for i, j in header_indices]
        for section_data_str in self.lines[(section_id+1):len(self.lines)]:
            data_rows.append(getting_line_inoutputfile(header_names, section_data_str))
        
        self.df = pd.DataFrame(data=data_rows, columns=header_names)
        self.convert_to_dates()
        #convert to numeric
        for colname in self.df.columns:
            if is_float_regex(str(self.df[colname].values[0])):
                self.df[colname] = self.df[colname].astype(float)
        self.df['WUE'] = self.df['HWAH']/self.df['PRCP']
        
        return self.df

    def convert_to_dates(self, format = '%Y%j'):
        
        datenames = [cname for cname in self.df.columns if cname.endswith('DAT')]
        self.df[datenames] = self.df[datenames].map(lambda x: datetime.strptime(x, format) if x != '-99' else np.nan )
        return self.df

   
class DSSATOutputData(BaseOutputData):
    @property
    def extent_files(self):
        return {"climate": ".WTH", "soil": ".SOL", "output": ".OUT"}

    def output_data(self, year: int = None):
        fn_path = self.get_files("output")
        fn_path = list(
            set(
                [
                    i
                    for i in fn_path
                    if os.path.basename(i).lower()[:-4]
                    not in ["error", "evaluate", "warning"]
                ]
            )
        )
        dflist = []
        for fn in fn_path:
            dssatoutput = DSSATOutput(fn)
            df = dssatoutput.read_output_file_aspddf()
            if year:
                df = df.loc[df["PDAT"].dt.year == year]

            if df.shape[0]>0:
                dflist.append(df)
        df = pd.concat(dflist)
        #lat, long = coords_from_soil_file(self.get_files("soil")[0])
        #df["LAT"] = lat
        #df["LONG"] = long
        self.data["output"] = df
        return df

    def weather_data(self, year=None):
        fn_path = self.get_files("climate")
        df = DSSATFiles.extract_table_segment(fn_path[0], "@  DATE")
        df["DATE"] = df["DATE"].map(lambda x: datetime.strptime(x, "%Y%j"))
        if year:
            df = df.loc[df["DATE"].dt.year == year]
        self.data["climate"] = df
        return df

    def soil_data(self):
        fn_path = self.get_files("soil")
        df = DSSATFiles.extract_table_segment(fn_path[0], pattern="@  SLB")
        self.data["soil"] = df
        return df

    