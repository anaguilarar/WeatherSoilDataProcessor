from .files_reading import delimitate_header_indices, getting_line_inoutputfile
import pandas as pd
from ._base import DSSATFiles, is_float_regex
from datetime import datetime

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
        self.df[datenames] = self.df[datenames].map(lambda x: datetime.strptime(x, format))
        return self.df
        