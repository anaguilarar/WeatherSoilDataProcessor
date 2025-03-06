

import os
import json
import pandas as pd

from datetime import datetime

from ..utils.model_base import BaseOutputData, ReporterBase


class SimpleModelOutputData(BaseOutputData):
    
    @property
    def extent_files(self):
        return {"climate": "smweather", "soil": "smsoil", "output": "output_"}

    def output_data(self, year: int = None) -> pd.DataFrame:
        fn_path = self.get_files("output")
        fn_path = list(
            set(
                [
                    i
                    for i in fn_path
                ]
            )
        )
        dflist = []
        fn_path.sort()

        for n_run, fn in enumerate(fn_path):
            df = pd.read_csv(fn)
            dflist.append(df)
        df = pd.concat(dflist)
        self.data["output"] = df
        return df

    def weather_data(self, year: int = None) -> pd.DataFrame:

        fn_path = self.get_files("climate")
        
        df = pd.read_csv(fn_path[0])
        if year:
            df = df.loc[df.year == year]
            
        year = df.year.values
        
        df["DATE"] = df["DATE"].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

        
        self.data["climate"] = df
        return df

    def soil_data(self) -> pd.DataFrame:
        fn_path = self.get_files("soil")
        df = pd.read_csv(fn_path[0])
        self.data["soil"] = df
        return df
        


class SimpleModelReporter(ReporterBase):
    
    def __init__(self):
        self._tmp_reporter_keys = ['crop', 'TRNO', 'soil_texture', 'longitude', 'latitude', 'altitude','co2','sowing_date', 'harvesting_date', 'cum_temp', 'madurity_day', 'biomass', 'crop_yield']
        super().__init__()
        self.set_reporter(self._tmp_reporter_keys)
        
    def clear_report(self):
        self.set_reporter(self._tmp_reporter_keys)
        