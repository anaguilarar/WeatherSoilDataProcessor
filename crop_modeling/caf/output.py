import glob
import os
import pandas as pd
from datetime import datetime
from ..utils.model_base import BaseOutputData

class CAFOutputData(BaseOutputData):
    
    @property
    def extent_files(self):
        return {"climate": "cafweather", "soil": "cafsoil", "output": "output_"}

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
            year = df.year.values
            doy = df.doy.values
            df["HDAT"] = [datetime.strptime('{:0.0f}-{:0.0f}'.format(y,d), '%Y-%j')
            for y,d in zip(year, doy)]
            df["n_cycle"] = n_run+1
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
        doy = df.doy.values
        df["DATE"] = [datetime.strptime('{:0.0f}-{:0.0f}'.format(y,d), '%Y-%j')
            for y,d in zip(year, doy)]

        
        self.data["climate"] = df
        return df

    def soil_data(self) -> pd.DataFrame:
        fn_path = self.get_files("soil")
        df = pd.read_csv(fn_path[0])
        self.data["soil"] = df
        return df
        