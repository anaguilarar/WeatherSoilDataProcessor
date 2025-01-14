import glob
import os
import pandas as pd
from datetime import datetime

class CAFOutputData:
    @property
    def extent_files(self):
        return {"climate": "cafweather", "soil": "cafsoil", "output": "output.csv"}

    def get_files(self, type):
        fns = glob.glob(self.path + f"/{self.extent_files[type]}*")
        assert len(fns) > 0, f"No files were found {type}"
        return fns

    def output_data(self, year: int = None):
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
        for fn in fn_path:
            df = pd.read_csv(fn)
            if year:
                df = df.loc[df.year == year]
            year = df.year.values
            doy = df.doy.values
            df["HDAT"] = [datetime.strptime('{:0.0f}-{:0.0f}'.format(y,d), '%Y-%j')
            for y,d in zip(year, doy)]
            
            dflist.append(df)
        df = pd.concat(dflist)
        self.data["output"] = df
        return df

    def weather_data(self, year=None):

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

    def soil_data(self):
        fn_path = self.get_files("soil")
        df = pd.read_csv(fn_path[0])
        self.data["soil"] = df
        return df

    def __init__(self, path) -> None:
        self.data = {}
        self.path = path
        print(path)
        