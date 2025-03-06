import math
import pandas as pd
import os

from .weather import Station, Weather
from . import TEXTURE_CLASSES, RUNOFF_CURVES
import numpy as np

class ARID:
    """
    Agricultural Reference Index for drought 
    Woli et al, 2012 https://doi.org/10.2134/agronj2011.0286
    
    """
    
    def __init__(self,  weather: Weather, eto_method = 'PT', **kwargs):
        
        self.eto_method = eto_method
        self.weather = weather
        self._initialize()        
    
    def _initialize(self):
        self._daily_wat = [] # availeble water in the root zone after transpiration
        self._daily_arid = [] # reference index for drought 
        
        self.Tpi:float = None # potential transpiration day i
        self.Et0:float = None # peotential evapotranspiration of the reference grass

    
    def soil_params(self, texture):
        self.awc = 0.13 ## Similarly, ARID uses 0.13 mm mm–1 for θm because the water holding capacities of many soil types, except the ones having >65% sand, is about 0.13 mm mm Wolin et al
        self.ddc = 0.55 # For simplicity, ARID assumes β (deep drainage) to be 0.55,
        self.rcn = RUNOFF_CURVES[TEXTURE_CLASSES.index(texture)]
        self.rzd = 1500 # root (mm) parameter for banana crop 
        self.wuc = 0.096
        self.texture = texture

    
    def soil_water_balance():
        
        pass
    

    def index_value(self, day):
        # Wi = Wi-1 + Pi + Ii - DRi + Ri
        
        etoi = self.weather.get_day_eto(self.eto_method)
        
        ## runoff woli et al Ri = (Pi - Ia)^2 / (Pi - Ia +S)
        if self.weather.rain>(0.2 * (25400/self.rcn - 254)):
            ro = (self.weather.rain -(0.2*(25400/self.rcn-254)))**2/(self.weather.rain+0.8*(25400/self.rcn-254))
        else: ro = 0
        
        cwbd = self.weather.rain - ro
        
        if len(self._daily_wat) == 0:
            w_pre = self.rzd * self.awc
            
        else:
            w_pre = self._daily_wat[-1]
        
        
        awrz = w_pre + cwbd
        
        # deep drainage
        if(awrz /self.rzd > self.awc):
            dri = self.rzd*self.ddc*(awrz/self.rzd - self.awc)
        else: dri = 0
        
        wad = awrz - dri
        
        tr  = min(self.wuc*self.rzd*wad/ self.rzd, etoi)
        
        self._daily_wat.append(wad - tr)
        self._daily_arid.append(1-tr/etoi)
    
    def calculate_range(self, start, end, **kwargs):
        data_range  = np.zeros(end-start)
        self._initialize()
        
        for day in range(start, end):

            self.weather.get_day_weather(day, **kwargs)
            self.index_value(day)
            data_range[day-start] = self._daily_arid[-1]
        return data_range
            
    def __call__(self, day, **kwargs):
        
        self.weather.get_day_weather(day, **kwargs)
        self.index_value(day)
        return self._daily_arid[-1]
        
        