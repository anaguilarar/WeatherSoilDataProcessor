from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

import numpy as np

def repeate_mpractice(ntimes, years, dayofyear, practice_value):
    array_c = np.zeros((ntimes*len(years),3), dtype=float)
    count = 0
    if ntimes == 1:
        array_c[:,0] = years
        array_c[:,1] = dayofyear
        array_c[:,2] = practice_value
        return array_c
        
    for y in years: 
        array_c[count:count+ntimes,0] = [y] * ntimes
        if isinstance(dayofyear, list) and ntimes>1:
            if ntimes == len(dayofyear):
                array_c[count:count+ntimes,1] = dayofyear[:ntimes]
            else:
                array_c[count:count+ntimes,1] = dayofyear + [0]*(ntimes - len(dayofyear))
        else:
            array_c[count:count+ntimes,1] = dayofyear
        array_c[count:count+ntimes,-1] = practice_value
        count +=ntimes
    return array_c

class prunningOrganizer():
    
    def restar_prunning_template(self):
        
        self.prunning_template = {
            'years': [],
            'dayofyear': [],
            'n_prunning_peryear': 1,
            'prun_fraction': 0
        }
    
    def __init__(self, planting_date):
        
        self.planting_dates = planting_date
        if isinstance(planting_date, list):
            self.n_cycles = len(planting_date)
        else:
            self.planting_dates = [planting_date]
            self.n_cycles = 1
        
        self._currentcycle = 1
        self.restar_prunning_template()
    
    @property
    def pdates(self):
        return [datetime.strptime(planting_date, '%Y-%m-%d') for planting_date in self.planting_dates]
    
    def add_prunning_event(self, year:int, days_of_year:int, prunning_fraction:float):
        
        
        pdate = self.pdates[self._currentcycle] 
        pyear = pdate.replace(year = pdate.year+year).year
        
        self.prunning_template['years'].append(pyear)
        self.prunning_template['dayofyear'].append(days_of_year)
        self.prunning_template['prun_fraction'] = prunning_fraction
    

    def create_event_cycle_prunning_schedule(self, id_cycle, years: List[int], days_ofthe_year: List[int], prunning_fraction: float) -> dict:
        self._currentcycle =id_cycle
        
        for year,doy in zip(years, days_ofthe_year):
            
            self.add_prunning_event(year, doy, prunning_fraction)

        prunning_events = self.prunning_template
        self.restar_prunning_template()
        return prunning_events
    
    def prunning_schedule(self, years: List[int], days_ofthe_year: List[int], prunning_fraction: float) -> dict:
        prunning_schedule = {}
        for i in range(self.n_cycles):
            prunning_schedule[f'cycle_treatment_{i+1}'] = self.create_event_cycle_prunning_schedule(i, years, days_ofthe_year, prunning_fraction)
        
        return prunning_schedule
    
class fertiOrganizer():
    
    def restart_ferti_template(self):
        self.fert_template ={
            'years': [],
            'dayofyear': [],
            'n_fertilization_per_year': 1,
            'N_amount': []
            }    

    def __init__(self, planting_date:List[str]):
        """
        

        Args:
            planting_date (str): %Y-%m-%d'
            n_cycles (_type_): _description_
        """
        
        self.planting_dates = planting_date
        if isinstance(planting_date, list):
            self.n_cycles = len(planting_date)
        else:
            self.planting_dates = [planting_date]
            self.n_cycles = 1
        
        self._currentcycle = 1
        self.restart_ferti_template()
        
    @property
    def pdates(self):
        return [datetime.strptime(planting_date, '%Y-%m-%d') for planting_date in self.planting_dates]
    
    def add_ferti_event(self, days_after_flowering:float, n_amount:float):
        
        ferti_date = self.pdates[self._currentcycle] + timedelta(days=days_after_flowering)
        year = ferti_date.year
        doy = int(ferti_date.strftime("%Y%j")[4:])
        
        self.fert_template['years'].append(year)
        self.fert_template['dayofyear'].append(doy)
        self.fert_template['N_amount'].append(n_amount)
    
    def create_event_cycle_fert_schedule(self, id_cycle, days_after_planting: List[float], n_ammounts: List[float]) -> dict:
        self._currentcycle =id_cycle
        
        for dap,namount in zip(days_after_planting, n_ammounts):
            
            self.add_ferti_event(dap, namount)
        
        ferti_events = self.fert_template
        self.restart_ferti_template()
        return ferti_events
    
    def fertilization_schedule(self, application_days, n_ammounts):
        ferti_schedule = {}
        for i in range(self.n_cycles):
            ferti_schedule[f'cycle_treatment_{i+1}'] = self.create_event_cycle_fert_schedule(i, application_days, n_ammounts)
        
        return ferti_schedule

        return flowering_dates, baseline_yields
class CAFManagement():
    
    def fertilization_schedule(self, n_fertilization_per_year = None, years = None, dayofyear = None, N_amount = None, schedule: Dict = None):
        fert_calendar = np.zeros((100,3), dtype=float)
        fert_calendar[:] = -1
        if schedule is None:
            sched = repeate_mpractice(ntimes=n_fertilization_per_year, years=years, dayofyear=dayofyear, practice_value = N_amount)
            fert_calendar[:sched.shape[0]] = sched
        else:
            pass # TODO 
        return fert_calendar
        
    def coffe_prunning_schedule(self, n_prunning_peryear = None, years = None, dayofyear = None, prun_fraction = 0.25, schedule: Dict = None):
        cprun_calendar = np.zeros((100,3), dtype=float)
        cprun_calendar[:] = -1
        if schedule is None:
            sched = repeate_mpractice(ntimes=n_prunning_peryear, years=years, dayofyear=dayofyear, practice_value = prun_fraction)
            cprun_calendar[:sched.shape[0]] = sched
            
        return cprun_calendar
    
    def tree_prunning_schedule(self, n_c = 3, n_prunning_peryear = None, years = None, dayofyear = None, prun_fraction = 0.25, tree_n = 1, schedule: Dict = None):
        tprun_calendar = np.zeros((n_c,100,3), dtype=float)
        tprun_calendar[:] = -1
        if schedule is None:
            sched = repeate_mpractice(ntimes=n_prunning_peryear, years=years, dayofyear=dayofyear, practice_value = prun_fraction)
            tprun_calendar[tree_n-1,:sched.shape[0]] = sched
        
        return tprun_calendar
    
    def tree_thinning_schedule(self, n_c = 3, n_thinning_peryear = None, years = None, dayofyear = None, thinning_fraction = 0.1, tree_n = 1, schedule: Dict = None):
        tthinning_calendar = np.zeros((n_c,100,3), dtype=float)
        tthinning_calendar[:] = -1
        if schedule is None:
            sched = repeate_mpractice(ntimes=n_thinning_peryear, years=years, dayofyear=dayofyear, practice_value = thinning_fraction)
            tthinning_calendar[tree_n-1,:sched.shape[0]] = sched
        
        return tthinning_calendar
