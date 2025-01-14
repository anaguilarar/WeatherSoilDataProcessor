import numpy as np
from typing import Dict

def repeate_mpractice(ntimes, years, dayofyear, practice_value):
    array_c = np.zeros((ntimes*len(years),3), dtype=float)
    count = 0
    for y in years: 
        array_c[count:count+ntimes,0] = [y] * ntimes
        if isinstance(dayofyear, list):
            if ntimes == len(dayofyear):
                array_c[count:count+ntimes,1] = dayofyear[:ntimes]
            else:
                array_c[count:count+ntimes,1] = dayofyear + [0]*(ntimes - len(dayofyear))
        else:
            array_c[count:count+ntimes,1] = dayofyear
        array_c[count:count+ntimes,-1] = practice_value
        count +=ntimes
    
    return array_c

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
