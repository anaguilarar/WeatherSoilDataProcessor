from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Sequence, Union, Any

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
    """
    Organizer to create fertilization schedules for coffee planting cycles.

    Parameters
    ----------
    planting_date : str or Sequence[str]
        Planting date string in '%Y-%m-%d' format or a list of such strings.
    Attributes
    ----------
    planting_dates : List[str]
        List of planting date strings.
    n_cycles : int
        Number of planting cycles.
    _currentcycle : int
        Current cycle index (0-based).
    fert_template : dict
        Temporary template used to accumulate fert events.
    """
    def restart_ferti_template(self) -> None:
        """Reset the internal fertilizer template."""
        self.fert_template = {
            "years": [],
            "dayofyear": [],
            "n_fertilization_per_year": 1,
            "N_amount": [],
        }    

    def __init__(self, planting_date: Union[str, Sequence[str]]):
        """
        Initialize the organizer.

        Parameters
        ----------
        planting_date : str or Sequence[str]
            Planting date (or list of planting dates) in '%Y-%m-%d' format.
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
    
    def add_ferti_event(self, days_after_flowering: float, n_amount: float) -> None:
        """
        Add a fertilization event relative to the planting date of the current cycle.

        Parameters
        ----------
        days_after_flowering : float
            Days after flowering to schedule the event.
        n_amount : float
            Amount of N for the event.
        """
        
        ferti_date = self.pdates[self._currentcycle] + timedelta(days=days_after_flowering)
        year = ferti_date.year
        doy = int(ferti_date.strftime("%Y%j")[4:])
        
        self.fert_template['years'].append(year)
        self.fert_template['dayofyear'].append(doy)
        self.fert_template['N_amount'].append(n_amount)
    
    def create_event_cycle_fert_schedule(
        self, id_cycle: int, days_after_planting: Sequence[float], n_amounts: Sequence[float]
    ) -> Dict[str, Any]:
        """
        Build fertilization events for a single cycle.

        Parameters
        ----------
        id_cycle : int
            Cycle index (0-based).
        days_after_planting : Sequence[float]
            Sequence of days after planting (one per event).
        n_amounts : Sequence[float]
            Sequence of N amounts (matching days_after_planting).

        Returns
        -------
        dict
            Fertilization events dict with keys: 'years', 'dayofyear', 'n_fertilization_per_year', 'N_amount'.
        """
        self._currentcycle =id_cycle
        
        for dap,namount in zip(days_after_planting, n_amounts):
            
            self.add_ferti_event(dap, namount)
        
        ferti_events = self.fert_template
        self.restart_ferti_template()
        return ferti_events
    
    def fertilization_schedule(
        self, application_days: Sequence[float], n_amounts: Sequence[float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a fertilization schedule for all cycles.

        Parameters
        ----------
        application_days : Sequence[float]
            Sequence of application days (relative to planting) for each event.
        n_amounts : Sequence[float]
            Sequence of N amounts corresponding to application_days.

        Returns
        -------
        dict
            Mapping "cycle_treatment_{i}" -> fert events dict.
        """
        ferti_schedule = {}
        for i in range(self.n_cycles):
            ferti_schedule[f'cycle_treatment_{i+1}'] = self.create_event_cycle_fert_schedule(i, application_days, n_amounts)
        
        return ferti_schedule
class CAFManagement():
    """
    Helper to build CAF management templates (fertilization, pruning, thinning).

    Notes
    -----
    Provides convenience factories producing schedules in the project's expected shapes.
    """
    @property
    def _general_template(self):
        return {"prunning_fraction": 0,
            "prunning_years": [3],
            "prunning_day_of_the_year": 55,
            "n_cycles": 1,
            "starting_date": '1991-04-1', 
            "TREE": {
                "species_name": 'sun',
                "tree_density": 0
            }}
    @property
    def _empty_fert(self):
        return {
            "years": [],
            "dayofyear": [],
            "n_fertilization_per_year": 1,
            "N_amount": [],
        }
    
    @property
    def _empty_coffee_prunning(self):
        return {
            "years": [1995],
            "dayofyear": [55],
            "n_prunning_peryear": 1,
            "prun_fraction": 0
        }
        
    @property
    def _empty_tree_prunning(self):
        return {
            "tree_n": 1,
            "years": [1995],
            "dayofyear": [55],
            "n_prunning_peryear": 1,
            "prun_fraction": 0
        }
        
    @property
    def _empty_tree_thinning(self):
        return {
            "tree_n": 1,
            "years": [1995],
            "dayofyear": [55],
            "n_thinning_peryear": 1,
            "thinning_fraction": 0
        }
    
    def restart_config_template(self):
        """Reset the internal configuration template."""
        self.config_template =  {
            **self._general_template,
           "cycle_treatment_1":{
                "planting_date": '1991-04-01',
                "life_cycle_years": 7
                }
        }
    
    def create_treatment_dict(self, **kwargs):
        """
        Create a treatment dictionary for a cycle.

        Parameters
        ----------
        kwargs : dict
            Supported keys: planting_date, life_cycle_years, fertilization, coffee_prunning,
            tree_prunning, tree_thinning.

        Returns
        -------
        dict
            Treatment dictionary.
        """
        treatment_dict = {}
        planting_date = kwargs.get('planting_date')
        treatment_dict.update({'planting_date':planting_date})
        life_cycle_years = kwargs.get('life_cycle_years', 7)
        treatment_dict.update({'life_cycle_years':life_cycle_years})
        fertilization = kwargs.get('fertilization', self._empty_fert)
        treatment_dict.update({'fertilization':fertilization})
        coffee_prunning = kwargs.get('coffee_prunning', self._empty_coffee_prunning)
        treatment_dict.update({'coffee_prunning':coffee_prunning})
        tree_prunning = kwargs.get('coffee_prunning', self._empty_tree_prunning)
        treatment_dict.update({'tree_prunning':tree_prunning})
        tree_thinning = kwargs.get('coffee_prunning', self._empty_tree_thinning)
        treatment_dict.update({'tree_thinning':tree_thinning})
        return treatment_dict

    def create_config_template(self, planting_dates: List[str], starting_date:str = None, coffe_plant_duration:int = 7) -> Dict:
        """
        Create a full configuration template for multiple planting cycles.

        Parameters
        ----------
        planting_dates : sequence of str
            Planting dates in '%Y-%m-%d'.
        starting_date : str, optional
            Override general starting date in '%Y-%m-%d'.
        coffe_plant_duration : int, optional
            Default life cycle years for coffee plants.

        Returns
        -------
        dict
            Combined general configuration and per-cycle treatment dictionaries.
        """
        treatment_dict = {}
        general_config = self._general_template
        if starting_date is not None: general_config['starting_date'] = starting_date
        general_config['coffe_plant_duration'] = coffe_plant_duration
        general_config['n_cycles'] = len(planting_dates)
        
        for i, n_cycle_date in enumerate(planting_dates):
            cycleyear = datetime.strptime(n_cycle_date, '%Y-%m-%d').year
            
            caf_ferti_scheduler = fertiOrganizer(n_cycle_date)
            caf_ferti_scheduler.fertilization_schedule([0], [0])
            ferti = caf_ferti_scheduler.fertilization_schedule([0], [0])['cycle_treatment_1']
            
            dictdata= self.create_treatment_dict(planting_date = n_cycle_date, life_cycle_years = general_config['coffe_plant_duration'], fertilization = ferti)
            for attr in ['coffee_prunning', 'tree_prunning', 'tree_thinning']: dictdata[attr]['years'] = [cycleyear+1]
            treatment_dict['cycle_treatment_{}'.format(i+1)] = dictdata

        return general_config | treatment_dict
        
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
    
    @staticmethod
    def planting_dates_from_aperiod(starting_date, ending_date, n_cycles, coffe_plant_duration:int = 7):
        
        start_date = datetime.strptime(starting_date, '%Y-%m-%d')
        end_years = datetime.strptime(ending_date, '%Y-%m-%d') - relativedelta(years=coffe_plant_duration)

        step_window = int((end_years.year - start_date.year) / n_cycles)
        newyear = start_date.year
        n_cycle_dates = [start_date.strftime('%Y-%m-%d')]
        for _ in range(n_cycles-1):
            nstrdate = datetime.strptime('{}-{}-{}'.format(newyear+ step_window, 
                                                        start_date.month, start_date.day), '%Y-%m-%d')
            newyear = nstrdate.year
            n_cycle_dates.append(nstrdate.strftime('%Y-%m-%d'))
            
        return n_cycle_dates
