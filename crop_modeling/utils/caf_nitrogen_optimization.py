"""Bayesian optimisation of the nitrogen fertilisation of coffee with the CAF2021 model.

Coffee counterpart of ``ferti_optimization.FertilizerBayesian`` (DSSAT crops). The fertilisation plan
that is optimised has one application for the establishment years and one or more applications tied to
flowering:

    month0, n0   N applied once a year during the first ``ESTABLISHMENT_YEARS`` years of every
                 plantation, on day of the year ``month0 * 30``
    monthK, nK   (K >= 1) N applied ``monthK`` months after every flowering of the plantation

The target is the mean harvested yield (kg DM/ha per harvest) over all the plantations (cycles) of the
management. The N amounts are searched in steps of ``NITROGEN_SCALE`` kg/ha and the months as integers.
"""
import glob
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from omegaconf import OmegaConf

from spatialdata.soil_data import TEXTURE_CLASSES
from spatialdata.utils import read_compressed_xarray

from ..caf.management import CAFManagement
from ..caf.output import CAFOutputData
from ..caf.trees import Tree
from ..spatial_cm import SpatialCAF
from .ferti_optimization import _rmtree_retry

NITROGEN_SCALE = 10          # kg N/ha per unit of the nK search bounds
DAYS_PER_MONTH = 30
ESTABLISHMENT_YEARS = 2      # years after planting that receive the (month0, n0) application
MIN_RADIATION = 1.0          # MJ m-2 d-1; CAF returns NaN after a day of 0 radiation
SIM_FOLDER = 'n_opt'         # per texture folder where the fertilised runs are done
ENV_FILES = ['cafdem.csv', 'cafsoil.csv', 'cafweather.csv']

DEFAULT_SCHEDULE = {
    'month0': (5, 9), 'n0': (0, 15),
    'month1': (1, 2), 'n1': (0, 30),
    'month2': (4, 6), 'n2': (0, 30),
}


def coffee_management(config, cultivar: str = 'sun', duration: int = 7, interval_years: int = 0,
                      end_date: str = '2024-12-31') -> Tuple[List[str], Dict]:
    """Planting dates and management of the plantations, built as in ``modeladocultivo_cafe``.

    Parameters
    ----------
    config : DictConfig
        Coffee configuration; ``MANAGEMENT.starting_date`` and ``MANAGEMENT.n_cycles`` are used.
    cultivar : str
        Shade system ('sun' or a tree species of ``caf.trees.Tree``).
    duration : int
        Life of a plantation in years.
    interval_years : int
        Years between plantations; 0 spreads ``n_cycles`` plantations over the period.
    end_date : str
        Last day with weather data.

    Returns
    -------
    tuple
        Planting dates ('%Y-%m-%d') and the management dictionary (unfertilised).
    """
    organizer = CAFManagement()
    starting_date = config.MANAGEMENT.get('starting_date', '1991-04-01')
    if interval_years:
        planting_dates = organizer.planting_dates_from_interval(starting_date, end_date, interval_years, duration)
    else:
        planting_dates = organizer.planting_dates_from_aperiod(
            starting_date, end_date, config.MANAGEMENT.get('n_cycles', 1), coffe_plant_duration=duration)
    management = organizer.create_config_template(planting_dates, starting_date=starting_date,
                                                  coffe_plant_duration=duration)
    management['TREE'] = {'species_name': cultivar, 'tree_density': Tree().tree_density(cultivar)}
    return planting_dates, management


def get_applications_from_kwargs(**kwargs) -> Tuple[List[int], List[int]]:
    """Months and N units (month0, n0, month1, n1, ...) of a point of the search space, rounded."""
    months, n_units = [], []
    n = 0
    while f'month{n}' in kwargs:
        months.append(int(round(kwargs[f'month{n}'])))
        n_units.append(int(round(kwargs[f'n{n}'])))
        n += 1
    return months, n_units


class CAFFertilizerBayesian(SpatialCAF):
    """Nitrogen application optimisation for coffee (CAF2021) using Bayesian optimisation.

    Parameters
    ----------
    configuration_dict : dict, optional
        Coffee configuration.
    configuration_path : str, optional
        Path to the coffee configuration file.
    load_env_data : bool
        Load the national soil, weather and dem datacubes (not needed with village packages).
    cultivar : str
        Shade system ('sun' or a tree species).
    duration : int
        Life of a plantation in years.
    interval_years : int
        Years between plantations; 0 spreads ``MANAGEMENT.n_cycles`` plantations over the period.
    """

    def __init__(self, configuration_dict: Optional[Dict] = None, configuration_path: Optional[str] = None,
                 load_env_data: bool = False, cultivar: str = 'sun', duration: int = 7, interval_years: int = 0):
        config = OmegaConf.load(configuration_path) if configuration_dict is None else configuration_dict
        planting_dates, management = coffee_management(config, cultivar=cultivar, duration=duration,
                                                       interval_years=interval_years)
        SpatialCAF.__init__(self, planting_date=planting_dates, configuration_dict=config,
                            load_env_data=load_env_data)
        self.config.MANAGEMENT = management
        self.cultivar_name = cultivar
        self._cm_envs = []
        self._evaluations = {}
        self.application_schedule = dict(DEFAULT_SCHEDULE)

    def set_application_schedule(self, application_schedule: Optional[Dict] = None) -> None:
        """Set the search bounds {'month0': (lo, hi), 'n0': (lo, hi), ...}; nK in NITROGEN_SCALE kg/ha."""
        if application_schedule is None:
            application_schedule = DEFAULT_SCHEDULE
        elif OmegaConf.is_config(application_schedule):
            application_schedule = OmegaConf.to_container(application_schedule)
        self.application_schedule = {k: tuple(v) for k, v in application_schedule.items()}
        self._evaluations = {}

    @property
    def n_applications(self) -> int:
        return len(self.application_schedule) // 2

    def set_enviromental_information(self, geocode: str) -> List[str]:
        """Create the CAF files (one folder per texture class) of a village.

        Parameters
        ----------
        geocode : str
            Village code (GEOCODIGO).

        Returns
        -------
        list
            Folders of the texture classes with CAF files.
        """
        geocode = str(geocode)
        roi = self.geo_features.loc[self.geo_features['GEOCODIGO'].astype(str) == geocode]
        if roi.empty:
            raise ValueError(f'{geocode} not in the village shapefile')
        village_info = self.config.SPATIAL_INFO.get('villages_folderpath', '')
        self.set_up_folders(site=geocode)
        datauploaded = False
        if not os.path.exists(self._dem_tmppath) and os.path.exists(os.path.join(village_info, geocode)):
            datauploaded = True
            for dataset in ['weather', 'soil', 'dem']:
                fn = os.path.basename(self.__dict__[f'_{dataset}_tmppath']).replace('.nc', '.zip')
                read_compressed_xarray(os.path.join(village_info, geocode, fn), self._tmp_path)

        self.create_roi_sp_data(roi=roi, group_codes=TEXTURE_CLASSES, export_spatial_data=not datauploaded)
        self._cm_envs = self.model.find_envworking_paths(self._tmp_path, 'csv')
        # the gridded radiation has days of 0 MJ/m2: CAF then returns NaN for the rest of the run
        for env_path in self._cm_envs:
            fn = os.path.join(env_path, 'cafweather.csv')
            weather = pd.read_csv(fn)
            if (weather['GR'] < MIN_RADIATION).any():
                weather['GR'] = weather['GR'].clip(lower=MIN_RADIATION)
                weather.to_csv(fn, index=False)
        return self._cm_envs

    def mean_harvest_yield(self, output: pd.DataFrame) -> float:
        """Mean yield of the harvests of all the plantations (kg DM/ha per harvest)."""
        cols = self._colnames.growth_colnames
        harvests = output.loc[output[cols['hdate']] != 0, cols['yield']].to_numpy(dtype=float)
        harvests = harvests[~np.isnan(harvests)]
        return float(harvests.mean()) if len(harvests) else 0.0

    def _clean_run_files(self, path: str) -> None:
        """Remove the outputs of a run; PyCAF does not overwrite an existing output_<cycle>.csv."""
        for fn in glob.glob(os.path.join(path, 'output_*.csv')):
            os.remove(fn)
        for n_cycle in range(self.config.MANAGEMENT.n_cycles):
            if os.path.exists(os.path.join(path, f'_{n_cycle}')):
                _rmtree_retry(os.path.join(path, f'_{n_cycle}'))

    def baseline_simulation(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
        """Run the unfertilised management on every texture class.

        Returns
        -------
        tuple
            Flowering dates (columns n_cycle and HDAT) and mean harvest yield, both by texture class.
        """
        cols = self._colnames.growth_colnames
        self.set_fertilization(None, [], [])
        self.model._process_paths = list(self._cm_envs)
        for env_path in self._cm_envs:
            self._clean_run_files(env_path)
        completed = self.run_caf(setup_envs=False, verbose=False)
        flowering_dates, baseline_yields = {}, {}
        for env_path in self._cm_envs:
            texture = os.path.basename(env_path)
            if completed.get(texture):
                output = CAFOutputData(env_path).output_data().sort_values(cols['date'])
                flowering_dates[texture] = output.loc[output[cols['flowering_date']] != 0,
                                                      [cols['number_of_cycle'], cols['date']]]
                baseline_yields[texture] = self.mean_harvest_yield(output)
            self._clean_run_files(env_path)
        return flowering_dates, baseline_yields

    def set_fertilization(self, flowering_dates: Optional[pd.DataFrame], months: List[int],
                          n_amounts: List[float]) -> None:
        """Write a fertilisation plan in the management of every plantation.

        Parameters
        ----------
        flowering_dates : pd.DataFrame
            Flowering dates of the unfertilised run (columns n_cycle and HDAT).
        months : list
            [month0, month1, ...]: month of the year of the establishment application and months after
            flowering of the other applications.
        n_amounts : list
            [n0, n1, ...] in kg N/ha per application.
        """
        n_cycles = self.config.MANAGEMENT.n_cycles
        if len(months) > 1:
            ferti_days = self.ferti_days_after_flowering(
                flowering_dates, [m * DAYS_PER_MONTH for m in months[1:]], n_amounts[1:])
        else:
            ferti_days = {z + 1: [[], []] for z in range(n_cycles)}

        for z in range(n_cycles):
            schedule = self.create_event_cycle_fert_schedule(z, *ferti_days[z + 1])
            if len(months) > 0 and n_amounts[0] > 0:
                for year in range(ESTABLISHMENT_YEARS):
                    schedule['years'].append(self.pdates[z].year + year)
                    schedule['dayofyear'].append(months[0] * DAYS_PER_MONTH)
                    schedule['N_amount'].append(n_amounts[0])
            if len(schedule['years']) == 0:   # CAF needs at least one event
                schedule = self.create_event_cycle_fert_schedule(z, [0], [0])
            self.config.MANAGEMENT[f'cycle_treatment_{z + 1}']['fertilization'] = schedule

    def yield_simulation(self, env_path: str, flowering_dates: pd.DataFrame, months: List[int],
                         n_amounts: List[float]) -> float:
        """Mean harvest yield of a texture class with a fertilisation plan."""
        sim_path = os.path.join(env_path, SIM_FOLDER)
        os.makedirs(sim_path, exist_ok=True)
        for fn in ENV_FILES:
            if not os.path.exists(os.path.join(sim_path, fn)):
                shutil.copy2(os.path.join(env_path, fn), sim_path)
        self._clean_run_files(sim_path)
        self.set_fertilization(flowering_dates, months, n_amounts)
        self.model._process_paths = [sim_path]
        try:
            self.run_caf(setup_envs=False, verbose=False)
            return self.mean_harvest_yield(CAFOutputData(sim_path).output_data())
        finally:
            self._clean_run_files(sim_path)

    def _target(self, env_path: str, flowering_dates: pd.DataFrame, **kwargs) -> float:
        months, n_units = get_applications_from_kwargs(**kwargs)
        key = tuple(months + n_units)
        if key not in self._evaluations:   # the search space is continuous, the plans are integers
            try:
                self._evaluations[key] = self.yield_simulation(
                    env_path, flowering_dates, months, [n * NITROGEN_SCALE for n in n_units])
            except (subprocess.CalledProcessError, AssertionError) as exc:
                print(f'CAF failed for {key}: {exc}')
                return 0.0
        return self._evaluations[key]

    def organize_results(self) -> pd.DataFrame:
        """Evaluated plans as a table: monthK, nK (kg N/ha), crop_yield, nit_ayear3 and nit_year1_2."""
        napp = self.n_applications
        rows = []
        for key, crop_yield in self._evaluations.items():
            row = {f'month{i}': key[i] for i in range(napp)}
            row.update({f'n{i}': key[napp + i] * NITROGEN_SCALE for i in range(napp)})
            row['crop_yield'] = crop_yield
            rows.append(row)
        results = pd.DataFrame(rows)
        # N per flowering (yearly once the plantation produces) and yearly N of the establishment years
        results['nit_ayear3'] = results[[f'n{i}' for i in range(1, napp)]].sum(axis=1)
        results['nit_year1_2'] = results['n0']
        return results

    def simulate_env(self, env: int, flowering_dates: pd.DataFrame, n_iter: int = 40, init_points: int = 10,
                     random_state: int = 123, verbose: int = 0, remove_sim_folder: bool = True) -> pd.DataFrame:
        """Optimise the fertilisation plan of one texture class.

        Parameters
        ----------
        env : int
            Index of the texture class in ``_cm_envs``.
        flowering_dates : pd.DataFrame
            Flowering dates of the unfertilised run of that texture class.
        n_iter : int
            Iterations of the Bayesian optimisation.
        init_points : int
            Random plans evaluated before the optimisation.

        Returns
        -------
        pd.DataFrame
            One row per evaluated plan (see ``organize_results``).
        """
        env_path = self._cm_envs[env]
        self._evaluations = {}
        optimizer = BayesianOptimization(
            f=lambda **kwargs: self._target(env_path, flowering_dates, **kwargs),
            pbounds=self.application_schedule,
            random_state=random_state,
            verbose=verbose,
            allow_duplicate_points=True,
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        if remove_sim_folder:
            _rmtree_retry(os.path.join(env_path, SIM_FOLDER))
        results = self.organize_results()
        results.to_csv(os.path.join(self._tmp_path, f'{os.path.basename(env_path)}_{self.n_applications}.csv'),
                       index=False)
        return results
