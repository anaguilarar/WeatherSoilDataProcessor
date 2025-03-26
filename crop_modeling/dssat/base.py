import concurrent.futures
import glob
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from DSSATTools.base.sections import Section, clean_comments
from DSSATTools.crop import CROPS_MODULES, VERSION, Crop
from DSSATTools.run import CONFILE

from ._base import DSSATFiles, coords_from_soil_file, section_indices
from .files_export import from_soil_to_dssat, from_weather_to_dssat
from .files_reading import (delimitate_header_indices,
                           join_row_using_header_indices)
from .management import DSSATManagement_base
from ..utils.model_base import ModelBase

def check_soil_id(management_pathfile:str, new_soil_id:str):
    

    lines = DSSATFiles.open_file(management_pathfile)
    section_id = list(section_indices(lines, pattern= '*FIELDS'))[0]+1
    section_header_str = lines[section_id]
    header_indices = delimitate_header_indices(section_header_str)
    data_rows = []
    for section_data_str in lines[(section_id+1):(section_id+2)]:
        data_rows.append([section_data_str[i:j].strip()
                    for i, j in header_indices])

    sec_header = section_header_str.split()
    
    datalineinfo = pd.DataFrame(data=data_rows, columns=sec_header)
    
    if datalineinfo.ID_SOIL.values[0] == '-99':
        datalineinfo.ID_SOIL = new_soil_id
        datalineinfo.FLNAME = '-99'
        section_data_str = lines[(section_id+1)]

        lines[section_id+1] = join_row_using_header_indices(section_header_str, section_data_str, row_to_replace = datalineinfo.values[0])
    
        with open(management_pathfile, 'w') as file:
            for line in lines:
                file.write(f"{line}")
                

def create_dssat_tmp_env(source_path, tmp_path, exp_file):
    if not os.path.exists(tmp_path): os.mkdir(tmp_path)
    
    wth = 'WTHE0001.WTH'
    soil = 'TR.SOL'
    crop = glob.glob(source_path+'/*.CUL*')
    eco = glob.glob(source_path+'/*.ECO*')
    spe = glob.glob(source_path+'/*.SPE*')
    
    
    shutil.copy2(os.path.join(source_path, soil), tmp_path)
    shutil.copy2(os.path.join(source_path, exp_file), tmp_path)
    shutil.copy2(os.path.join(source_path, wth), tmp_path)
    shutil.copy2(crop[0], tmp_path)
    shutil.copy2(eco[0], tmp_path)
    shutil.copy2(spe[0], tmp_path)

def create_dssat_config_path_file(workdir_path,  crop_code, crop, dssat_path, wth_path, crd_path, soil_path, std_path) -> None:
    
    crop_model = 'CRGRO' if crop_code == 'BN' else CROPS_MODULES[crop]
    
    with open(os.path.join(workdir_path, CONFILE), 'w') as f:
        f.write(f'WED    {wth_path}\n')
        f.write(f'M{crop_code}    {workdir_path} dscsm048 {crop_model}{VERSION}\n')
        f.write(f'CRD    {crd_path}\n')
        f.write(f'PSD    {os.path.join(dssat_path, "Pest")}\n')
        f.write(f'SLD    {soil_path}\n')
        f.write(f'STD    {std_path}\n')

def check_exp_summary_name(workdir_path, run_path, experiment_id, removeworking_path_folder: bool = False):
    
    if os.path.exists(
        os.path.join(workdir_path,'Summary.OUT')):
        valtoreturn = {os.path.basename(run_path): True}
        if not os.path.exists(os.path.join(run_path, f'Summary_{experiment_id}.OUT')):
            shutil.copyfile(os.path.join(workdir_path, 'Summary.OUT'), os.path.join(run_path,f'Summary_{experiment_id}.OUT'))
        if removeworking_path_folder:
            shutil.rmtree(workdir_path, ignore_errors=False, onerror=None)
    else:
        valtoreturn = {os.path.basename(run_path): False}
        
    return valtoreturn

def run_dssat_simulation(experiment_id, path, crop_code, crop, dssat_home, dssat_bin_path, crd_path, sl_path, std_path, sim_experiment_path):
    
    exp_pathfile = glob.glob(path+'/*.{}X*'.format(crop_code))
    
    if len(exp_pathfile)==0:
        print(' There is no experimental file, please generated first')
        return {os.path.basename(path): False}
    if len(exp_pathfile)>1:
        print(' There are more than one experimental file, please only leave one')
        return {os.path.basename(path): False}
    
    exp_pathfile = os.path.basename(exp_pathfile[0])
    dssat_home = os.path.dirname(dssat_bin_path) if dssat_home is None else dssat_home
    wth_path = os.path.join(sim_experiment_path,'WTHE0001')
    create_dssat_tmp_env(path, sim_experiment_path, exp_pathfile)
    create_dssat_config_path_file(sim_experiment_path, crop_code, crop, dssat_home, wth_path, crd_path, sl_path, std_path)
    
    subprocess.run([dssat_bin_path, 'C', exp_pathfile, str(experiment_id)],
                    cwd=sim_experiment_path, capture_output=True, text=True,
                    env={"DSSAT_HOME": dssat_home, })


def run_experiment_dssat(path, experimentid,crop_code, crop,bin_path = None, dssat_path = None, remove_folder = False, sim_experiment_path = None):
    sim_experiment_path = sim_experiment_path or os.path.join(path, f'_{experimentid}')
    
    run_dssat_simulation(experimentid, path, crop_code, crop, dssat_path, bin_path, os.path.join(dssat_path, 'Genotype'),
            os.path.join(dssat_path, 'Soil'),os.path.join(dssat_path, 'StandardData'),sim_experiment_path)
    
    proccess_finsihed = check_exp_summary_name(sim_experiment_path, path, experimentid, removeworking_path_folder=remove_folder)
    
    return proccess_finsihed


def run_experiment_dssat_bin(path, experimentid,crop_code, crop, remove_folder = False, sim_experiment_path = None):
    from DSSATTools.run import CRD_PATH, SLD_PATH, DSSAT_HOME, BIN_PATH, STD_PATH
    
    sim_experiment_path = sim_experiment_path or os.path.join(path, f'_{experimentid}')
    
    run_dssat_simulation(experimentid, path, crop_code, crop, DSSAT_HOME, BIN_PATH, CRD_PATH,
            SLD_PATH,STD_PATH,sim_experiment_path)
    
    proccess_finsihed = check_exp_summary_name(sim_experiment_path, path, experimentid, removeworking_path_folder=remove_folder)
    
    return proccess_finsihed


class DSSATBase(ModelBase):
    """
    A class for managing DSSAT-related file processing, configuration, and execution.

    Provides methods for setting up DSSAT experiments, converting data from other formats to DSSAT-compatible files, 
    and running simulations using R scripts.
    """
    def __init__(self, path):
        super().__init__(path)
    
    def set_up_management(
        self,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        planting_date: Optional[str] = None,
        harvesting_date: Optional[str] = None,
        template: Optional[str] = None,
        roi_id: int = 1,
        plantingWindow: Optional[int] = None,
        index_soilwat: int = 1,
        fertilizer_schedule = None
        ) -> None:
        """
        Set up the management configuration and create DSSAT experiment files.

        Parameters
        ----------
        crop : str, optional
            Crop name.
        cultivar : str, optional
            Crop cultivar identifier.
        planting_date : str, optional
            Planting date in 'YYYY-MM-DD' format.
        harvesting_date : str, optional
            Harvesting date in 'YYYY-MM-DD' format.
        template : str, optional
            Path to the experiment template file.
        roi_id : int, default=1
            Region of interest ID.
        plantingWindow : int, optional
            Planting window in days.
        fertilizer : bool, optional
            Whether to include fertilizer in the configuration.
        index_soilwat : int, default=1
            Soil water index for the experiment.
        """
        #if len(self._process_paths) == 0: self.find_envworking_paths()
        assert len(self._process_paths) > 0, "Soil and weather data must be obtained first."

        dssatm = DSSATManagement_base(path = template, crop = crop, variety = cultivar, 
                                planting_date=planting_date, harvesting_date= harvesting_date,
                                n_planting_windows = plantingWindow)
        
        for pathiprocess in self._process_paths:

            soil = glob.glob(pathiprocess+'/*.SOL*')
            lat, long = coords_from_soil_file(soil[0])
            
            output = dssatm.create_file(template, pathiprocess, roi_id = roi_id,plantingWindow = plantingWindow,
                    index_soilwat = index_soilwat, fertilizer_schedule =fertilizer_schedule,
                    long = long, lat = lat)
            self.crop_code = dssatm.crop_code
            if output is None:
                continue
            experiment_config = OmegaConf.load(os.path.join(pathiprocess, 'experimental_file_config.yaml'))
            
            management_pathfile = glob.glob(pathiprocess+'/*.{}*'.format(experiment_config.MANAGEMENT.crop_code))
            print(f'experimental file created: {management_pathfile}')
            
            check_soil_id(management_pathfile[0], experiment_config.SOIL.ID_SOIL )
            
    
    def set_up_crop(self, crop: str, cultivar: str, cultivar_template: str = None) -> None:
        """
        Set up crop and cultivar configurations.

        Parameters
        ----------
        crop : str
            Crop name.
        cultivar : str
            Crop cultivar code.
        cultivar_template : str
            Path to the cultivar template file.
        """

        crop_manager = DSSATCrop_base(crop.lower(), cultivar_code=cultivar)
        if cultivar_template is not None:
            crop_manager.update_cultivar_using_path(cultivar_template)
        for pathtiprocess in self._process_paths:
            crop_manager.write(pathtiprocess)
    
    def run(self, crop_code, crop, planting_window, bin_path = None, parallel_tr= True, ncores = 10, remove_tmp_folder = False, dssat_path:str = None, sim_experiment_path = None) -> None:
        
        """
        Run DSSAT simulations for all processing paths.

        Parameters
        ----------
        crop_code : str
            Crop code for DSSAT simulations.
        planting_window : int
            Number of treatments to simulate.
        bin_path : str, optional
            Path to the DSSAT executable.
        parallel_tr : bool, default=True
            Whether to run treatments in parallel.
        ncores : int, default=10
            Number of cores to use for parallel processing.
        remove_tmp_folder : bool, default=False
            Whether to remove temporary folders after simulations.
        dssat_path: str, optional
            Path to DSSAT folder.

        Returns
        -------
        Dict[str, bool]
            Dictionary with processing path names as keys and success status as values.
        """
        process_completed = {}
        if parallel_tr:
            for pathiprocess in tqdm(self._process_paths):
                if not os.path.exists(os.path.join(pathiprocess, 'TR.SOL')): 
                        print(f'soil file not found in :{pathiprocess}')
                        process_completed[os.path.basename(pathiprocess)] = False 
                        continue
                        
                file_path_pertr = {}

                with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
                        if bin_path is None:
                            future_to_tr ={executor.submit(run_experiment_dssat_bin, pathiprocess, i, 
                                                        crop_code,crop, remove_folder = remove_tmp_folder, sim_experiment_path = sim_experiment_path): (i) for i in range(1,planting_window+1)}
                        else:
                            future_to_tr ={executor.submit(run_experiment_dssat, pathiprocess, i, 
                                                        crop_code,crop, bin_path, dssat_path, remove_folder = remove_tmp_folder, sim_experiment_path = sim_experiment_path): (i) for i in range(1,planting_window+1)}

                        for future in concurrent.futures.as_completed(future_to_tr):
                                tr = future_to_tr[future]
                                try:
                                    file_path_pertr[str(tr)] = future.result()
                                        
                                except Exception as exc:
                                    print(f"Request for treatment {tr} generated an exception: {exc}")
                                        
                process_completed[os.path.basename(pathiprocess)] = any([v[list(v.keys())[0]] for k,v in file_path_pertr.items()])
            
        else:
            
            for pathiprocess in self._process_paths:
                if not os.path.exists(os.path.join(pathiprocess, 'TR.SOL')): 
                            print(f'soil file not found in :{pathiprocess}')
                            process_completed[os.path.basename(pathiprocess)] = False 
                            continue
                file_path_pertr = {}
                
                for tr in range(1,planting_window+1):
                    if bin_path is None:
                        file_path_pertr[str(tr)] = run_experiment_dssat_bin(pathiprocess, tr, 
                                                crop_code,crop, remove_folder = remove_tmp_folder, sim_experiment_path = sim_experiment_path)
                    else:
                        file_path_pertr[str(tr)] = run_experiment_dssat(pathiprocess, tr, 
                                                            crop_code,bin_path, remove_folder = remove_tmp_folder, sim_experiment_path = sim_experiment_path)
            
                process_completed[os.path.basename(pathiprocess)] = any([v[list(v.keys())[0]] for k,v in file_path_pertr.items()])

        return process_completed

    def from_datacube_to_files(
            self,
            dfdata,
            data_source: str = 'climate',
            dim_name: str = None,
            group_by: Optional[str] = None,
            group_codes: Optional[dict] = None,
            outputpath: str = None,
            country = None,
            site = None,
            sub_working_path = None,
            verbose = True
        ) -> pd.DataFrame:
        """
        Converts data from a datacube to DSSAT-compatible files.

        Parameters
        ----------
        xrdata : xarray.DataArray or xarray.Dataset
            Input data in a datacube format.
        data_source : str, default='climate'
            Source of data ('climate' or 'soil').
        dim_name : str, default='date'
            Name of the time dimension in the datacube.
        target_crs : str, default='EPSG:4326'
            Target coordinate reference system.
        group_by : str, optional
            Grouping column for data aggregation.
        group_codes : dict, optional
            Group codes for the data.
        sub_workingdict_path: str, optional
            Sub directory taht will be allocate the dssat files
        Returns
        -------
        pd.DataFrame
            Data summarized as a DataFrame.
        """
        
        if not dim_name:
            dim_name = 'date' if data_source == 'climate' else 'depth'
        
        if data_source == 'climate':
            from_weather_to_dssat(dfdata, date_name = dim_name, group_by = group_by,
                        outputpath=outputpath, outputfn = 'WTHE0001', codes=group_codes, 
                        sub_working_path= sub_working_path)
        
        if data_source == 'soil':
            from_soil_to_dssat(dfdata, group_by=group_by, depth_name= dim_name,
                                    outputpath=outputpath, outputfn='SOL', codes=group_codes, 
                                    country = country,site = site, soil_id='TRAN00001', 
                                    sub_working_path = sub_working_path, verbose = verbose)
        
        
class DSSATCrop_base(Crop):
    
    def __init__(self, crop_name: str = 'Maize', cultivar_code: str = None):
        super().__init__(crop_name, cultivar_code)
        self.orig_cultivar_code = cultivar_code
        if self.orig_cultivar_code!= self.cultivar_code: print(f"Update with a new genotype file")
        
    
    def update_cultivar_using_path(self, genotype_path):
        with open(genotype_path, 'r') as f:
            file_lines = f.readlines()
            
        file_lines = clean_comments(file_lines)

        newcult = Section(
            name="cultivar", file_lines=file_lines, crop_name=self.crop_name,
            code=self.orig_cultivar_code
        )
        
        self.cultivar = newcult
        print(f'updated: {self.cultivar}')
        
        eco_file = genotype_path[:-3] + 'ECO'
        with open(eco_file, 'r') as f:
                file_lines = f.readlines()
        file_lines = clean_comments(file_lines)
            
        self.ecotype = Section(
                name="ecotype", file_lines=file_lines, crop_name=self.crop_name,
                code=self.cultivar["ECO#"]
            )