
import os
from DSSATTools.base.sections import Section, clean_comments
from DSSATTools.crop import Crop
from typing import List
import pandas as pd
from typing import Optional
from pathlib import Path
from ..utils.process import summarize_datacube_as_df

from .files_export import from_soil_to_dssat,from_weather_to_dssat
from .management import DSSATManagement_base
import glob
from .files_reading import delimitate_header_indices, join_row_using_header_indices
from omegaconf import OmegaConf
import subprocess
from ._base import DSSATFiles, section_indices, coords_from_soil_file
from multiprocessing import Pool
import concurrent.futures
import shutil


def check_soil_id(management_pathfile, new_soil_id):
    

    lines = DSSATBase.open_file(management_pathfile[0])
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
    
        with open(management_pathfile[0], 'w') as file:
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
    
    
    

  
def create_DSSBatch(ExpFilePath: str, selected_treatments: Optional[list[str]]=None, 
                    command: str = 'DSCSM048.EXE Q DSSBatch.v48'):
    """
    Create DSSBatch file using DSSAT X file

    :param         ExpFilePath: DSSAT X file complete path in str or Path
    :param selected_treatments: Treatments selected from the X file in list
    :param             command: DSSAT command to run dssat, defaults to 
                                'DSCSM048.EXE Q DSSBatch.v48'
    :return: None

    """
    ExpFilePath = Path(ExpFilePath)
    ExpDir = ExpFilePath.parent
    ExpFile = ExpFilePath.name
    v = command.split('.EXE')[0][-2:]
    DSSBatchFile = ExpDir / ('DSSBatch.v'+v)
    
    treatments_text = ''
    TRTNO, SQ, OP, CO, TNAME = [], [], [], [], []
    with open(ExpFilePath) as Fexp:
        param = 0
        for line in Fexp.readlines():
            if line.startswith('@N R O C TNAME...'):
                param = 1
                continue
            if param == 1 and line.startswith('\n'):
                break
            if param == 1:
                treatments_text = line
                if selected_treatments is not None and \
                treatments_text[9:33].strip() in selected_treatments:
                    TRTNO.append(treatments_text[:2])
                    SQ.append(treatments_text[2:4])
                    OP.append(treatments_text[4:6])
                    CO.append(treatments_text[6:8])
                    TNAME.append(treatments_text[9:33])
                else:
                    TRTNO.append(treatments_text[:2])
                    SQ.append(treatments_text[2:4])
                    OP.append(treatments_text[4:6])
                    CO.append(treatments_text[6:8])
                    TNAME.append(treatments_text[9:33])
    treatment_df = pd.DataFrame({'TRTNO' : TRTNO, 'SQ' : SQ,
                                 'OP': OP, 'CO': CO})
    batch_text = '$BATCH(%s)' % ('Sequence'.upper()) + '\n' + '!' + '\n'
    batch_text = batch_text + '@FILEX                                                                                        TRTNO     RP     SQ     OP     CO\n'
    
    for row in range(treatment_df.shape[0]):
        batch_text = ''.join([batch_text, 
                              ExpFile.ljust(94),
                              treatment_df.loc[row, 'TRTNO'].rjust(5), 
                              treatment_df.loc[row, 'OP'].rjust(7),
                              treatment_df.loc[row, 'SQ'].rjust(7),
                              treatment_df.loc[row, 'OP'].rjust(7),
                              treatment_df.loc[row, 'CO'].rjust(7),
                              '\n'])                                            # type: ignore
    with open(DSSBatchFile, 'w') as Fbatch:
        Fbatch.write(batch_text)
    return None

def run_batch_dssat(path, crop_code, bin_path):
    #bin_path = 'C:/DSSAT48/DSCSM048.exe'
    soil = glob.glob(path+'/*.SOL*')
    if len(soil)==0:
        print(f'soil file not found in :{path}')
        #pathiprocess[os.path.basename(pathiprocess)] = False 
        return {os.path.basename(path): False}
    if not os.path.exists(os.path.join(path, 'TR.SOL')): 
        os.rename(soil[0], os.path.join(path, 'TR.SOL'))
    
    exp_pathfile = glob.glob(path+'/*.{}X*'.format(crop_code))
    create_DSSBatch(exp_pathfile[0])

    batch_pathfile = glob.glob(path+'/*.V48*')
    subprocess.call([bin_path,'B' , os.path.basename(batch_pathfile[0])] ,
                    shell= True, cwd=path,
                    
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return {os.path.basename(path): os.path.exists(
        os.path.join(path,'Summary.OUT'))}
            

def run_experiment_dssat(path, experimentid,crop_code, bin_path, remove_folder = False):
    exp_pathfile = glob.glob(path+'/*.{}X*'.format(crop_code))
    if len(exp_pathfile)==0:
        print(' There is no experimental file, please generated first')
        return {os.path.basename(path): False}
    if len(exp_pathfile)>1:
        print(' There are more than one experimental file, please only leave one')
        return {os.path.basename(path): False}
        
    exp_pathfile = os.path.basename(exp_pathfile[0])
    #create_DSSBatch(exp_pathfile[0])
    
    tmppath = os.path.join(path, f'_{experimentid}')
    create_dssat_tmp_env(path, tmppath, exp_pathfile)
    
    subprocess.call([bin_path, 'C', exp_pathfile, str(experimentid)],
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    shell= True, cwd=tmppath)
    if os.path.exists(
        os.path.join(tmppath,'Summary.OUT')):
        valtoreturn = {os.path.basename(path): True}
        if not os.path.exists(os.path.join(path, f'Summary_{experimentid}.OUT')):
           shutil.copyfile(os.path.join(tmppath, 'Summary.OUT'), os.path.join(path,f'Summary_{experimentid}.OUT'))
        if remove_folder:
            shutil.rmtree(tmppath, ignore_errors=False, onerror=None)
        
    else:
        valtoreturn = {os.path.basename(path): False}
    return valtoreturn


class DSSATBase(DSSATFiles):
    """
    A class for managing DSSAT-related file processing, configuration, and execution.

    Provides methods for setting up DSSAT experiments, converting data from other formats to DSSAT-compatible files, 
    and running simulations using R scripts.
    """
    def __init__(self, path: str) -> None:
        """
        Initialize the DSSABase object.

        Parameters
        ----------
        path : str
            Path to the working directory.
        """
        self.path = path
        self._tmp_path = ""
        self._process_paths: List[str] = []
        
    def set_up(self, **kwargs) -> None:
        """
        Set up the working directory, temporary paths, and general information about the site and country.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to configure site and country.

            - site : str, optional
                Site name to be used as a subdirectory for temporary files.
            - country : str, optional
                Country code or name for soil data.
        """
        self._process_paths = None
        assert os.path.exists(self.path)
        
        self.site = kwargs.get('site', None)
        if self.site is None:
            self._tmp_path = os.path.join(self.path, 'tmp')
        else:
            self._tmp_path = os.path.join(self.path, self.site)
            
        if not os.path.exists(self._tmp_path): os.mkdir(self._tmp_path)
        
        self.country = kwargs.get('country', None)
    
    def set_up_management(
        self,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        planting_date: Optional[str] = None,
        harvesting_date: Optional[str] = None,
        template: Optional[str] = None,
        roi_id: int = 1,
        plantingWindow: Optional[int] = None,
        fertilizer: Optional[bool] = None,
        index_soilwat: int = 1,
        use_r = False
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
        if self._process_paths is None: self.find_envworking_paths()
        assert len(self._process_paths) > 0, "Soil and weather data must be obtained first."

        dssatm = DSSATManagement_base(path = template, crop = crop, variety = cultivar, 
                                planting_date=planting_date, harvesting_date= harvesting_date,
                                n_planting_windows = plantingWindow)
        
        for pathiprocess in self._process_paths:
            if use_r:
                
                output = dssatm.create_file_using_rdssat(template, pathiprocess, roi_id = roi_id, plantingWindow = plantingWindow, 
                                            fertilizer = fertilizer, index_soilwat = index_soilwat)
            else:
                
                soil = glob.glob(pathiprocess+'/*.SOL*')
                lat, long = coords_from_soil_file(soil[0])
                
                output = dssatm.create_file(template, pathiprocess, roi_id = roi_id,plantingWindow = plantingWindow,
                        fertilizer = fertilizer, index_soilwat = index_soilwat,
                        long = long, lat = lat)
                self.crop_code = dssatm.crop_code

            if output is None:
                continue
            experiment_config = OmegaConf.load(os.path.join(pathiprocess, 'experimental_file_config.yaml'))
            
            management_pathfile = glob.glob(pathiprocess+'/*.{}*'.format(experiment_config.MANAGEMENT.crop_code))
            print(f'experimental file created: {management_pathfile}')
            
            check_soil_id(management_pathfile, experiment_config.SOIL.ID_SOIL )
            
    
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
    
    def run(self, crop_code, planting_window, bin_path = 'C:/DSSAT48/DSCSM048.exe', parallel_tr= True, ncores = 10, remove_tmp_folder = False) -> None:
        process_completed = {}
        """
        Run DSSAT simulations for all processing paths.

        Parameters
        ----------
        crop_code : str
            Crop code for DSSAT simulations.
        planting_window : int
            Number of treatments to simulate.
        bin_path : str, default='C:/DSSAT48/DSCSM048.exe'
            Path to the DSSAT executable.
        parallel_tr : bool, default=True
            Whether to run treatments in parallel.
        ncores : int, default=10
            Number of cores to use for parallel processing.
        remove_tmp_folder : bool, default=False
            Whether to remove temporary folders after simulations.

        Returns
        -------
        Dict[str, bool]
            Dictionary with processing path names as keys and success status as values.
        """
        if parallel_tr:
            for pathiprocess in self._process_paths:
                    if not os.path.exists(os.path.join(pathiprocess, 'TR.SOL')): 
                            print(f'soil file not found in :{pathiprocess}')
                            process_completed[os.path.basename(pathiprocess)] = False 
                            continue
                            
                    file_path_pertr = {}

                    with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
                            future_to_tr ={executor.submit(run_experiment_dssat, pathiprocess, i, 
                                                            crop_code,bin_path, remove_folder = remove_tmp_folder): (i) for i in range(1,planting_window+1)}

                            for future in concurrent.futures.as_completed(future_to_tr):
                                    tr = future_to_tr[future]
                                    try:
                                            file_path_pertr[str(tr)] = future.result()
                                            
                                    except Exception as exc:
                                            print(f"Request for treatment {tr} generated an exception: {exc}")
                    
                    process_completed[os.path.basename(pathiprocess)] = any([v[list(v.keys())[0]] for k,v in file_path_pertr.items()])
                
        else:
            result = []
            for pathiprocess in self._process_paths:
                result.append(run_batch_dssat(pathiprocess, crop_code, bin_path))
            
        return process_completed
                
    def run_using_r(self) -> None:
        """
        Run DSSAT simulations using an R script.
        """
        for pathiprocess in self._process_paths:
            soil = glob.glob(pathiprocess+'/*.SOL*')
            if len(soil)==0: continue
            dirname = os.path.dirname(soil[0])
            if os.path.exists(os.path.join(dirname, 'TR.SOL')): os.remove(os.path.join(dirname, 'TR.SOL'))
            
            os.rename(soil[0], os.path.join(dirname, 'TR.SOL'))
            config_path = os.path.join(pathiprocess, 'experimental_file_config.yaml')
            returned_value = subprocess.call(['RScript', './r_scripts/r_run_dssat.R', f'{config_path}'] , shell= True)
            #if os.path.exists(os.path.join(dirname, 'TR.SOL')): os.remove(os.path.join(dirname, 'TR.SOL'))
    
    def find_envworking_paths(self):
        folders = [i for i in os.listdir(self._tmp_path) if os.path.isdir(os.path.join(self._tmp_path,i))]
        list_files = []
        for folder in folders: 
            pathsin = glob.glob(os.path.join(self._tmp_path, folder)+'/*.{}*'.format('WTH')) 
            if pathsin: list_files.append(pathsin[0])

        #list_files = glob.glob(self._tmp_path+'/*.{}*'.format('SOL'))
        self._process_paths = [os.path.dirname(fn) for fn in list_files]
        return self._process_paths
    
    def from_datacube_to_dssatfiles(
            self,
            xrdata,
            data_source: str = 'climate',
            dim_name: str = None,
            target_crs: str = 'EPSG:4326',
            group_by: Optional[str] = None,
            group_codes: Optional[dict] = None,
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

        Returns
        -------
        pd.DataFrame
            Data summarized as a DataFrame.
        """
        
        if not dim_name:
            dim_name = 'date' if data_source == 'climate' else 'depth'
            
        dfdata = summarize_datacube_as_df(xrdata, dimension_name= dim_name, group_by = group_by, project_to= target_crs)
        
        if data_source == 'climate':
            from_weather_to_dssat(dfdata, date_name = dim_name, group_by = group_by,
                        outputpath=self._tmp_path, outputfn = 'WTHE0001', codes=group_codes)
        
        if data_source == 'soil':
            from_soil_to_dssat(dfdata, group_by=group_by, depth_name= dim_name,
                                    outputpath=self._tmp_path, outputfn='SOL', codes=group_codes, 
                                    country = self.country.upper(),site = self.site, soil_id='TRAN00001')
            
        return dfdata

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

                