
import os
from DSSATTools.base.sections import Section, clean_comments
from DSSATTools.crop import Crop,GENOTYPE_PATH
from typing import List
import pandas as pd
from typing import Optional
from pathlib import Path
from ..utils.process import summarize_datacube_as_df

from .files_export import from_soil_to_dssat,from_weather_to_dssat
from .management import DSSATManagement_base
import glob
from .files_reading import section_indices, delimitate_header_indices, join_row_using_header_indices
from omegaconf import OmegaConf
import subprocess
from ._base import DSSATFiles

def check_soil_id(management_pathfile, new_soil_id):
    

    lines = DSSABase.open_file(management_pathfile[0])
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


class DSSABase(DSSATFiles):
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
        self._process_paths = []
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
        self.specific_paths()
        assert len(self._process_paths) > 0, "Soil and weather data must be obtained first."

        dssatm = DSSATManagement_base(crop, cultivar, 
                                planting_date=planting_date, harvesting_date= harvesting_date)
        
        for pathtiprocess in self._process_paths:
            output = dssatm.create_file_using_rdssat(template, pathtiprocess, roi_id = roi_id, plantingWindow = plantingWindow, 
                                            fertilizer = fertilizer, index_soilwat = index_soilwat)

            if output is None:
                continue
            experiment_config = OmegaConf.load(os.path.join(pathtiprocess, 'experimental_file_config.yaml'))
            
            management_pathfile = glob.glob(pathtiprocess+'/*.{}*'.format(experiment_config.MANAGEMENT.crop_code))
            print(f'experimental file created: {management_pathfile}')
            
            check_soil_id(management_pathfile, experiment_config.SOIL.ID_SOIL )
            
    
    def set_up_crop(self, crop: str, cultivar: str, cultivar_template: str) -> None:
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
        crop_manager.update_cultivar_using_path(cultivar_template)
        for pathtiprocess in self._process_paths:
            crop_manager.write(pathtiprocess)
    
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
    
    def specific_paths(self):
        path = self._tmp_path if self._tmp_path.endswith('/') else self._tmp_path+'/'
        list_files = glob.glob(path+'**/*.{}*'.format('WTH'),recursive=True)
        #list_files = glob.glob(self._tmp_path+'/*.{}*'.format('SOL'))
        self._process_paths = [os.path.dirname(fn) for fn in list_files]
    
    def from_datacube_to_dssatfiles(
            self,
            xrdata,
            data_source: str = 'climate',
            dim_name: str = 'date',
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

                