
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
        print(' new values', datalineinfo.values)
        
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


class DSSABase(object):
    
    def set_up(self, **kwargs):
        self._process_paths = []
        assert os.path.exists(self.path)
        
        self.site = kwargs.get('site', None)
        if self.site is None:
            self._tmp_path = os.path.join(self.path, 'tmp')
        else:
            self._tmp_path = os.path.join(self.path, self.site)
            
        if not os.path.exists(self._tmp_path): os.mkdir(self._tmp_path)
        
        self.country = kwargs.get('country', None)
    
    def set_up_management(self, crop = None, cultivar = None, planting_date = None, harvesting_date = None, 
                          template = None, roi_id = 1, plantingWindow = None, fertilizer = None, index_soilwat = 1):
        print(self._tmp_path)
        self.specific_paths()
        print(self._process_paths)
        assert len(self._process_paths)>0, 'soil and weather data must be obtained first'
        dssatm = DSSATManagement_base(crop, cultivar, 
                                planting_date=planting_date, harvesting_date= harvesting_date)
        
        for pathtiprocess in self._process_paths:
            dssatm.create_file_using_rdssat(template, pathtiprocess, roi_id = roi_id, plantingWindow = plantingWindow, 
                                            fertilizer = fertilizer, index_soilwat = index_soilwat)

            experiment_config = OmegaConf.load(os.path.join(pathtiprocess, 'experimental_file_config.yaml'))
            
            management_pathfile = glob.glob(pathtiprocess+'/*.{}*'.format(experiment_config.MANAGEMENT.crop_code))
            print(f'experimental file created: {management_pathfile}')
            
            check_soil_id(management_pathfile, experiment_config.SOIL.ID_SOIL )
    
    def set_up_crop(self, crop, cultivar, cultivar_template):

        crop = DSSATCrop_base(crop.lower(), cultivar_code=cultivar)
        crop.update_cultivar_using_path(cultivar_template)
        for pathtiprocess in self._process_paths:
            crop.write(pathtiprocess)
    
    def run_using_r(self):
        
        for pathiprocess in self._process_paths:
            soil = glob.glob(pathiprocess+'/*.SOL*')
            
            dirname = os.path.dirname(soil[0])
            os.rename(soil[0], os.path.join(dirname, 'TR.SOL'))
    
            config_path = os.path.join(pathiprocess, 'experimental_file_config.yaml')
            returned_value = subprocess.call(['RScript', './r_scripts/r_run_dssat.R', f'{config_path}'] , shell= True)
    
        
        
    def __init__(self, path) -> None:
        self.path = path

    
    @staticmethod
    def open_file(path: str) -> List[str]:
        """
        Reads a text file and returns its lines as a list.

        Parameters
        ----------
        path : str
            Path to the DSSAT soil file.

        Returns
        -------
        List[str]
            A list of lines in the file.

        Raises
        ------
        AssertionError
            If the file does not exist at the specified path.
        """
        
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf-8") as fn:
            lines = fn.readlines()
            
        return lines
    
    def specific_paths(self):
        path = self._tmp_path if self._tmp_path.endswith('/') else self._tmp_path+'/'
        list_files = glob.glob(path+'**/*.{}*'.format('WTH'),recursive=True)
        #list_files = glob.glob(self._tmp_path+'/*.{}*'.format('SOL'))
        self._process_paths = [os.path.dirname(fn) for fn in list_files]
    
    def from_datacube_to_dssatfiles(self, xrdata, 
                                    data_source = 'climate', dim_name = 'date', 
                                    target_crs = 'EPSG:4326', group_by = None, 
                                    group_codes = None):
        
        dfdata = summarize_datacube_as_df(xrdata, dimension_name= dim_name, group_by = group_by, project_to= target_crs)
        
        if data_source == 'climate':
            from_weather_to_dssat(dfdata, date_name = dim_name, group_by = group_by,
                        outputpath=self._tmp_path, outputfn = 'WTHE0001', codes=group_codes)
        
        if data_source == 'soil':
            from_soil_to_dssat(dfdata, group_by=group_by, depth_name= dim_name,
                                    outputpath=self._tmp_path, outputfn='SOL', codes=group_codes, 
                                    country = self.country.upper(),site = self.site, soil_id='TRAN00001')
            
        return dfdata

        #subprocess.call(['RScript', 'r_create_experimental_files.R', f"'{fn}'"], shell= True)

        #returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
        #print('returned value:', returned_value)
        
        #excinfo.stdout = re.sub("\n{2,}", "\n", excinfo.stdout)
        #excinfo.stdout = re.sub("\n$", "", excinfo.stdout)
        
        #assert excinfo.returncode == 0, 'DSSAT execution Failed, check '\
        #    + f'{os.path.join(self._RUN_PATH, "ERROR.OUT")} file for a'\
        #    + ' detailed report'

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

                