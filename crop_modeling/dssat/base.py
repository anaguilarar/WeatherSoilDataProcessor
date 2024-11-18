
import os
from DSSATTools.base.sections import Section, clean_comments
from DSSATTools.crop import Crop,GENOTYPE_PATH


class DSSABase(object):
    
    def set_up(self):
        assert os.path.exists(self.path)
        self._tmp_path = os.path.join(self.path, 'tmp')
        if not os.path.exists(self._tmp_path): os.mkdir(self._tmp_path)
        
    def __init__(self, path) -> None:
        self.path = path
        self.set_up()
    
            


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
        