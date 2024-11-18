import os
import numpy as np
import pandas as pd

from .files_reading import section_indices, delimitate_header_indices
from ..soil_funs import find_soil_textural_class, calculate_sks, slu1

from DSSATTools.soil import SoilProfile, SoilLayer
from typing import List
            
class DSSATSoil_base():
    """
    A utility class for handling DSSAT soil files.

    This class provides static methods to read soil files, extract soil IDs, 
    check and modify soil IDs, and parse soil properties into a DataFrame.
    """
    
    def __init__(self) -> None:
        pass
    
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
    
    @staticmethod
    def get_soil_id(path: str) -> str:
        """
        Extracts the soil ID from a DSSAT soil file.

        Parameters
        ----------
        path : str
            Path to the DSSAT soil file.

        Returns
        -------
        str
            The extracted soil ID.
        """
        lines = DSSATSoil_base.open_file(path)
        
        infoindices = list(section_indices(lines, pattern='*'))
        line = lines[infoindices[1]]
        return line.strip().split(':')[0][1:].split('  ')[0]
    
    @staticmethod
    def check_id(path: str) -> str:
        """
        Checks and modifies the soil ID in a DSSAT soil file if its length exceeds 8 characters.

        If the ID is too long, it replaces the ID with an string and updates the file.

        Parameters
        ----------
        path : str
            Path to the DSSAT soil file.

        Returns
        -------
        str
            The updated or existing soil ID.
        """
        
        soilid = DSSATSoil_base.get_soil_id(path)
        if len(soilid) > 8:
            lines = DSSATSoil_base.open_file(path)
            infoindices = list(section_indices(lines, pattern='*'))
            line = lines[infoindices[1]]
            lines[infoindices[1]] =line.replace(soilid,'1')
            with open(path, 'w') as file:
                file.writelines( lines )
            soilid = '1'
            
        return soilid
    
    @staticmethod
    def soil_properties_as_df(path: str) -> pd.DataFrame:
        """
        Reads soil properties from a DSSAT soil file and returns them as a DataFrame.

        The soil properties are typically located in the last section of the file.

        Parameters
        ----------
        path : str
            Path to the DSSAT soil file.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing soil properties.

        """
        lines = DSSATSoil_base.open_file(path)
        section_ids = list(section_indices(lines))
        lastsection = section_ids[-1]
        # the soil table is in the last section
        section_header_str = lines[lastsection]
        header_indices = delimitate_header_indices(section_header_str)
        
        data_rows = []
        for section_data_str in lines[(lastsection+1):len(lines)]:
            data_rows.append([section_data_str[i:j].strip()
                        for i, j in header_indices])
        
        sec_header = section_header_str.split()
        return pd.DataFrame(data=data_rows, columns=sec_header).drop('@', axis = 1)
    
    
    
    
class DSSATSoil_fromSOILGRIDS(SoilProfile):
    """
    DSSAT Soil profile class built from SoilGrids data.

    This class is used to construct soil profiles for DSSAT from SoilGrids data. 
    It provides methods for mapping soil data, calculating hydraulic conductivity, 
    and setting general soil and location parameters.
    
    Attributes
    ----------
    GENERAL_DESCRIPTION : dict
        Descriptions for each soil parameter.
    SOILGRID_MAPPING : dict
        Mapping of SoilGrid parameters to DSSAT parameters and conversion factors.
    ALBEDOS : tuple
        Albedo values for various texture classes.
    RUNOFF_CURVES : tuple
        Runoff curve values for various texture classes.
    TEXTURE_CLASSES : tuple
        Textural classes of the soil.
    SUM_TEXTURE_CLASSES : tuple
        DSSAT textural classes for summary purposes.
    """

    GENERAL_DESCRIPTION = {
        'SLLL': 'Lower limit of plant extractable soil water, cm3 cm-3',
        'SDUL': 'Drained upper limit, cm3 cm-3',
        'SSAT': 'Upper limit, saturated, cm3 cm-3',
        'SRGF': 'Root growth factor, soil only, 0.0 to 1.0',
        'SBDM': 'Bulk density, moist, g cm-3',
        'SLOC': 'Organic carbon, %',
        'SLCL': 'Clay (<0.002 mm), %',
        'SLSI': 'Silt (0.05 to 0.002 mm), %',
        'SLCF': 'Coarse fraction (>2 mm), %',
        'SLNI': 'Total nitrogen, %',
        'SLHW': 'Soil pH',
        'SMHB': 'pH in buffer determination method, code',
        'SCEC': 'Cation exchange capacity, cmol kg-1',
        'SADC': 'Anion adsorption coefficient, cm3 (H2O) g [soil]-1',
        'SSKS': 'Saturated hydraulic conductivity, macropore, cm h-1'
    }

    SOILGRID_MAPPING = {
        'clay': ['SLCL', 0.01],
        'silt': ['SLSI', 0.01],
        'nitrogen': ['SLNI', 0.01],
        'bdod': ['SBDM', 0.01],
        'cfvo': ['SLCF', 0.1],
        'phh2o': ['SLHW', 0.1],
        'soc': ['SLOC', 0.01],
        'cec': ['SCEC', 0.1],
        'wv1500': ['SLLL', 0.001],
        'wv0033': ['SDUL', 0.001],
        'wv0010': ['SSAT', 0.001]
    }

    ALBEDOS = (0.12, 0.12, 0.13, 0.13, 0.12, 0.13, 0.13, 0.14, 0.13, 0.13, 0.16, 0.19, 0.13)
    RUNOFF_CURVES = (73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 68.0, 73.0, 68.0, 68.0, 73.0)
    TEXTURE_CLASSES = ("clay", "silty clay", "sandy clay", "clay loam", "silty clay loam", "sandy clay loam", "loam", 
                    "silty loam", "sandy loam", "silt", "loamy sand", "sand", "unknown")
    
    SUM_TEXTURE_CLASSES = ("C", "SIC", "SC", "CL", "SICL", "SCL", "L", "SIL", "SL", "SI", "LS", "S", "unknown")

    @staticmethod
    def get_init_from_texture(sand: float, clay: float):
        """
        Find the soil texture class based on sand and clay content.

        Parameters
        ----------
        sand : float
            Sand content as a percentage.
        clay : float
            Clay content as a percentage.

        Returns
        -------
        str
            The soil texture class.
        """
        return find_soil_textural_class(sand, clay)
    
    def soilgrid_dict(self, dict_names  = None):
        """
        SoilGrid convention compared with DSSAT.
        
        Parameters
        ----------
        dict_names : dict, optional
            A dictionary mapping SoilGrid variables to DSSAT variables. If not provided, the default mapping is used.

        Returns
        -------
        dict
            The mapping dictionary used for converting SoilGrid data to DSSAT format.
        """

        if dict_names:
            self._dict_names = dict_names
        else:
            self._dict_names = {'clay': ['SLCL', 0.1],
         'silt': ['SLSI', 0.1],
         'nitrogen': ['SLNI', 0.01],
         'bdod': ['SBDM', 0.01],
         'cfvo': ['SLCF', 0.01],
         'phh2o': ['SLHW', 0.1],
         'soc': ['SLOC', 0.01],
         'cec': ['SCEC', 0.1],
         'wv1500': ['SLLL',0.001],
         'wv0033': ['SDUL',0.001],
         'wv0010': ['SSAT',0.001]}
        
        return self._dict_names

    def from_df_to_dssat_list(self, df: pd.DataFrame, depth_col_name: str = 'depth') -> list:
        """
        Convert soil data from a DataFrame to a DSSAT-compatible format.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing soil data.
        depth_col_name : str, optional
            The name of the column containing depth information. Default is 'depth'.

        Returns
        -------
        list
            A list of soil layers in DSSAT-compatible format.
        """
        colnames = df.columns
        soillayers = []
        for row in range(df.shape[0]):
            dict_dssat = {}
            for col in list(self._dict_names.keys()):
                kdssat, fac = self._dict_names[col]
                if col not in colnames:
                    dict_dssat[kdssat]  = None
                else:
                    #if np.max(df[col].values[row])>1/fac:
                    val = fac*df[col].values[row]
                    #else:
                    #    val = df[col].values[row]
                        
                    dict_dssat[kdssat]  = val

            dict_dssat['SSKS'] = calculate_sks(
                            silt = dict_dssat['SLSI'], 
                            clay = dict_dssat['SLCL'], 
                            bulk_density = dict_dssat.get('SBDM',None), 
                            field_capacity = dict_dssat.get('SDUL',None),
                            permanent_wp = dict_dssat.get('SLLL',None))

            depth = df[depth_col_name].values[row]
            depth = int(depth.split('-')[1]) if isinstance(depth, str) else depth

            soillayers.append((depth, dict_dssat))
        return soillayers
    
    def add_soil_layers_from_df(self, df: pd.DataFrame, depth_col_name: str = 'depth') -> list:
        """
        Add soil layers from a DataFrame and convert them to DSSAT format.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing soil data.
        depth_col_name : str, optional
            The column name for depth. Default is 'depth'.

        Returns
        -------
        list            List of soil layers added to the profile.
        """
        #soil.SoilLayer(20, {'SLCL': 50, 'SLSI': 45}),
        listvals = self.from_df_to_dssat_list(df, depth_col_name= depth_col_name)
        ## sort depths 
        sortedbydepth = np.argsort([depth for depth,val in listvals])
        listvals = [listvals[i] for i in sortedbydepth]
        soillayers = [SoilLayer(d, v) for d,v in listvals]
        for layer in soillayers: self.add_layer(layer)
        
        return listvals
            
    def soil_general_soilline_parameters(self):
        """
        Set general soil parameters based on soil texture.

        This method sets parameters like albedo, root growth factor, runoff curve, 
        and pH buffer method based on the soil's texture class.
        """
        self._texture = self.get_init_from_texture(self._sand, self._clay)

        self._pos_text = self.TEXTURE_CLASSES.index(self._texture)

        self.SALB = self.ALBEDOS[self._pos_text]

        self.SLU1 = slu1(self._clay, self._sand)

        #self.SLDR = 0.5
        self.SLRO = self.RUNOFF_CURVES[self._pos_text]
        self.SLNF = 1.
        self.SLPF =0.92
        self.SMHB,  self.SMPX,  self.SMKE = "IB001", "IB001", "IB001"

    def set_general_location_parameters(self, **kwargs):
        """
        Set general location parameters such as description, country, site, latitude, and longitude.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional location parameters including 'description', 'country', 'site', 'lat', and 'lon'.
        """
        self.description = kwargs.get('description',None)
        self.description = "ISRIC V2 {} {}".format(self.SUM_TEXTURE_CLASSES[self._pos_text],
                                                                self._texture) if self.description is None else self.description
        
        self.country = kwargs.get('country', 'COL')[:3].upper()
        self.site = kwargs.get('site', 'CAL')[:3].upper()
        self.lat = np.round(kwargs.get('lat', -99), 2)
        self.lon = np.round(kwargs.get('lon', -99), 2)

        self.id = kwargs.get('id', None)
        self.id = f'{self.country}-{self.site}'.replace(' ', '') if self.id is None else self.id


    def __init__(self, **kwargs) -> None:
        """
        Initialize the DSSAT soil profile from SoilGrids data.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional soil profile parameters like sand, clay, and silt content.
        """

        self._sand = kwargs.get('sand',None)
        self._clay = kwargs.get('clay',None)
        self._silt = kwargs.get('silt',None)

        super().__init__(pars = {})
        self.soil_general_soilline_parameters()
        self.soilgrid_dict()
        self.set_general_location_parameters(**kwargs)
