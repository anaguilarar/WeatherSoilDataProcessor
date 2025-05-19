import os
import glob
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta
from omegaconf import OmegaConf

from .soil import DSSATSoil_base
from .weather import DSSAT_Weather
from .files_reading import (DSSATFileModifier, remove_header_lines, 
                           flat_list_of_list)
from DSSATTools.base.sections import SECTIONS_HEADER_FMT, SECTIONS_ROW_FMT
from DSSATTools.crop import CROPS_MODULES

SECTIONS_ROW_FMT['field'][0] = '1X,I1,1X,A8,1X,A8,1X,A5,1X,I5,1X,A5,2(1X,I5),1X,A5,1X,A4,1X,I5,2X,A9,A36'

SECTION_NAMES = {
    'TREATMENTS': '@N',
    'CULTIVARS': '@C',
    'FIELDS': '@L',
    'INITIAL CONDITIONS': '@C',
    'PLANTING DETAILS': '@P',
    'FERTILIZERS (INORGANIC)': '@F',
    'HARVEST DETAILS': '@H',
    'SIMULATION CONTROLS': '@N',
    '@  AUTOMATIC MANAGEMENT': '@N'
}


class Management_FileModifier(DSSATFileModifier):
    """A class to modify DSSAT management files with various planting and treatment options.
    
    Parameters
    ----------
    planting_date : str
        Planting date in 'YYYY-MM-DD' format.
    harvesting_date : str
        Harvesting date in 'YYYY-MM-DD' format.
    n_planting_windows : int, optional
        Number of planting windows to consider (default is 1).
    path : str, optional
        Path to the DSSAT file to modify (default is None).
    """
    
    def __init__(self, planting_date: str, harvesting_date: str = None, n_planting_windows: int = None, path: str = None,):
        
        self.n_windows = n_planting_windows if n_planting_windows else 1
        super().__init__(path, SECTION_NAMES)

        self.planting_date = datetime.strptime(planting_date,  '%Y-%m-%d')
        self.harvesting_date = datetime.strptime(harvesting_date,  '%Y-%m-%d') if harvesting_date else None
        self.starting_date = self.planting_date - relativedelta(months=1)
        self._section_indices_list()
    
    def _section_indices_list(self):
        section_idx = {}
        for k in SECTION_NAMES.keys():
            if k.startswith('@'):
                section_idx[k] = next(iter(self.get_section_indices(self.lines, k)))
            else: 
                section_idx[k] = next(iter(self.get_section_indices(self.lines, f'*{k}')))
                
        self._section_idx = section_idx
    
    def _extract_section_info_base(self,section_ffname, section_name, sub_section_pattern):
        """Extract base information from a section for modification.
        
        Parameters
        ----------
        section_ffname : str or list of str
            Section name(s) in the fixed format.
        section_name : str
            Name of the section to extract.
        sub_section_pattern : str
            Pattern to identify subsections.
            
        Returns
        -------
        tuple
            Contains:
            - dflist: List of DataFrames with section data
            - headerff_style: List of header formats
            - rowff_style: List of row formats
        """
        
        header_ffstyle = [SECTIONS_HEADER_FMT[secname] for secname in section_ffname
                        ] if isinstance(section_ffname, list) else SECTIONS_HEADER_FMT[section_ffname]
        row_ffstyle = [SECTIONS_ROW_FMT[secname] for secname in section_ffname
                        ] if isinstance(section_ffname, list) else SECTIONS_ROW_FMT[section_ffname]
        
        dflist = self.extract_section_asdf(self.file_path, section_name, sub_section_pattern)
    
        return dflist, header_ffstyle, row_ffstyle

    
    def treatment_modifier(self, **kwargs):
        """Modify the treatments section of the DSSAT file.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - fertilizer: str, whether fertilizer is applied ('R' for yes)
            
        Returns
        -------
        list of str
            Modified lines for the treatments section.
        """
        ffkeys = "treatments"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*TREATMENTS', SECTION_NAMES['TREATMENTS'])
        thereisferti = kwargs.get('fertilizer', False) == 'R'
        thereisharvest = kwargs.get('harvest', True)
        
        newsection = []
        for i in range(self.n_windows):
            tmpdf = dflist[0].copy()
            tmpdf.loc[:,'IC'] = i+1
            tmpdf.loc[:,'MP'] = i+1
            tmpdf.loc[:,'MH'] = i+1 if thereisharvest else 0
            tmpdf.loc[:,'SM'] = i+1
            tmpdf.loc[:,'@N'] = i+1
            tmpdf.loc[:,'MF'] = i+1 if thereisferti else 0
                
            if i>0:
                tmpdf.loc[:,'TNAME....................'] = f"planting+{i}week" if i==1 else f"planting+{i}weeks"
            if i>=9:
                rowff_style = '4(1X,I1),1X,A25,3(2X,I1),2(1X,I2),6(2X,I1),2(1X,I2)'
                if thereisferti: 
                    rowff_style = '4(1X,I1),1X,A25,3(2X,I1),2(1X,I2),2X,I1,1X,I2,4(2X,I1),2(1X,I2)'

            dssatlines = self.write_df_asff(tmpdf, headerff_style, rowff_style)
            if i>=9: 
                dssatlines[-1] = str(i+1) + dssatlines[-1][2:]
                
            newsection.append(dssatlines)
            
        newlines = [self.lines[self._section_idx['TREATMENTS']]] + remove_header_lines(newsection)
        
        return newlines
    
    def cultivar_modifier(self, **kwargs):
        """Modify the cultivars section of the DSSAT file.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - crop: str, crop code
            - cname: str, cultivar name
            - variety_id: int, variety identifier
            
        Returns
        -------
        list of str
            Modified lines for the cultivars section.
        """
        dflist = self.extract_section_asdf(self.file_path, '*CULTIVARS', SECTION_NAMES['CULTIVARS'])
        crop = kwargs.get('crop',None)
        cname = kwargs.get('cname',None)
        variety_id = kwargs.get('variety_id',None)
        dflist[0]['CR'] = crop if crop else -99
        dflist[0]['INGENO'] = variety_id if variety_id else -99
        dflist[0]['CNAME'] = cname if cname else 'NONE'

        newsection = self._modify_lines(dflist, '*CULTIVARS', SECTION_NAMES['CULTIVARS'])
        newlines = [self.lines[self._section_idx['CULTIVARS']]] + newsection
        return newlines
        
    def fields_modifier(self, **kwargs):
        """Modify the fields section of the DSSAT file.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - weather_stname: str, weather station name
            - soil_id: str, soil identifier
            - elevation: float, elevation in meters
            - long: float, longitude
            - lat: float, latitude
            
        Returns
        -------
        list of str
            Modified lines for the fields section.
        """
        ffkeys = "field"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*FIELDS', SECTION_NAMES['FIELDS'])
        weather_station_name = kwargs.get('weather_stname',None) 
        soil_id = kwargs.get('soil_id',None)
        elevation = kwargs.get('elevation',None)
        long = kwargs.get('long',None)
        lat = kwargs.get('lat',None)
        
                
        dflist[0]['WSTA....'] = weather_station_name if weather_station_name else -99
        dflist[0]['ID_SOIL'] = soil_id if soil_id else -99
        dflist[1]['.....ELEV'] = elevation if elevation else 390
        dflist[1]['...........XCRD'] = long if long else 90.
        dflist[1]['...........YCRD'] = lat if lat else 5.
        newsection = [self.write_df_asff(dflist[j], headerff_style[j], rowff_style[j]) for j in range(len(headerff_style))]
        newlines= [self.lines[self._section_idx['FIELDS']]] + flat_list_of_list(newsection)
        
        return newlines
    
    def initial_conditions_modifier(self, **kwargs):
        """Modify the initial conditions section of the DSSAT file.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - slb: list, soil layer depths
            - slll: list, lower limit of soil water
            - sdul: list, drained upper limit
            - ind_soil_water: int, soil water indicator
            - crop: str, crop code
            
        Returns
        -------
        list of str
            Modified lines for the initial conditions section.
        """
        ffkeys = ["initial conditions","initial conditions_table"]
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*INITIAL CONDITIONS', SECTION_NAMES['INITIAL CONDITIONS'])
        
        slb = kwargs.get('slb',None)
        slll = kwargs.get('slll',None)
        sdul = kwargs.get('sdul',None)
        ind_soil_water = kwargs.get('ind_soil_water',1)
        crop = kwargs.get('crop',None)
        
        if slb is not None: 
            dflist[1] = dflist[1].iloc[:len(slb)]
            dflist[1].loc[:,'ICBL'] = slb
        
        if slll is not None and sdul is not None: dflist[1].loc[:,'SH2O'] = slll + ((sdul-slll) * ind_soil_water)
        if crop: dflist[0].loc[:,'PCR'] = crop
        
        newsection = []
        for i in range(self.n_windows):
            orig_df = dflist.copy()        
            orig_df[0].loc[:,'ICDAT'] = ((self.starting_date + timedelta(days=(7*(i)))).strftime('%y%j'))
            orig_df[0].loc[:,'@C'] = i+1
            orig_df[1].loc[:,'@C'] = i+1
            listesect = []
            for j in range(len(headerff_style)):
                dssatlines = self.write_df_asff(orig_df[j], headerff_style[j], rowff_style[j])
                if i>=9: 
                    for z in range(1,len(dssatlines)): dssatlines[z] = str(i+1) + dssatlines[z][2:]
                
                listesect.append(dssatlines)
            newsection.append(listesect)
        newlines = [self.lines[self._section_idx['INITIAL CONDITIONS']]] + flat_list_of_list(flat_list_of_list(newsection))
        return newlines
    
    def planting_modifier(self):
        """Modify the planting details section of the DSSAT file.
        
        Returns
        -------
        list of str
            Modified lines for the planting details section.
        """
        ffkeys = "planting details"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*PLANTING DETAILS', SECTION_NAMES['PLANTING DETAILS'])
        newsection = []
        for i in range(self.n_windows):
            orig_df = dflist[0].copy()
            pdate = (self.planting_date + timedelta(days=(7*(i))))
            orig_df.loc[:,'PDATE'] = pdate.strftime('%y%j')
            orig_df.loc[:,'@P'] = i+1
            orig_df.loc[:,'PLNAME'] = pdate.strftime('%d-%b')
            dssatlines = self.write_df_asff(orig_df, headerff_style, rowff_style)
            if i>=9: dssatlines[-1] = str(i+1) + dssatlines[-1][2:]
            newsection.append(dssatlines)
        newlines = [self.lines[self._section_idx['PLANTING DETAILS']]] +  remove_header_lines(newsection)
        return newlines
    
    def harvesting_modifier(self):
        """Modify the harvest details section of the DSSAT file.
        
        Returns
        -------
        list of str
            Modified lines for the harvest details section.
        """
        ffkeys = "harvest details"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*HARVEST DETAILS', SECTION_NAMES['HARVEST DETAILS'])
        if self.harvesting_date is None: return None
        newsection = []
        for i in range(self.n_windows):
            orig_df = dflist[0].copy()
            orig_df.loc[:,'HDATE'] = (self.harvesting_date + timedelta(days=(7*(i)))).strftime('%y%j')
            orig_df.loc[:,'@H'] = i+1
            dssatlines = self.write_df_asff(orig_df, headerff_style, rowff_style)
            if i>=9: dssatlines[-1] = str(i+1) + dssatlines[-1][2:]
            newsection.append(dssatlines)
        
        newlines =[self.lines[self._section_idx['HARVEST DETAILS']]] +  remove_header_lines(newsection)
        return newlines
        
    def simulation_control_modifier(self, **kwargs):
        """Modify the simulation controls section of the DSSAT file.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - sname: str, simulation name
            - nyears: int, number of years
            - fertilizer: str, fertilizer type
            
        Returns
        -------
        list of list of str
            Modified lines for the simulation controls section.
        """
        ffkeys = "simulation controls"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*SIMULATION CONTROLS', SECTION_NAMES['SIMULATION CONTROLS'])
        
        simul_name = kwargs.get('sname','Default')
        nyears = kwargs.get('nyears',1)
        ferttype = kwargs.get('fertilizer','N')
        hrtype = 'R' if kwargs.get('harvest',True) else 'M'
        smodel = -99#kwargs.get('simulation_cropmodel', -99)
        crop = kwargs.get('crop',None)
        newsection = []
        for i in range(self.n_windows):
            simulation_list = []
            orig_df = dflist.copy()
            for j in range(len(orig_df)): orig_df[j].loc[:,'@N'] = i+1
            orig_df[0].loc[:,'SDATE'] = ((self.starting_date + timedelta(days=(7*(i)))).strftime('%y%j'))
            #orig_df[0].loc[:,'RSEED'] = 2409
            orig_df[0].loc[:,'NYERS'] = nyears
            orig_df[0].loc[:,'SNAME....................'] = simul_name
            if smodel != -99:
                orig_df[0].loc[:,'SMODEL'] = -99
            orig_df[1].loc[:,'SYMBI'] = 'Y'
            if crop is not None and crop.lower() != 'cassava':
                #if ferttype == 'R': orig_df[1].loc[:,'NITRO'] = 'Y'
                orig_df[1].loc[:,'NITRO'] = 'Y'
            orig_df[2].loc[:,'PHOTO'] = 'C' 
            orig_df[2].loc[:,'NSWIT'] = '1'
            orig_df[2].loc[:,'MESOM'] = 'G'
            orig_df[2].loc[:,'MESEV'] = 'S'
            orig_df[3].loc[:,'RESID'] = 'N'
            orig_df[3].loc[:,'FERTI'] = ferttype
            orig_df[3].loc[:,'HARVS'] = hrtype
            for j in range(len(headerff_style)):
                dssatlines = self.write_df_asff(orig_df[j], headerff_style[j], rowff_style[j])
                if i>=9: dssatlines[-1] = str(i+1) + dssatlines[-1][2:]
                simulation_list.append(dssatlines)
            newsection.append(simulation_list)
        
        return newsection
    
    def fertilizer_pertreatment(self, **kwargs):
        """Modify fertilizer application per treatment.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - fertilizer_dates_after_planting: list, days after planting for applications
            - fertilizer_npk: list, NPK values for each application
            
        Returns
        -------
        list of list of str or None
            Modified fertilizer lines or None if no applications specified.
        """

        daysafterplanting = kwargs.get('fertilizer_dates_after_planting', None)
        npk_values = kwargs.get('fertilizer_npk', None)
        if daysafterplanting is None: return None
        assert len(daysafterplanting) == len(npk_values), "fertlizer and dates must have same length"
        ferti_lines = []
        for i in range(self.n_windows):
            pdate = (self.planting_date + timedelta(days=(7*(i))))
            trt_schedule = {'date':[],'npk':[]}
            for dap,npk in zip(daysafterplanting, npk_values):
                trt_schedule['date'].append(pdate + timedelta(days=dap))
                trt_schedule['npk'].append(npk)
            ferti_line = self.fertilizer_modifier(i+1, fertilizer_schedule = trt_schedule)
            if i > 0:
                ferti_lines.append(ferti_line[2:])
            else:
                ferti_lines.append(ferti_line)
        return ferti_lines
    
    def fertilizer_modifier(self, fid, fertilizer_schedule, fname = -99):
        """Modify fertilizer application per treatment.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - fertilizer_dates_after_planting: list, days after planting for applications
            - fertilizer_npk: list, NPK values for each application
            
        Returns
        -------
        list of list of str or None
            Modified fertilizer lines or None if no applications specified.
        """
        def table_mod(df, fid, fdate, nval, pval, kval, fname):
            df.loc[:, '@F'] = fid
            df.loc[:, 'FMCD'] = 'FE001'
            df.loc[:, 'FACD'] = 'AP001'
            df.loc[:, 'FDATE'] = '00001' if fdate is None else fdate.strftime('%y%j')
            df.loc[:, 'FAMN'] = 0 if nval == -99 else nval
            df.loc[:, 'FAMP'] = 0 if pval == -99 else pval
            df.loc[:, 'FAMK'] = 0 if kval == -99 else kval
            df.loc[:, 'FAMC'] = 0
            df.loc[:, 'FAMO'] = 0
            df.loc[:, 'FERNAME'] = fname
            return df
        
        ffkeys = "fertilizers_table"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys, '*FERTILIZERS (INORGANIC)', SECTION_NAMES['FERTILIZERS (INORGANIC)'])

        newsection = []
        orig_df = dflist[0].copy()
        if fertilizer_schedule is not None:
            n_applications = len(list(fertilizer_schedule.values())[0])
            
            for i in range(n_applications):
                fdate = fertilizer_schedule['date'][i]
                fdate = datetime.strptime(fdate, '%Y-%m-%d') if isinstance(fdate, str) else fdate
                npkvalue = fertilizer_schedule['npk'][i]
                orig_df = table_mod(orig_df, fid, fdate, *npkvalue, fname)
                #if check_fertilizer_date(fdate, dssatm.planting_date): continue
                dssatlines = self.write_df_asff(orig_df, headerff_style, rowff_style)
                if fid>9: dssatlines[-1] = str(fid) + dssatlines[-1][2:]
                newsection.append(dssatlines)
        else:
            orig_df = table_mod(orig_df, 1, None, *[0,0,0], fname)
            dssatlines = self.write_df_asff(orig_df, headerff_style, rowff_style)
            newsection.append(dssatlines)
        
        newlines =[self.lines[self._section_idx['FERTILIZERS (INORGANIC)']]] +  remove_header_lines(newsection)
        return newlines
    
    def automatic_management_modifier(self, **kwargs):
        """Modify the automatic management section of the DSSAT file.
        
        Parameters
        ----------
        **kwargs : dict
            Optional arguments including:
            - hrfst: int, harvest first parameter
            
        Returns
        -------
        list of list of str
            Modified lines for the automatic management section.
        """
        ffkeys = "automatic management"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'@  AUTOMATIC MANAGEMENT', SECTION_NAMES['@  AUTOMATIC MANAGEMENT'])
        
        newsection = []
        for i in range(self.n_windows):
            manlist = []
            orig_df = dflist.copy()
            for j in range(len(orig_df)):
                orig_df[j].loc[:,'@N'] = i+1
            orig_df[0].loc[:,'PFRST'] = ((self.planting_date + timedelta(days=(7*(i))) - timedelta(days=3)).strftime('%y%j'))
            orig_df[0].loc[:,'PLAST'] = ((self.planting_date + timedelta(days=(7*(i))) + timedelta(days=3)).strftime('%y%j'))
            orig_df[0].loc[:,'PSTMX'] = 40
            orig_df[0].loc[:,'PSTMN'] = 40
            orig_df[3].loc[:,'RIPCN'] = 100
            orig_df[3].loc[:,'RTIME'] = 1
            orig_df[3].loc[:,'RIDEP'] = 20
            orig_df[-1].loc[:,'HFRST'] = kwargs.get('hrfst',-99)
            orig_df[-1].loc[:,'HLAST'] = -99
            for j in range(len(headerff_style)):
                dssatlines = self.write_df_asff(dflist[j], headerff_style[j], rowff_style[j])
                if i>=9: dssatlines[-1] = str(i+1) + dssatlines[-1][2:]
                manlist.append(dssatlines)
            newsection.append(manlist)
            
        return newsection

class DSSATManagement_base(Management_FileModifier):
    """
    A class for managing DSSAT experiments and configuration files.

    Provides utilities for handling DSSAT crop, soil, and weather data, as well as
    generating configuration files for simulations.
    """
    
    @property
    def extensions(self) -> Dict[str, str]:
        """
        Returns file extensions for DSSAT data types.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping data types to file extensions.
        """
        
        return {
        'weather': 'WTH',
        'soil' : 'SOL',
        'experiment' : 'X'
        }
        
    @property
    def crop_codes(self) -> Dict[str, str]:
        """
        Returns crop codes for DSSAT simulations.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping crop names to their DSSAT crop codes.
        """
        
        return {
        "Maize": "MZ",
        'Millet': "ML",
        'Sugarbeet': "BS",
        'Rice': "RI",
        'Sorghum': "SG",
        'Sweetcorn': "SW",
        'Alfalfa': "AL",
        'Bermudagrass': "BM",
        'Soybean': "SB",
        'Canola': "CN",
        'Sunflower': "SU",
        'Potato': "PT",
        'Tomato': "TM",
        'Cabbage': "CB",
        'Sugarcane': "SC",
        "Wheat": "WH",
        "Bean": "BN",
        "Cassava": "CS"
        }
        
    def __init__(self, crop: str, variety: str, planting_date: str, harvesting_date: str, n_planting_windows: int = 1, path = None):
        """
        Initialize a DSSAT management object.

        Parameters
        ----------
        path : str
            file experiment template path
        crop : str
            Name of the crop (e.g., "Maize").
        variety : str
            Crop variety identifier.
        planting_date : str
            Planting date in 'YYYY-MM-DD' format.
        harvesting_date : str
            Harvesting date in 'YYYY-MM-DD' format.
            
        n_planting_windows: int
            Number of planting windows, the window size is one week
        """
        
        super().__init__(path = path, planting_date = planting_date, harvesting_date = harvesting_date, n_planting_windows = n_planting_windows)
    
        self.crop = crop
        self.variety = variety
        self.crop_code = self.crop_codes.get(crop, None)

    
    def total_climate_years(self) -> int:
        """
        Calculates the total number of years in the climate data.

        Returns
        -------
        int
            Total number of years covered by the climate data.
        """
        weather_dates = DSSAT_Weather.get_dates_from_file(self._weather[0])
        years = [datetime.strptime(d,  '%Y%j').year for d in weather_dates]
        return abs(years[-1] - self.planting_date.year)
    

    def check_files(self, template: str) -> None:
        """
        Validates the existence of required files based on the template.

        Parameters
        ----------
        template : str
            Path to the experiment template file.

        Raises
        ------
        AssertionError
            If required soil or weather files are missing.
        """
        assert template.endswith(self.extensions['experiment']),  f"Template must end with '{self.extensions['experiment']}'"
        self._soil = glob.glob(self.path+'/*{}*'.format(self.extensions['soil']))
        self._weather = glob.glob(self.path+'/*{}*'.format(self.extensions['weather']))
        assert len(self._weather)>0, "No weather files found."
        assert len(self._soil)>0, "No soil files found."
    
    def general_info(self, **kwargs) -> None:
        """
        Updates general experiment configuration with provided keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            General experiment configuration options, such as: output_name, roi_id, plantingWindow, and index_soilwat.
        """
        self.output_name = kwargs.get('output_name', 'EXPS')
        self.roi_id = kwargs.get('roi_id', 1)
        self.n_windows = kwargs.get('plantingWindow',1)
        self.fertilizer = kwargs.get('fertilizer_schedule',None)
        self.index_soilwat = kwargs.get('index_soilwat',1)
        self._long = kwargs.get('long',None)
        self._lat = kwargs.get('lat',None)
        
    def soil_data(self) -> Tuple[str, pd.DataFrame]:
        """
        Retrieves soil properties and ID.

        Returns
        -------
        Tuple[str, pd.DataFrame]
            Soil ID and a DataFrame of soil properties.
        """
        
        soildata = DSSATSoil_base.soil_properties_as_df(self._soil[0])
        soildata = soildata[['SDUL', 'SLLL','SLB']].values.T.astype(float)
        soilid =  DSSATSoil_base.check_id(self._soil[0])
        
        return (soilid, soildata)

    def check_weather_fn(self) -> str:
        """
        Ensures the weather file has a valid DSSAT filename.

        Returns
        -------
        str
            Updated weather file name.
        """
        
        weather_fn = os.path.basename(self._weather[0])
        if len(weather_fn)>15:
            dirname = os.path.dirname(self._weather[0])
            weather_fn = 'WHTE0001.WTH'
            os.rename(self._weather[0], os.path.join(dirname, weather_fn))

        return weather_fn
    
    def create_file(self, filex_template: str = None, path: str = 'tmp', verbose = True, **kwargs) -> None:
        """
        Creates an experimental file using R-DSSAT tools.
        To implement this function it is neccesary to previosly setup RScript in terminal
        besides the working path must contains the SOIL and WEATHER dssat files
        Parameters
        ----------
        filex_template : str
            Path to the experiment template file.
        path : str
            Path to the working directory.
        **kwargs : dict
            Additional configuration options.
        """
        self.path = path 
        if filex_template is not None:
            self.file_path = filex_template
            if self.file_path:
                self.lines = self.open_file(self.file_path)
                
        config_path = self._write_configuration_file(self.file_path, **kwargs)
        if verbose:
            print(f"Configuration file written: {config_path}")
            
        config_management = OmegaConf.load(config_path)
        sdul = np.array(config_management.SOIL.SDUL)
        slll = np.array(config_management.SOIL.SLLL)
        slb = config_management.SOIL.SLB
        ind_soil_water = config_management.SOIL.index_soilwat
        #path = config.MANAGEMENT.template
        nyears = config_management.GENERAL.number_years 
        
        treatments_list = []
        #Cultivar options
        cullines = self.cultivar_modifier(crop = self.crop_code, variety_id = self.variety)
        #Field data
        fllines = self.fields_modifier(weather_stname = config_management.WEATHER.file_name,
                    soil_id = config_management.SOIL.ID_SOIL,
                    long = config_management.SOIL.long, lat = config_management.SOIL.lat)
        #Initial conditions ## soil info
        initlines = self.initial_conditions_modifier(sdul = sdul, slb = slb, slll = slll, ind_soil_water = ind_soil_water, 
                                                    crop = self.crop_code)

        #planting dates
        pllines = self.planting_modifier()
        #harvesting dates
        hrlines = self.harvesting_modifier()
        # fertilizer
        fertilizer_listlines = self.fertilizer_pertreatment(**config_management.MANAGEMENT)
        ferti = 'N' if fertilizer_listlines is None else 'R'
        harvest_schedule = not hrlines is None 
        # treatment
        trlines = self.treatment_modifier(fertilizer = ferti, harvest = harvest_schedule)
        # Simulation control # Currently Irrigation is deactivated, Symbiosis is Yes
        simul_listlines = self.simulation_control_modifier(sname = f'{self.crop}_{nyears}', nyears = nyears, fertilizer = ferti, crop = self.crop, harvest = harvest_schedule,
                                                        simulation_cropmodel = config_management.MANAGEMENT.get('simulation_cropmodel', -99))
        # Simulation control # Currently Irrigation is deactivated, Symbiosis is Yes
        autman_listlines = self.automatic_management_modifier()
        
        
        # both simulation and automatic management are combined for the output
        listsimulationout = []
        for i in range(len(simul_listlines)):
            listsimulationout.append(flat_list_of_list(simul_listlines[i]
                                                    ) + [self.lines[self._section_idx['@  AUTOMATIC MANAGEMENT']]
                                                        ] + flat_list_of_list(autman_listlines[i]))
        siumautolines = [self.lines[self._section_idx['SIMULATION CONTROLS']]] + flat_list_of_list(listsimulationout)
        
        treatments_list.append(trlines)
        treatments_list.append(cullines)
        treatments_list.append(fllines)
        treatments_list.append(initlines)
        treatments_list.append(pllines)
        if fertilizer_listlines is not None:
            treatments_list.append(flat_list_of_list(fertilizer_listlines))
        if hrlines is not None:
            treatments_list.append(hrlines)
        treatments_list.append(siumautolines)

        fnman = os.path.join(os.path.dirname(config_path), f'{config_management.GENERAL.output_name}0001.{self.crop_code}X')
        
        with open(fnman, 'w') as fn:
            for line in self.lines[:self._section_idx['TREATMENTS']]:
                fn.write(line)
            fn.write('\n')
            for i, sect in enumerate(treatments_list):
                for line in sect:
                    fn.write(line)
                if i<len(treatments_list): fn.write('\n') #TODO check automatic management last
                
        return fnman
        
    def management_configuration_file(self, filex_template: str, number_years: int, soilid: str,
                             soildata: pd.DataFrame, weather_fn: str) -> Dict:
        
        """
        Generates the DSSAT R configuration file.

        Parameters
        ----------
        filex_template : str
            Path to the experiment template.
        number_years : int
            Number of years for simulation.
        soilid : str
            Soil ID.
        soildata : pd.DataFrame
            Soil properties.
        weather_fn : str
            Weather file name.

        Returns
        -------
        Dict
            Configuration data for the experiment.
        """
        
        return{
            'GENERAL': {
                'roi_id' : 1,
                'working_path': self.path,
                'number_years': 1 if number_years == 0 else number_years,
                'output_name': self.output_name
            },
            'MANAGEMENT': {
                'template_path':  filex_template,
                'crop_code': self.crop_code,
                'varietyid': self.variety,
                'plantingWindow': self.n_windows,
                'startingDate': self.starting_date.strftime('%Y-%m-%d'), ## one month before of actual date
                'plantingDate': self.planting_date.strftime('%Y-%m-%d'),
                'harvestDate':  self.harvesting_date.strftime('%Y-%m-%d') if self.harvesting_date else None,
                'fertilizer_dates_after_planting': np.array(self.fertilizer['days_after_planting']).tolist() if self.fertilizer['days_after_planting'] is not None else None,
                'fertilizer_npk': np.array(self.fertilizer['npk']).tolist() if self.fertilizer['npk'] is not None else None,
                'simulation_cropmodel': CROPS_MODULES[self.crop.title()]
            },
            'SOIL':{
                'source': 'ISRIC V2',
                'index_soilwat': self.index_soilwat,
                'ID_SOIL': soilid,
                'SDUL': soildata[0].tolist(),
                'SLLL': soildata[1].tolist(),
                'SLB': soildata[2].tolist(),
                'long': self._long,
                'lat': self._lat
            },
            'WEATHER':{
                'file_name': weather_fn[:-4] ## the .WTH extension is not saved
            }
            
        }
    
    def _check_fertlizers_length(self):
        'fertilizer must contain values for nitrogen phosphorus and potasium, thereby it must be a list of 3 length'
        if self.fertilizer is None:
            self.fertilizer = {'days_after_planting': None, 'npk': None}
        if self.fertilizer['npk'] is None: return None
        listfert = []
        for i in range(len(self.fertilizer['npk'])):
            if len(self.fertilizer['npk'][i]) < 3:
                tmplist = [0] * 3
                tmplist[:len(self.fertilizer['npk'][i])] = self.fertilizer['npk'][i]
            elif len(self.fertilizer['npk'][i]) > 3:
                tmplist = self.fertilizer['npk'][i][:3]
            else:
                tmplist = self.fertilizer['npk'][i]
            listfert.append(tuple(tmplist))
        self.fertilizer['npk'] = listfert
        
    def _write_configuration_file(self, filex_template, **kwargs):
        try:
            self.check_files(filex_template)
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)
            return None
        
        number_years = self.total_climate_years()
        soilid, soildata = self.soil_data()
        self.general_info(**kwargs)
        weather_fn = self.check_weather_fn()
        self._check_fertlizers_length()
        
        config_info = self.management_configuration_file(filex_template, number_years, soilid, soildata, weather_fn)
        
        config_path = os.path.join(self.path, 'experimental_file_config.yaml')

        with open(config_path, 'w') as file:
            yaml.dump(config_info, file)
        
        return config_path