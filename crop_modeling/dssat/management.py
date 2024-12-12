from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
import subprocess
import yaml

from .soil import DSSATSoil_base
from .weather import DSSAT_Weather
from .files_reading import DSSATFileModifier, remove_header_lines, flat_list_of_list
from omegaconf import OmegaConf
from typing import Dict, Tuple

import os
import pandas as pd

import numpy as np

from DSSATTools.base.sections import SECTIONS_HEADER_FMT, SECTIONS_ROW_FMT

SECTIONS_ROW_FMT['field'][0] = '1X,I1,1X,A8,1X,A8,1X,A5,1X,I5,1X,A5,2(1X,I5),1X,A5,1X,A4,1X,I5,2X,A9,A36'

SECTION_NAMES = {
    'TREATMENTS': '@N',
    'CULTIVARS': '@C',
    'FIELDS': '@L',
    'INITIAL CONDITIONS': '@C',
    'PLANTING DETAILS': '@P',
    'HARVEST DETAILS': '@H',
    'SIMULATION CONTROLS': '@N',
    '@  AUTOMATIC MANAGEMENT': '@N'
}





class Management_FileModifier(DSSATFileModifier):
    
    def __init__(self, planting_date: str, harvesting_date: str, n_planting_windows: int = None, path: str = None,):
        """
                planting_date : str
            Planting date in 'YYYY-MM-DD' format.
        harvesting_date : str
            Harvesting date in 'YYYY-MM-DD' format.
        """
        self.n_windows = n_planting_windows if n_planting_windows else 1
        super().__init__(path, SECTION_NAMES)

        self.planting_date = datetime.strptime(planting_date,  '%Y-%m-%d')
        self.harvesting_date = datetime.strptime(harvesting_date,  '%Y-%m-%d')
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
        header_ffstyle = [SECTIONS_HEADER_FMT[secname] for secname in section_ffname
                        ] if isinstance(section_ffname, list) else SECTIONS_HEADER_FMT[section_ffname]
        row_ffstyle = [SECTIONS_ROW_FMT[secname] for secname in section_ffname
                        ] if isinstance(section_ffname, list) else SECTIONS_ROW_FMT[section_ffname]
        
        dflist = self.extract_section_asdf(self.file_path, section_name, sub_section_pattern)
    
        return dflist, header_ffstyle, row_ffstyle

    
    def treatment_modifier(self, **kwargs):
        ffkeys = "treatments"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*TREATMENTS', SECTION_NAMES['TREATMENTS'])
        
        dflist[0]['TNAME....................'] = kwargs.get('tname', 'initial_planting')

        newsection = []
        for i in range(self.n_windows):
            tmpdf = dflist[0].copy()
            tmpdf.loc[:,'IC'] = i+1
            tmpdf.loc[:,'MP'] = i+1
            tmpdf.loc[:,'MH'] = i+1
            tmpdf.loc[:,'SM'] = i+1
            tmpdf.loc[:,'@N'] = i+1
            if i>0:
                tmpdf.loc[:,'TNAME....................'] = f"planting+{i}week" if i==1 else f"planting+{i}weeks"
            if i>=9:
                rowff_style = '4(1X,I1),1X,A25,3(2X,I1),2(1X,I2),6(2X,I1),2(1X,I2)'
                rowff_style = '4(1X,I1),1X,A25,3(2X,I1),2(1X,I2),6(2X,I1),2(1X,I2)'
            dssatlines = self.write_df_asff(tmpdf, headerff_style, rowff_style)
            if i>=9: 
                dssatlines[-1] = str(i+1) + dssatlines[-1][2:]
                
            newsection.append(dssatlines)
            
        newlines = [self.lines[self._section_idx['TREATMENTS']]] + remove_header_lines(newsection)
        
        return newlines
    
    def cultivar_modifier(self, **kwargs):
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
    
    def planting_modifier(self, **kwargs):
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
    
    def harvesting_modifier(self, **kwargs):
        ffkeys = "harvest details"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*HARVEST DETAILS', SECTION_NAMES['HARVEST DETAILS'])
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
        ffkeys = "simulation controls"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'*SIMULATION CONTROLS', SECTION_NAMES['SIMULATION CONTROLS'])
        
        simul_name = kwargs.get('sname','Default')
        nyears = kwargs.get('nyears',1)
        
        newsection = []
        for i in range(self.n_windows):
            simulation_list = []
            orig_df = dflist.copy()
            for j in range(len(orig_df)): orig_df[j].loc[:,'@N'] = i+1
            orig_df[0].loc[:,'SDATE'] = ((self.starting_date + timedelta(days=(7*(i)))).strftime('%y%j'))
            orig_df[0].loc[:,'NYERS'] = nyears
            orig_df[0].loc[:,'SNAME....................'] = simul_name
            for j in range(len(headerff_style)):
                dssatlines = self.write_df_asff(dflist[j], headerff_style[j], rowff_style[j])
                if i>=9: dssatlines[-1] = str(i+1) + dssatlines[-1][2:]
                simulation_list.append(dssatlines)
            newsection.append(simulation_list)
        
        return newsection
    
    def automatic_management_modifier(self, **kwargs):
        ffkeys = "automatic management"
        dflist,headerff_style, rowff_style = self._extract_section_info_base(ffkeys,'@  AUTOMATIC MANAGEMENT', SECTION_NAMES['@  AUTOMATIC MANAGEMENT'])
        
        newsection = []
        for i in range(self.n_windows):
            manlist = []
            orig_df = dflist.copy()
            for j in range(len(orig_df)):
                orig_df[j].loc[:,'@N'] = i+1
            orig_df[0].loc[:,'PFRST'] = (self.starting_date.strftime('%y')) + '001'
            orig_df[0].loc[:,'PLAST'] = (self.starting_date.strftime('%y')) + '001'
            orig_df[-1].loc[:,'HFRST'] = kwargs.get('hrfst',-99)
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
        self.fertilizer = kwargs.get('fertilizer',False) # TODO: implement true scenario
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
    
    def create_file(self, filex_template: str, path: str, **kwargs) -> None:
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
        config_path = self._write_configuration_file(filex_template, **kwargs)
        self.file_path = filex_template
        if self.file_path:
            self.lines = self.open_file(self.file_path)
            
        config_management = OmegaConf.load(config_path)
        
        sdul = np.array(config_management.SOIL.SDUL)
        slll = np.array(config_management.SOIL.SLLL)
        slb = config_management.SOIL.SLB
        ind_soil_water = config_management.SOIL.index_soilwat
        #path = config.MANAGEMENT.template
        nyears = config_management.GENERAL.number_years 
        
        trlines = self.treatment_modifier()
        cullines = self.cultivar_modifier(crop = self.crop_code, variety_id = self.variety)
        fllines = self.fields_modifier(weather_stname = config_management.WEATHER.file_name,
                    soil_id = config_management.SOIL.ID_SOIL,
                    long = config_management.SOIL.long, lat = config_management.SOIL.lat)
        initlines = self.initial_conditions_modifier(sdul = sdul, slb = slb, slll = slll, ind_soil_water = ind_soil_water, 
                                                     crop = self.crop_code)
        pllines = self.planting_modifier()
        hrlines = self.harvesting_modifier()
        
        simul_listlines = self.simulation_control_modifier(sname = f'{self.crop}_{nyears}', nyears = nyears)
        autman_listlines = self.automatic_management_modifier()
        
        # both simulation and automatic management are combined for the output
        listsimulationout = []
        for i in range(len(simul_listlines)):
            listsimulationout.append(flat_list_of_list(simul_listlines[i]
                                                    ) + [self.lines[self._section_idx['@  AUTOMATIC MANAGEMENT']]
                                                        ] + flat_list_of_list(autman_listlines[i]))
        siumautolines = [self.lines[self._section_idx['SIMULATION CONTROLS']]] + flat_list_of_list(listsimulationout)
        
        
        allsections = [trlines, cullines, fllines, initlines, pllines, hrlines, siumautolines]
        fnman = os.path.join(os.path.dirname(config_path), f'{config_management.GENERAL.output_name}0001.{self.crop_code}X')
        
        with open(fnman, 'w') as fn:
            for line in self.lines[:self._section_idx['TREATMENTS']]:
                fn.write(line)
            fn.write('\n')
            for i, sect in enumerate(allsections):
                for line in sect:
                    fn.write(line)
                if i<len(allsections): fn.write('\n') #TODO check automatic management last
                
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
                'harvestDate':  self.harvesting_date.strftime('%Y-%m-%d'),
                'fertilizer': self.fertilizer
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
        config_info = self.management_configuration_file(filex_template, number_years, soilid, soildata, weather_fn)
        
        config_path = os.path.join(self.path, 'experimental_file_config.yaml')
        #fn = os.path.join(self.path, 'experimental_file_config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(config_info, file)
            
        print(f"Configuration file written: {config_path}")
        return config_path
        
    
    def create_file_using_rdssat(self, filex_template: str, path: str, **kwargs) -> None:
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
        config_path = self._write_configuration_file(filex_template, **kwargs)
        
        returned_value = subprocess.call(['RScript', './r_scripts/r_create_experimental_files.R', f'{config_path}'] , shell= True)
        return returned_value