import numpy as np
from .datacube import DataCubeBase
import tqdm


from soilgrids import SoilGrids
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds

import os
import pandas as pd

def find_soil_textural_class(sand,clay):
    """
    Function that returns the USDA-NRCS soil textural class given 
    the percent sand and clay.
    
    Parameters:
    sand (float, integer): Sand content as a percentage  
    clay (float, integer): Clay content as a percentage
    
    Returns:
    string: One of the 12 soil textural classes
    
    Authors:
    Andres Patrignani
    
    Date created:
    12 Jan 2024
    
    Source:
    E. Benham and R.J. Ahrens, W.D. 2009. 
    Clarification of Soil Texture Class Boundaries. 
    Nettleton National Soil Survey Center, USDA-NRCS, Lincoln, Nebraska.
    adapted: https://soilwater.github.io/pynotes-agriscience/exercises/soil_textural_class.html
    """
    
    if not isinstance(sand, (int, float, np.int64)):
        raise TypeError(f"Input type {type(sand)} is not valid.")

    try:
        # Determine silt content
        silt = 100 - sand - clay
        
        if sand + clay > 100:
            raise Exception('Inputs add over 100%')
        elif sand < 0 or clay < 0:
            raise Exception('One or more inputs are negative')
            
    except ValueError as e:
        return f"Invalid input: {e}"
    
    # Classification rules
    if silt + 1.5*clay < 15:
        textural_class = 'sand'

    elif silt + 1.5*clay >= 15 and silt + 2*clay < 30:
        textural_class = 'loamy sand'

    elif (clay >= 7 and clay < 20 and sand > 52 and silt + 2*clay >= 30) or (clay < 7 and silt < 50 and silt + 2*clay >= 30):
        textural_class = 'sandy loam'

    elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
        textural_class = 'loam'

    elif (silt >= 50 and clay >= 12 and clay < 27) or (silt >= 50 and silt < 80 and clay < 12):
        textural_class = 'silt loam'

    elif silt >= 80 and clay < 12:
        textural_class = 'silt'

    elif clay >= 20 and clay < 35 and silt < 28 and sand > 45:
        textural_class = 'sandy clay loam'

    elif clay >= 27 and clay < 40 and sand > 20 and sand <= 45:
        textural_class = 'clay loam'

    elif clay >= 27 and clay < 40 and sand <= 20:
        textural_class = 'silty clay loam'

    elif clay >= 35 and sand > 45:
        textural_class = 'sandy clay'

    elif clay >= 40 and silt >= 40:
        textural_class = 'silty clay'

    elif clay >= 40 and sand <= 45 and silt < 40:
        textural_class = 'clay'

    else:
        textural_class = 'unknown' # in case we failed to catch any errors earlier

    return textural_class

TEXTURE_CLASSES = {
    0 : 'unknown',
    1 : 'sand',
    2 : 'loamy sand',
    3 : 'sandy loam',
    4 : 'loam',
    5 : 'silt loam',
    6 : 'silt',
    7 : 'sandy clay loam',
    8 : 'clay loam',
    9 : 'silty clay loam',
    10 : 'sandy clay',
    11 : 'silty clay',
    12 : 'clay'
}

def find_soil_textural_class_in_nparray(sand,clay):
    if not isinstance(sand, (np.ndarray)):
        raise TypeError(f"Input type {type(sand)} is not valid.")
    
    silt = 100 - sand - clay
    silt[silt==100] = 0

    cond1 = silt + 1.5*clay < 15
    cond2 = np.logical_and(silt + 1.5*clay >= 15, silt + 2*clay < 30)
    cond3 = np.logical_or(np.logical_and(np.logical_and(clay >= 7, clay < 20), np.logical_and(sand > 52, silt + 2*clay >= 30)),np.logical_and(clay < 7, silt < 50, silt + 2*clay >= 30))
    cond4 = np.logical_and(np.logical_and(clay >= 7,clay < 27,silt >= 28),np.logical_and(silt < 50, sand <= 52))
    cond5 = np.logical_or(np.logical_and(silt >= 50, clay >= 12, clay < 27), np.logical_and(silt >= 50, silt < 80, clay < 12))
    cond6 = np.logical_and(silt >= 80, clay < 12)
    cond7 = np.logical_and(np.logical_and(clay >= 20, clay < 35), np.logical_and(silt < 28, sand > 45))
    cond8 = np.logical_and(np.logical_and(clay >= 27, clay < 40), np.logical_and(sand > 20, sand <= 45))
    cond9 = np.logical_and(clay >= 27, clay < 40, sand <= 20)
    cond10 = np.logical_and(clay >= 35, sand > 45)
    cond11 = np.logical_and(clay >= 40, silt >= 40)
    cond12 = np.logical_and(clay >= 40, sand <= 45, silt < 40)

    texts = np.zeros(clay.shape, dtype=int)
    texts[clay == 0] = -1

    texts[np.logical_and(texts==0, cond1)] = 1
    texts[np.logical_and(texts==0, cond2)] = 2
    texts[np.logical_and(texts==0, cond3)] = 3
    texts[np.logical_and(texts==0, cond4)] = 4
    texts[np.logical_and(texts==0, cond5)] = 5
    texts[np.logical_and(texts==0, cond6)] = 6
    texts[np.logical_and(texts==0, cond7)] = 7
    texts[np.logical_and(texts==0, cond8)] = 8
    texts[np.logical_and(texts==0, cond9)] = 9
    texts[np.logical_and(texts==0, cond10)] = 10
    texts[np.logical_and(texts==0, cond11)] = 11
    texts[np.logical_and(texts==0, cond12)] = 12
    texts[texts == -1] = 0
    return texts



def download_using_rasterio(url, extent, output_path):
    x1, y1, x2, y2 = extent
    with rasterio.open(url) as src:
    
        kwds = src.profile
        tags = src.tags() # Add soilgrids tags with creation info.
        kwds['driver'] = 'GTiff'
        kwds['tiled'] = True
        tr = src.window_transform(from_bounds(x1, y1, x2, y2, src.transform))
        kwds['transform'] = tr
        kwds['compress'] = 'deflate' # lzw or deflate
        kwds['dtype'] = 'int16' # soilgrids datatype
        kwds['nodata'] = -32768 # default nodata

        kwds.update({
                        'height': abs(int((y1 - y2)/250)),
                        'width': abs(int((x1 - x2)/250)),
                        'transform': tr})
        
        with rasterio.open(output_path, 'w', **kwds) as dst:
            dst.update_tags(**tags)
            dst.write(src.read(window=from_bounds(x1, y1, x2, y2, src.transform)))
        
GOOGLESTORAGE = 'https://storage.googleapis.com/isric-share-soilgrids/pre-release/'
AGGREGATESTORAGE1000 = 'https://files.isric.org/soilgrids/latest/data_aggregated/1000m/'

class SoilGridDataDonwload():

    def _ckeck_source(self, var):
        if var in ["bdod","cfvo", "clay", "nitrogen","phh2o", "sand", "silt", "soc", "cec"]:
            return self._get_from_soilgrid_package 
        
        if var in ["wv0010", "wv0033", "wv1500"]:
            return self._get_from_soilgrid_1000aggregate 


    def __init__(self,soil_layers, depths, output_folder) -> None:
        self._soil_layers = soil_layers
        self._depths = depths
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    def download_soilgrid(self, boundaries):
        x1, y1, x2, y2 = boundaries
        
        for var in self._soil_layers:
            for depth in self._depths:
                fun = self._ckeck_source(var)
                fun(var, depth ,[x1, y1, x2, y2], self.output_folder)
                out_file = "{}_{}cm_mean.tif".format(var, depth)
                print(f"File created: {out_file} ")

    
    @staticmethod
    def _get_from_soilgrid_1000aggregate(var, depth, extent, output_folder):
        #wv0010/wv0010_0-5cm_mean_1000.tif
        file_name = "{}_{}cm_mean_1000.tif".format(var, depth)
        url = AGGREGATESTORAGE1000 + "{}/".format(var) + file_name
        print(url)
        output_path = os.path.join(output_folder, file_name)
        download_using_rasterio(url, extent, output_path)

    @staticmethod
    def _get_from_soilgrid_package(var, depth, extent, output_folder):
            
            soil_grids = SoilGrids()
            x1, y1, x2, y2 = extent
            output_file = "{}_{}cm_mean_30s.tif".format(var, depth)
            data = soil_grids.get_coverage_data(service_id=var, coverage_id='{}_{}cm_mean'.format(var, depth), 
                                    west=int(x1), south=int(y1), east=int(x2), north=int(y2),
                                    crs='urn:ogc:def:crs:EPSG::152160' ,
                                    output=os.path.join(output_folder, output_file))
            

class SoilDataCube(DataCubeBase):
    
    @staticmethod
    def mask_mldata(xr_dict,geometry, clip = True):
        xrdict_masked = {}
        for d, v in tqdm.tqdm(xr_dict.items()):
            xrdict_masked[d] = DataCubeBase.mask_using_geometry(v,geometry, clip = clip)
        
        return xrdict_masked
    
    @staticmethod
    def to_dataframe(xr_dict):
        data= []
        for d, v in tqdm.tqdm(xr_dict.items()):
            df = v.to_dataframe()
            data.append(df)
            
        return pd.concat(data)  
    
    @staticmethod
    def add_date_dim(xrdata, dim_value , dim_name ='depth', new_dimpos = 0):
        xrdate = xrdata.expand_dims(dim = {dim_name:1}, axis = new_dimpos)
        xrdate[dim_name] = [dim_value]
        return xrdate
    
    def get_depth_paths(self):
        query_paths = self.folder_manager.get_all_paths(by='depth')
        return query_paths

    def multi_depth_data(self, reference_variable = 'wv0033', verbose = False):
        self.xr_dict = {}
        for k,v in tqdm.tqdm(self._query_paths.items()):
            print(v)
            self.xr_dict[k] = self.stack_mlt_data(v, reference_variable= reference_variable, verbose = verbose)
        return self.xr_dict

    def __init__(self,folder_manager, extent=None) -> None:
        super().__init__(extent)
        self.folder_manager = folder_manager
        self._query_paths = self.get_depth_paths()