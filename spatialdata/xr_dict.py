import numpy as np

import xarray
from shapely.geometry import Polygon
from rasterio import windows
from .gis_functions import (list_tif_2xarray,
                            crop_using_windowslice)

from typing import  Optional, Dict, List
import pickle
import json
import os

def get_crs_fromxarray(xrdata):
    crs = None
    if 'crs' in xrdata.attrs.keys():
        print(xrdata.attrs['crs'])
        crs = xrdata.attrs['crs']
    else:
        try:
            crs = xrdata.rio.crs
            crs = xrdata[list(xrdata.data_vars.keys())[0]].rio.crs if crs is None else None

        except:
            crs = None
    if crs is not None:
        crs = crs.to_string() if not isinstance(crs, str) else crs
    return crs

def check_output_fn(func):
    """
    A decorator that checks the existence of a specified path, creates it if necessary, 
    and constructs the full file path with the correct suffix.

    Parameters:
    func (function): A function that requires path, filename (fn), and suffix as arguments.

    Returns:
    function: A wrapper function that adds path validation and adjustment to the original function.

    Raises:
    ValueError: If the specified path cannot be used or created.
    """
    
    def inner(file, path, fn = None, suffix = None):
        
        if fn is None:
            fn = os.path.basename(path)
            path = os.path.dirname(path)
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            raise ValueError(f"Unable to use or create the specified path: {path}. Error: {e}")
            
        if suffix:
            fn = os.path.join(path, fn if fn.endswith(suffix) else fn + suffix)
          
        return func(file, path=path, fn=fn)
    
    return inner

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def crop_xarray_using_mask(maskdata: np.ndarray,
                           xrdata: xarray.Dataset, 
                           min_threshold_mask : float = 0,
                           buffer: int = None) -> xarray.Dataset:
    """
    Crop an xarray dataset using a mask.

    Parameters:
    -----------
    mask_data : np.ndarray
        Array representing the mask.
    xr_data : xr.Dataset
        Xarray dataset to be cropped.
    min_threshold_mask : float, optional
        Minimum threshold value for the mask. Defaults to 0.
    buffer : int, optional
        Buffer value from the mask data to the image border in pixels. Deafault None.
    Returns:
    --------
    xr.Dataset
        Cropped xarray dataset.
    """
    boolmask = maskdata > min_threshold_mask 

    y1, y2 = np.where(boolmask)[0].min(), np.where(boolmask)[0].max()
    x1, x2 = np.where(boolmask)[1].min(), np.where(boolmask)[1].max()
    
    if 'width' in list(xrdata.attrs.keys()) and 'height' in list(xrdata.attrs.keys()):
        ncols_img, nrows_img = xrdata.attrs['width'], xrdata.attrs['height']
    else:
        nrows_img, ncols_img = xrdata[list(xrdata.keys())[0]].values.shape
    
    if buffer:
        y1 = 0 if (y1-buffer)< 0 else y1-buffer
        x1 = 0 if (x1-buffer)< 0 else x1-buffer
        x2 = ncols_img if (x2+buffer)> ncols_img else x2+buffer
        y2 = nrows_img if (y2+buffer)> nrows_img else y2+buffer
        
      
    big_window = windows.Window(col_off=0, row_off=0, 
                                width=ncols_img, height=nrows_img)
        
    crop_window = windows.Window(col_off=x1, row_off=y1, width=abs(x2 - x1),
                            height=abs(y2-y1)).intersection(big_window)
    
    assert 'transform' in list(xrdata.attrs.keys())
    transform = windows.transform(crop_window, xrdata.attrs['transform'])
    
    xrfiltered = crop_using_windowslice(xrdata.copy(), crop_window, transform)
    
    return xrfiltered
    
def from_dict_toxarray(dictdata, dimsformat = 'DCHW'):
    """
    Convert spatial data from a custom dictionary to an xarray dataset.

    Parameters:
    -----------
    dictdata : Dict[str, Any]
        Custom dictionary containing spatial data.
    dimsformat : str, optional
        Format of dimensions in the resulting xarray dataset. Either 'DCHW' or CHW. Defaults to 'DCHW'.

    Returns:
    --------
    xr.Dataset
        Xarray dataset containing the converted spatial data.
    """
    
    import affine
        
    trdata = dictdata['attributes']['transform']
    crsdata = dictdata['attributes']['crs']
    varnames = list(dictdata['variables'].keys())
    listnpdata = get_data_from_dict(dictdata)
    
    # Process transform data
    if type(trdata) is str:
        trdata = trdata.replace('|','')
        trdata = trdata.replace('\n ',',')
        trdata = trdata.replace(' ','')
        trdata = trdata.split(',')
        trdata = [float(i) for i in trdata]
        if trdata[0] == 0.0 or trdata[4] == 0.0:
            pxsize = abs(dictdata['dims']['y'][0] - dictdata['dims']['y'][1])
            trdata[0] = pxsize
            trdata[4] = pxsize
        
    trd = affine.Affine(*trdata)
    nodata  = dictdata['attributes'].get('nodata', 0)
    if(nodata!= 0): float(nodata)
    dtype = dictdata['attributes'].get('nodata', None)
    dtype = np.float32 if(dtype== 'float32') else float
    datar = list_tif_2xarray(listnpdata, trd,
                                crs=crsdata,
                                bands_names=varnames,
                                dimsformat = dimsformat,
                                dimsvalues = dictdata['dims'],
                                nodata = nodata,
                                dtype = dtype)
    
    if 'date' in list(dictdata['dims'].keys()):
        datar = datar.assign_coords(date=np.sort(
            np.unique(dictdata['dims']['date'])))

        
    return datar

def from_xarray_to_dict(xrdata: xarray.Dataset) -> dict:
    """
    Transform spatial xarray data to a custom dictionary.

    Parameters:
    -----------
    xrdata : xr.Dataset
        Input xarray dataset to be transformed.

    Returns:
    --------
    dict
        Custom dictionary containing variables, dimensions, and attributes of the input xarray dataset.
    """
    
    datadict = {
        'variables':{},
        'dims':{},
        'attributes': {}}

    variables = list(xrdata.keys())
    
    for feature in variables:
        datadict['variables'][feature] = xrdata[feature].values.astype(np.float32) if isinstance(xrdata[feature].values, float) else xrdata[feature].values

    for dim in xrdata.sizes.keys():
        if dim == 'date':
            datadict['dims'][dim] = np.unique(xrdata[dim])
        else:
            datadict['dims'][dim] = xrdata[dim].values
    
    
    
    for attr in xrdata.attrs.keys():
        if attr == 'transform':
            datadict['attributes'][attr] = list(xrdata.attrs[attr])
        elif attr == 'crs':
            crs = get_crs_fromxarray(xrdata)
            datadict['attributes'][attr] = crs
        else:
            datadict['attributes'][attr] = '{}'.format(xrdata.attrs[attr])
            
    datadict['attributes']['nodata'] = '{}'.format(np.nan_to_num(np.float32(np.inf)))
    datadict['attributes']['dtype'] = 'float32'
    
    return datadict


def get_data_from_dict(data: Dict[str, Dict[str, np.ndarray]], 
                       onlythesechannels: Optional[List[str]] = None) -> np.ndarray:
    """
    Extracts data for specified channels from a dictionary and converts it into a NumPy array.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        A dictionary where the 'variables' key contains another dictionary mapping channel names to their data.
    onlythesechannels : Optional[List[str]], optional
        A list specifying which channels' data to extract. If None, data for all channels is extracted, by default None.

    Returns
    -------
    np.ndarray
        An array containing the data for the specified channels. The array's shape is (N, ...) where N is the number of channels.

    Examples
    --------
    >>> data = {'variables': {'red': np.array([1, 2, 3]), 'green': np.array([4, 5, 6]), 'blue': np.array([7, 8, 9])}}
    >>> get_data_from_dict(data, onlythesechannels=['red', 'blue'])
    array([[1, 2, 3],
        [7, 8, 9]])
    """
        
    dataasarray = []
    channelsnames = list(data['variables'].keys())
    
    if onlythesechannels is not None:
        channelstouse = [i for i in onlythesechannels if i in channelsnames]
    else:
        channelstouse = channelsnames
    for chan in channelstouse:
        dataperchannel = data['variables'][chan] 
        dataasarray.append(dataperchannel)

    return np.array(dataasarray)
    
class CustomXarray(object):
    """A custom class for handling and exporting UAV data using xarray.

    This class allows for exporting UAV data into pickle and/or JSON files
    and includes functionalities for reading and converting xarray datasets.

    Attributes:
        xrdata (xarray.Dataset): Contains the xarray dataset.
        customdict (dict): Custom dictionary containing channel data, dimensional names, and spatial attributes.
    """
    
    def __init__(self, xarraydata: Optional[xarray.Dataset]= None, 
                 file: Optional[str] = None, 
                 customdict: Optional[bool] = False,
                 filesuffix: str = '.pickle',
                 dataformat: str = 'DCHW') -> None:
        """Initializes the CustomXarray class.

        Args:
            xarraydata (xarray.Dataset, optional):
                An xarray dataset to initialize the class.
            file (str, optional):
                Path to a pickle file containing xarray data.
            customdict (bool, optional):
                Indicates if the pickle file is a dictionary or an xarray dataset.
            filesuffix (str, optional):
                Suffix of the file to read. Defaults to '.pickle'.
            dataformat (str, optional):
                Format of the multi-dimensional data. Defaults to 'DCHW', 'CDHW', 'CHWD', 'CHW'.

        Raises:
            ValueError:
                If the provided data is not of type xarray.Dataset when 'xarraydata' is used.

        Examples:
            ### Initializing by loading data from a pickle file
            custom_xarray = CustomXarray(file='/path/to/data.pickle')
        """
        
        self.xrdata = None
        self._customdict = None
        self._arrayorder = dataformat
        
        if xarraydata:
            #assert type(xarraydata) is 
            if not isinstance(xarraydata, xarray.Dataset):
                raise ValueError("Provided 'xarraydata' must be an xarray.Dataset")
        
            self.xrdata = xarraydata
            
        elif file:
            data = self._read_data(path=os.path.dirname(file), 
                                   fn = os.path.basename(file),
                                   suffix=filesuffix)
              
            if customdict:
                self.xrdata = from_dict_toxarray(data, 
                                                 dimsformat = self._arrayorder)
                
            else:
                self.xrdata = data
            
    
    @check_output_fn
    def _export_aspickle(self, path, fn, suffix = '.pickle') -> None:
        """Private method to export data as a pickle file.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.
            suffix (str, optional): File suffix. Defaults to '.pickle'.

        Returns:
            None
        """

        with open(fn, "wb") as f:
            pickle.dump([self._filetoexport], f)
    
    @check_output_fn
    def _export_asjson(self, path, fn, suffix = '.json'):
        """Private method to export data as a JSON file.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.
            suffix (str, optional): File suffix. Defaults to '.json'.

        Returns:
            None
        """
        
        json_object = json.dumps(self._filetoexport, cls = NpEncoder, indent=4)
        with open(fn, "w") as outfile:
            outfile.write(json_object)
    
    @check_output_fn
    def _read_data(self, path, fn, suffix = '.pickle'):
        """Private method to read data from a file.

        Args:
            path (str): Path to the file.
            fn (str): Filename.
            suffix (str, optional): File suffix. Defaults to '.pickle'.

        Returns:
            Any: Data read from the file.
        """
        
        with open(fn,"rb") as f:
            data = pickle.load(f)
        if suffix == '.pickle':
            if type(data) is list:
                data = data[0]
        return data
      
    def export_as_dict(self, path: str, fn: str, asjson: bool = False,**kwargs):
        """Export data as a dictionary, either in pickle or JSON format.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.
            asjson (bool, optional): If True, export as JSON; otherwise, export as pickle.

        Returns:
            None
        """
        
        self._filetoexport = self.custom_dict
        if asjson:
            self._export_asjson(path, fn,suffix = '.json')
            
        else:
            self._export_aspickle(path, fn,suffix = '.pickle', **kwargs)

    def export_as_xarray(self, path: str, fn: str,**kwargs):
        """Export data as an xarray dataset in pickle format.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.

        Returns:
            None
        """
        
        self._filetoexport = self.xrdata
        self._export_aspickle(path, fn,**kwargs)
    
    @property
    def custom_dict(self) -> dict:
        """Get a custom dictionary representation of the xarray dataset.

        Returns:
            dict: Dictionary containing channel data in array format [variables], dimensional names [dims],
            and spatial attributes [attrs].
        """
        
        if self._customdict is None:
            return from_xarray_to_dict(self.xrdata)
        else:
            return self._customdict
    
    @staticmethod
    def to_array(customdict: Optional[dict]=None, onlythesechannels: Optional[List[str]] = None) -> np.ndarray:
        """Static method to convert a custom dictionary to a numpy array.

        Args:
            customdict (dict, optional): Custom dictionary containing the data.
            onlythesechannels (List[str], optional): List of channels to include in the array.

        Returns:
            np.ndarray: Array representation of the data.
        """
        data = get_data_from_dict(customdict, onlythesechannels)
        return data
        

