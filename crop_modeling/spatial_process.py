from .utils.u_soil import get_layer_texture
from spatialdata.datacube import create_dimension
from .utils.process import get_crs_fromxarray
from spatialdata.gis_functions import masking_rescaling_xrdata
from tqdm import tqdm

def reproject_xarray(xrdata, target_crs, src_crs = None):
    if src_crs is None:
        try:
            src_crs = xrdata.rio.crs
        except:
            src_crs = xrdata.attrs.get('crs', None)
            
    assert src_crs is not None, "Please provide the source crs"
                        
    return xrdata.rio.write_crs(src_crs).rio.reproject(target_crs)

import xarray

def get_roi_data(roi, weather_datacube_s, soil_datacube_dict, aggregate_by = None, min_area = 15, scale_factor = 10):
    """_summary_

    Parameters
    ----------
    min_area : int
        the minimun feature geometry area to apply buffer in km2

    Returns
    -------
    (xarray.Dataset, xarray.Dataset)
        a tuple with the Multi-dimension climate and soil data for the given region of interest.
    """
    area = roi.area.values[0]/ (1000*1000)
    if area < min_area:
        narea = (min_area*1.2) - area
        buffer = (narea*100)
    else:
        buffer = None 

    weather_datacube_m = masking_rescaling_xrdata(weather_datacube_s, roi, buffer=buffer, scale_factor=scale_factor, return_original_size=True, method = 'nearest')
    xr_reference = weather_datacube_m.isel(date = 0)
    soil_datacube_m = {k: masking_rescaling_xrdata(v, roi, buffer=buffer, resample_ref =xr_reference)  for k,v in tqdm(soil_datacube_dict.items())}
    weather_datacube_m.attrs['crs'] = get_crs_fromxarray(weather_datacube_s)
    
    if aggregate_by == 'texture':
        soilref = get_layer_texture(soil_datacube_m[list(soil_datacube_m.keys())[0]])
        weatherdatavars = list(weather_datacube_m.data_vars.keys())
        weather_datacube_m = xarray.merge([weather_datacube_m,soilref])[weatherdatavars+ ['texture']]
        #
        soil_datacube_m = create_dimension(soil_datacube_m, newdim_name = 'depth', isdate = False)
        soildatavars = list(soil_datacube_m.data_vars.keys())
        soil_datacube_m = xarray.merge([soil_datacube_m,soilref['texture']])[soildatavars+ ['texture']]
    else:
        soil_datacube_m = create_dimension(soil_datacube_m, newdim_name = 'depth', isdate = False)

    soil_datacube_m.attrs['crs'] = get_crs_fromxarray(weather_datacube_m)

    return weather_datacube_m, soil_datacube_m

