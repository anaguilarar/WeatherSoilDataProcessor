
import numpy as np
import rasterio
import geopandas as gpd


def summarize_datacube_as_df(xrdata, dimension_name = 'date', group_by = None, project_to = None):
    
    datavar_names = get_variables_to_summarize(xrdata, dimension_name)
    df = xrdata.to_dataframe().reset_index().dropna()
    
    if group_by:
        datavar_names.pop(group_by)
        ddf = df.groupby([group_by, dimension_name], dropna = True).agg(datavar_names).reset_index()
    else:
        ddf = df.copy()
        ddf['group'] = 0
        group_by= 'group'
        ddf = ddf.groupby([group_by,dimension_name], dropna = True).agg(datavar_names).reset_index()
    
    if project_to is not None:

        src_crs = get_crs_fromxarray(xrdata)

        ddf = gpd.GeoDataFrame(ddf, geometry=gpd.points_from_xy(ddf.x,ddf.y))
        ddf = ddf.set_crs(src_crs, allow_override=True)
        ddf = ddf.to_crs(project_to)
        ddf['x'] = np.nanmean(ddf.geometry.values.map(lambda x: float(x.coords.xy[0][0])))
        ddf['y'] = np.nanmean(ddf.geometry.values.map(lambda x: float(x.coords.xy[1][0])))
        
    return ddf


def get_variables_to_summarize(xrdata, indexname):
     ## add coords

    datavars = {i:'mean' for i in xrdata.data_vars.keys()}
    for i in xrdata.sizes.keys():
        if indexname == i:
            continue
        datavars.update({i:'mean'})
        datavars.update({i:'mean'})

    return datavars


def summarize_dataframe(date, xrdata, variables, groupby = None):
        ddf = xrdata.to_dataframe().reset_index().dropna()
        if groupby:
            ddf = ddf.groupby([groupby], dropna = True).agg(variables).reset_index()
        else:
            ddf['tmp'] = 0
            groupby= 'tmp'
            ddf = ddf.dropna().reset_index()
        ddf['date'] = date

        return ddf

def check_percentage(value):
    return value *0.1 if np.max(value)>100 else value


def set_encoding(xrdata, compress_method = 'zlib'):
    return {k: {compress_method: True} for k in list(xrdata.data_vars.keys())}

def check_crs_inxrdataset(xrdataset):
    if 'crs' in xrdataset.attrs.keys():
        crs = xrdataset.attrs['crs']
        if isinstance(crs, rasterio.crs.CRS):
            xrdataset.attrs['crs'] = crs.to_string()
    return xrdataset

def get_crs_fromxarray(xrdata):
    crs = None
    if 'crs' in xrdata.attrs.keys():
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