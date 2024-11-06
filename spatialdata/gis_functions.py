from typing import List
from rasterio.warp import reproject
from rasterio.enums import Resampling
import shapely
from shapely.geometry import Polygon
import xarray
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from rasterio.transform import Affine
from rasterio import windows
import rasterio
from shapely.geometry import mapping
import geopandas as gpd
import os
import math
from rasterio.transform import from_bounds

import xarray
import numpy as np

def add_2dlayer_toxarrayr(imageasarray, xarraydata: xarray.Dataset, variable_name: str) -> xarray.Dataset:
    """
    Add a 2D layer to an existing xarray dataset.

    Parameters:
    -----------
    image_as_array : np.ndarray, optional
        Image data as a numpy array. Either `fn` or `image_as_array` must be provided.

    xarraydata : xarray.Dataset
        Existing xarray dataset.
    variable_name : str
        Name of the variable to be added.

    Returns:
    --------
    xarray.Dataset
        Updated xarray dataset with the added 2D layer.
    """
    #dimsnames = list(xarraydata.sizes.keys())
    #sizexarray = [dict(xarraydata.sizes)[i] for i in dict(xarraydata.sizes)]
    refdimnames = xarraydata.sizes

    
    assert len(imageasarray.shape) < 3
        
    xrimg = xarray.DataArray(imageasarray)    
        #y_index =[i for i in range(len(sizexarray)) if xrimg.shape[1] == sizexarray[i]][0]
        #x_index = 0 if y_index == 1 else 1
    newdims = {}
    for keyval in xrimg.sizes:
        xrimg.sizes[keyval]
        posdims = [j for j,keyvalref in enumerate(
            refdimnames.keys()) if xrimg.sizes[keyval] == refdimnames[keyvalref]]
        newdims[keyval] = posdims
    # check double same axis sizes
    if len(newdims[list(newdims.keys())[1]]) >1:
        newdims[list(newdims.keys())[1]] = list(refdimnames.keys())[1]
        newdims[list(newdims.keys())[0]] = list(refdimnames.keys())[0]
    else:
        newdims[list(newdims.keys())[1]] = list(refdimnames.keys())[newdims[list(newdims.keys())[1]][0]]
        newdims[list(newdims.keys())[0]] = list(refdimnames.keys())[newdims[list(newdims.keys())[0]][0]]

    xrimg.name = variable_name
    xrimg = xrimg.rename(newdims)
    
    return xarray.merge([xarraydata, xrimg])

class SpatialBoundaries:
    def __init__(self, sp_vector) -> None:
        self.spvector = sp_vector
    
    def vector_geometry(self) -> List:
        if isinstance(self.spvector, shapely.geometry.multipolygon.MultiPolygon):
            geom = self.spvector.geoms[0]
        else:
            geom = self.spvector

        return geom

    @property
    def extent(self):
        l, b, r, t = from_polygon_2bbox(self.vector_geometry)
        return l, b, r, t

def get_transform_fromxy(x: np.ndarray, 
                     y: np.ndarray):
    height = len(y)
    width = len(y)

    transform = from_bounds(np.sort(x)[0], np.sort(y)[0], np.sort(x)[-1], np.sort(y)[-1], width, height)
    
    return transform

def get_new_coords_for_newshape(oldx, oldy, newheight,newidth):
    """
    Generate new x, y coordinates and an affine transform for a raster with a new shape.

    Parameters
    ----------
    oldx : np.ndarray
        1D array of old x coordinates (pixel centers).
    oldy : np.ndarray
        1D array of old y coordinates (pixel centers).
    newheight : int
        New number of rows (height) for the resampled raster.
    newwidth : int
        New number of columns (width) for the resampled raster.

    Returns
    -------
    Tuple[(np.ndarray, np.ndarray), Affine]
        A tuple containing the new x, y coordinates and the affine transform.
    """

    sprx = abs(oldx[0]-oldx[1])
    spry = abs(oldy[0]-oldy[1])

    xmin = oldx[0] - sprx/2 if oldx[0]<oldx[1] else oldx[-1]-sprx/2
    ymin = oldy[0] - spry/2 if oldy[0]<oldy[1] else oldy[-1]-spry/2

    xmax = oldx[-1] + sprx/2 if oldx[0]<oldx[1] else oldx[0]+sprx/2
    ymax = oldy[-1] + spry/2 if oldy[0]<oldy[1] else oldy[0]+spry/2

    newspx = (xmax-xmin)/newidth
    newspy = (ymax-ymin)/newheight
    
    newx = np.linspace(xmin+(newspx/2),xmax-(newspx/2), newidth)
    newy = np.linspace(ymin+(newspy/2),ymax-(newspy/2), newheight)
    # Ensure correct ordering of coordinates
    newx = newx if oldx[0] < oldx[1] else newx[::-1]
    newy = newy if oldy[0] < oldy[1] else newy[::-1]
    
    new_transform = get_transform_fromxy(newx, newy)

    return [(newx, newy), new_transform]

def reproject_xrdata(xrsource, target_crs, xdim_name = 'x', ydim_name = 'y'):
    """
    Reproject xarray data to a new coordinate reference system (CRS).

    Parameters
    ----------
    xrsource : xarray.Dataset
        The xarray dataset to reproject.
    target_crs : str
        The target CRS (e.g., 'EPSG:4326').
    xdim_name : str, optional
        Name of the x dimension in the dataset. Defaults to 'x'.
    ydim_name : str, optional
        Name of the y dimension in the dataset. Defaults to 'y'.

    Returns
    -------
    xarray.Dataset
        The reprojected dataset.
    """
    variables = list(xrsource.data_vars.keys())
    assert len(xrsource.sizes.keys())<3, "not supported 3 dimensions yet" 
    print(xrsource.rio.crs)
    print(xrsource)
    #
    tr = xrsource.rio.transform() if xrsource.rio.transform() else transform_fromxy(
        x =xrsource[xdim_name].values, y = xrsource[ydim_name].values)[0]

    list_tif = []

    for i, var in enumerate(variables):
        source_image = xrsource[var].values

        nodata = np.nan if np.any(np.isnan(source_image)) else 0

        destination = np.zeros_like(source_image)

        img, target_transform = reproject(
            xrsource[var].values,
            destination,

            src_transform=tr,
            src_crs=xrsource.rio.crs,
            dst_crs=target_crs,
            src_nodata = nodata,
            resampling=Resampling.bilinear,

        )
        list_tif.append(np.squeeze(img))

    return list_tif_2xarray(list_tif, target_transform,crs= target_crs, nodata=nodata,bands_names=variables,dimsformat='CHW')



def get_boundaries_from_path(path, crs = None, round_numbers = False):
    assert os.path.exists(path)

    features = gpd.read_file(path)

    if crs:
        features = features.to_crs(crs)
    
    x1, y1, x2, y2 = features.total_bounds
    if round_numbers:
        x1 = int(math.floor(x1))
        y1 = int(math.floor(y1))
        x2 = int(math.ceil(x2))
        y2 = int(math.ceil(y2))

    return x1, y1, x2, y2


def get_windows_from_polygon(ds, polygon = None, xyxy = None):
    """
    Generates a window or list of windows from the given dataset and a polygon by computing the bounding box 
    of the polygon and using it to create corresponding raster windows.

    Parameters
    ----------
    ds : dict
        A dictionary containing raster metadata, which must include a 'transform' key.
    polygon : Any
        A geometry object representing a polygon, from which the bounding box is derived.

    Returns
    -------
    List[rasterio.windows.Window]
        A list of rasterio window objects that define the portion of the dataset covered by the polygon.

    Notes
    -----
    The function assumes that the polygon coordinates and the dataset's coordinate system are compatible.
    """
    bbox = from_polygon_2bbox(polygon) if  xyxy is None else xyxy
    if 'transform' in ds:
        trfun = Affine(*ds['transform'][:6]) if isinstance(ds['transform'], np.ndarray) else ds['transform']
    else:
        raise ValueError(' transform is neccesary')
    #width, height = (bbox[0] - bbox[2]), (bbox[1] - bbox[3])
    #xy_fromtransform(ds['transform'])
    b,t = (bbox[1], bbox[3]) if bbox[1] <= bbox[3] else (bbox[3], bbox[1])
    l,r = (bbox[0], bbox[2]) if bbox[0] <= bbox[2] else (bbox[2], bbox[0])
    if trfun[4] > 0:
        b,t = t,b
    if trfun[0] < 0:
        l,r = r, l

    window_ref = windows.from_bounds(left=l,bottom=b,right=r, top=t,transform=trfun)
    transform = windows.transform(window_ref, trfun)

    return [window_ref, transform]


def crop_using_windowslice(xr_data: xarray.Dataset, 
                           window: windows.Window, transform: Affine) -> xarray.Dataset:
    """
    Crop an xarray dataset using a window.

    Parameters:
    -----------
    xr_data : xr.Dataset
        The xarray dataset to be cropped.
    window : Window
        The window object defining the cropping area.
    transform : Affine
        Affine transformation defining the spatial characteristics of the cropped area.

    Returns:
    --------
    xr.Dataset
        Cropped xarray dataset.
    """
    
    # Extract the data using the window slices
    
    xrwindowsel = xr_data.isel(y=window.toslices()[0],
                                     x=window.toslices()[1]).copy()

    xrwindowsel.attrs['width'] = xrwindowsel.sizes['x']
    xrwindowsel.attrs['height'] = xrwindowsel.sizes['y']
    xrwindowsel.attrs['transform'] = transform

    return xrwindowsel

def clip_xarraydata(xarraydata:xarray.Dataset, polygon: Polygon = None, xyxy: List[float] = None, xdim_name = 'x', ydim_name = 'y'):
    """
    Clip an xarray dataset based on a GeoPandas DataFrame.

    Parameters:
    -----------
    xarraydata : xarray.Dataset
        Xarray dataset to be clipped.
    gpdata : geopandas.GeoDataFrame
        GeoPandas DataFrame used for clipping.
    buffer : float, optional
        Buffer distance for clipping geometry. Defaults to None.

    Returns:
    --------
    xarray.Dataset
        Clipped xarray dataset.
    """
    xrmetadata = xarraydata.attrs.copy()
    if xyxy is not None:
        crs = 'EPSG:4326' if xarraydata.rio.crs is None else xarraydata.rio.crs
        prmasked = xarraydata.rio.write_crs(crs)
        x1, y1, x2, y2 = xyxy
        return prmasked.rio.clip_box(minx=x1, miny=y1, maxx=x2, maxy=y2)
    
    if 'transform' not in xrmetadata:
        #xrmetadata['transform'] = transform_fromxy(x = xarraydata[xdim_name].values, y=xarraydata[ydim_name].values)[0]
        xrmetadata['transform'] = xarraydata.rio.transform()
    window, dsttransform = get_windows_from_polygon(xrmetadata,
                                                    polygon= polygon, xyxy=xyxy)
    
    clippedmerged = crop_using_windowslice(xarraydata, 
                                        window, dsttransform)

    return clippedmerged

def from_polygon_2bbox(pol: Polygon, factor: float = None) -> List[float]:
    """
    Get the minimum and maximum coordinate values from a Polygon geometry.

    Parameters
    ----------
    pol : Polygon
        The polygon geometry.

    Returns
    -------
    list of float
        A list containing the bounding box coordinates in the format [xmin, ymin, xmax, ymax].
    """
    points = list(pol.exterior.coords)
    x_coordinates, y_coordinates = zip(*points)
    l = min(x_coordinates)
    b = min(y_coordinates)
    r = max(x_coordinates)
    t = max(y_coordinates)
    if factor:
        l = l - factor if l>0 else l + factor
        b -=  factor
        r =  r + factor if r>0 else r - factor
        t +=  factor

    return [l, b, r, t]



def from_xyxy_2polygon(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    """
    Create a polygon from the coordinates of two opposite corners of a bounding box.

    Parameters:
    -----------
    x1 : float
        x-coordinate of the first corner.
    y1 : float
        y-coordinate of the first corner.
    x2 : float
        x-coordinate of the second corner.
    y2 : float
        y-coordinate of the second corner.

    Returns:
    --------
    shapely.geometry.Polygon
        Polygon geometry created from the bounding box coordinates.
    """
    
    xpol = [x1, x2,
            x2, x1,
            x1]
    ypol = [y1, y1,
            y2, y2,
            y1]

    return Polygon(list(zip(xpol, ypol)))


def coordinates_fromtransform(transform, imgsize):
    """Create a longitude, latitude meshgrid based on the spatial affine.

    Args:
        transform (Affine): Affine matrix transformation
        imgsize (list): height and width 

    Returns:
        _type_: coordinates list in columns and rows
    """
    # All rows and columns
    rows, cols = np.meshgrid(np.arange(imgsize[0]), np.arange(imgsize[1]))

    # Get affine transform for pixel centres
    T1 = transform * Affine.translation(0, 0)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: T1 *(c, r) 

    # All east and north (there is probably a faster way to do this)
    rows, cols = np.vectorize(rc2en,
                                       otypes=[np.float64,
                                               np.float64])(rows, cols)
    #
    return [cols, rows]


def list_tif_2xarray(listraster:List[np.ndarray], transform: Affine, 
                     crs: str, nodata: int=0, 
                     bands_names: List[str] = None,
                     dimsformat: str = 'CHW',
                     dimsvalues: Dict[str, np.ndarray] = None):
    
    """
    Convert a list of raster images to an xarray dataset.

    Parameters:
    -----------
    list_raster : List[np.ndarray]
        List of numpy arrays representing the raster images.
    transform : Affine
        Affine transformation information.
    crs : str
        Coordinate reference system.
    nodata : int, optional
        Value representing nodata. Defaults to 0.
    bands_names : List[str], optional
        List of band names. Defaults to None.
    dims_format : str, optional
        Format of dimensions. Defaults to 'CHW'.
    dims_values : Dict[str, np.ndarray], optional
        Values for the dimensions. Defaults to None.

    Returns:
    --------
    xr.Dataset
        Xarray dataset containing the raster images.
    """
    
    assert len(listraster)>0
    
    if len(listraster[0].shape) == 2:            
        if dimsformat == 'CHW':
            width = listraster[0].shape[1]
            height = listraster[0].shape[0]
            dims = ['y','x']
        if dimsformat == 'CWH':
            width = listraster[0].shape[0]
            height = listraster[0].shape[1]
            dims = ['y','x']
            
    if len(listraster[0].shape) == 3:
        
        ##TODO: allow multiple formats+
        if dimsformat == 'CDWH':
            width = listraster[0].shape[1]
            height = listraster[0].shape[2]
            dims = ['date','y','x']
            
        if dimsformat == 'CDHW':
            width = listraster[0].shape[2]
            height = listraster[0].shape[1]
            dims = ['date','y','x']
            
        if dimsformat == 'DCHW':
            width = listraster[0].shape[2]
            height = listraster[0].shape[1]
            dims = ['date','y','x']
            
        if dimsformat == 'CHWD':
            width = listraster[0].shape[1]
            height = listraster[0].shape[0]
            dims = ['y','x','date']

    dim_names = {'dim_{}'.format(i):dims[i] for i in range(
        len(listraster[0].shape))}
    
           
    metadata = {
        'transform': transform,
        'crs': crs,
        'width': width,
        'height': height,
        'count': len(listraster)
        
    }
    if bands_names is None:
        bands_names = ['band_{}'.format(i) for i in range(len(listraster))]

    riolist = []
    imgindex = 1
    for i in range(len(listraster)):
        img = listraster[i]
        xrimg = xarray.DataArray(img)
        xrimg.name = bands_names[i]
        riolist.append(xrimg)
        imgindex += 1

    # update nodata attribute
    metadata['nodata'] = nodata
    metadata['count'] = imgindex

    multi_xarray = xarray.merge(riolist)
    multi_xarray.attrs = metadata
    multi_xarray = multi_xarray.rename(dim_names)
    
    # assign coordinates
    if dimsvalues:
        y = dimsvalues['y']
        x = dimsvalues['x']
        multi_xarray = multi_xarray.assign_coords(dimsvalues)
    else:
        y,x = coordinates_fromtransform(transform,
                                     [height, width])
        multi_xarray = multi_xarray.assign_coords(x=np.sort(np.unique(x)))
        ys = np.sort(np.unique(y))[::-1] if list(transform)[4] < 0 else np.unique(y)
        multi_xarray = multi_xarray.assign_coords(y=ys)

    return multi_xarray


def mask_xarray_using_rio(xrdata, geometry, drop = True, all_touched = True, reproject_to_raster = True):
    import rioxarray as rio

    if reproject_to_raster:
        geometry = geometry.to_crs(xrdata.rio.crs)
    else:
        xrdata = xrdata.rio.reproject(geometry.crs)
    
    xrdata = xrdata.rio.write_crs(xrdata.rio.crs)

    x1, y1, x2, y2 = geometry.total_bounds
    sub = xrdata.rio.clip_box(minx=x1, miny=y1, maxx=x2, maxy=y2)
    clipped = sub.rio.clip(geometry.geometry.apply(mapping), geometry.crs, drop=drop, all_touched=all_touched)

    return clipped

def mask_xarray_using_gpdgeometry(xrdata, geometry, xdim_name = 'x', ydim_name = 'y', clip = True, all_touched = True):
    import rioxarray as rio
    #print(xrdata.rio.transform())
    try:
        src_transform = xrdata.rio.transform() 
    except:
        src_transform = xrdata.attrs['transform']
    ShapeMask = rasterio.features.geometry_mask(geometry,
                                    out_shape=(len(xrdata[ydim_name]), len(xrdata[xdim_name])),
                                    transform=src_transform,
                                    all_touched = all_touched,
                                    invert=True)

    ShapeMask = xarray.DataArray(ShapeMask , dims=("y", "x"))

    prmasked = xrdata.where(ShapeMask == True)
    if clip:
        prmasked = clip_xarraydata(prmasked,xyxy=geometry.total_bounds)

    return prmasked

def read_raster_data(path, crop_extent: List[float] = None, xdim_name = 'x', ydim_name = 'y'):
    assert os.path.exists(path), f"{path} does not exits" 
    
    xr_data = xarray.open_dataset(path)
    dimnames = list(xr_data.sizes.keys())
    
    if 'lon' in dimnames and xdim_name!='lon':
        xr_data = xr_data.rename({'lon':xdim_name,'lat':ydim_name})

    elif 'longitude' in dimnames and xdim_name!='longitude':
        xr_data = xr_data.rename({'longitude':xdim_name,'latitude':ydim_name})
    
    elif 'x' in dimnames and xdim_name!='x':
        xr_data = xr_data.rename({'x':xdim_name,'y':ydim_name})
    
    if crop_extent is not None:
        xr_data = clip_xarraydata(xr_data,xyxy=crop_extent)
    
    return xr_data


def re_scale_xarray(xrdata, scale_factor, xdim_name = 'x', ydim_name = 'y', method ='nearest' ):
    height, width = xrdata[list(xrdata.data_vars.keys())[0]].shape
    newheight = int(height*scale_factor) 
    newwidth = int(width*scale_factor)

    oldx = xrdata[xdim_name].values
    oldy = xrdata[ydim_name].values

    (newx, newy), new_transform = get_new_coords_for_newshape(oldx, oldy, newheight,newwidth)

    dst_re = xrdata.interp({'x': newx,'y': newy}, method = method)
    
    dst_re = dst_re.rio.write_transform(new_transform)
    dst_re.attrs['transform'] = new_transform
    dst_re.attrs['height'] = len(newy)
    dst_re.attrs['width'] = len(newx)

    return dst_re

def resample(xrdata, newx, newy, target_transform, method = 'nearest', xdim_names = 'x', ydim_name = 'y'):

    dst_re = xrdata.interp({xdim_names: newx,ydim_name: newy}, method = method)

    dst_re.attrs['transform'] = target_transform
    dst_re.attrs['height'] = len(newy)
    dst_re.attrs['width'] = len(newx)
    dst_re = dst_re.rio.write_transform(target_transform)

    return dst_re

def resample_xarray(xarraydata, xrreference, method='linear', xrefdim_name = 'x', yrefdim_name = 'y', target_crs = None):
    """
    Function to resize an xarray data and update its attributes based on another xarray reference 
    this script is based on the xarray's interp() function

    Parameters:
    -------
    xarraydata: xarray
        contains the xarray that will be resized
    xrreference: xarray
        contains the dims that will be used as reference
    method: str
        method that will be used for interpolation
        ({"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "linear")
    
    Returns:
    -------
    a xarray data with new dimensions
    """
    from rasterio.enums import Resampling


    xrref = xrreference.copy()

    if yrefdim_name in xarraydata.sizes.keys():
        xdim_name, ydim_name = 'x', 'y'
    elif 'lat' in xarraydata.sizes.keys():
        xdim_name, ydim_name = 'lon', 'lat'
    elif 'lonfitude' in xarraydata.sizes.keys():
        xdim_name, ydim_name = 'longitude', 'latitude'

    if target_crs is not None:
        if str(target_crs) != str(xarraydata.rio.crs) and xarraydata.rio.crs is not None: ### TODO: there must be a warning here
            xarraydata = reproject_xrdata(xarraydata, xrreference.rio.crs)

    xrresampled = xarraydata.interp({xdim_name: xrref[xrefdim_name].values,
                                    ydim_name: xrref[yrefdim_name].values},method = method
                                    )
    
    #xrresampled = xarraydata.rio.reproject(
    #    xrref.rio.crs,
    #    shape = (len(xrref[yrefdim_name].values), len(xrref[xrefdim_name].values)),
    #    Resampling = Resampling.bilinear)

    
    xrresampled.attrs['transform'] = get_transform_fromxy(xrref[xrefdim_name].values,
                                                          xrref[yrefdim_name].values)#transform_fromxy(xrref[xrefdim_name].values,xrref[yrefdim_name].values, xrref.attrs['transform'][0])[0]
    #xrresampled.attrs['transform'] = xrref.rio.transform()

    xrresampled.attrs['height'] = xrresampled[list(xrresampled.keys())[0]].shape[0]
    xrresampled.attrs['width'] = xrresampled[list(xrresampled.keys())[0]].shape[1]
    xrresampled.attrs['dtype'] = xrresampled[list(xrresampled.keys())[0]].data.dtype
    
    return xrresampled

def transform_fromxy(x: np.ndarray, 
                     y: np.ndarray, 
                     spr: Union[float, List[float]] = None) -> Affine:
    """
    Generates an affine transform from coordinate arrays x and y and the given spatial resolution.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the grid.
    y : np.ndarray
        The y-coordinates of the grid.
    spr : Union[float, List[float]]
        The spatial resolution. Can be a single float or a list of two floats.
        If a list, the first element is the x-resolution and the second is the y-resolution.

    Returns
    -------
    tuple
        A tuple containing the affine transformation and the shape of the resulting grid as a tuple (rows, columns).
    """
    
    if spr is None:
        sprx = abs(x[1]- x[0])
        spry = abs(y[1]- y[0])
        print(sprx, spry)
    elif type(spr) is not list:
        sprx = spr
        spry = spr
    else:
        sprx, spry = spr
    gridX, gridY = np.meshgrid(x, y)

    signy = -1. if y[-1]<y[0] else 1.
    signx = -1. if x[-1]<x[0] else 1.
    affinematrix = [Affine.translation(
        gridX[0][0] - sprx / 2, gridY[0][0] - spry / 2) * Affine.scale(sprx*signx, spry*signy),
            gridX.shape]
    
    return affinematrix

def xy_fromtransform(transform, width, height):
    """
    this function is for create longitude and latitude range values from the 
    spatial transformation matrix

    Parameters:
    ----------
    transform: list
        spatial tranform matrix
    width: int
        width size
    height: int
        heigth size

    Returns:
    ----------
    two list, range of unique values for x and y
    """
    T0 = transform
    T1 = T0 * Affine.translation(0.5, 0.5)
    rc2xy = lambda r, c: T1 * (c, r)
    xvals = []
    for i in range(width):
        xvals.append(rc2xy(0, i)[0])

    yvals = []
    for i in range(height):
        yvals.append(rc2xy(i, 0)[1])

    xvals = np.array(xvals)
    if transform[0] < 0:
        xvals = np.sort(xvals, axis=0)[::-1]

    yvals = np.array(yvals)
    if transform[4] < 0:
        yvals = np.sort(yvals, axis=0)[::-1]

    return [xvals, yvals]


