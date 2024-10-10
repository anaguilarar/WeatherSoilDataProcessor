


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

def plot_multichanels(data: np.ndarray, 
                       num_rows: int = 2, 
                       num_columns: int = 2, 
                       figsize: Tuple[int, int] = (10, 10),
                       label_name: Optional[str] = None,
                       chanels_names: Optional[List[str]] = None, 
                       cmap: str = 'viridis', 
                       fontsize: int = 12, 
                       legfontsize: int = 15,
                       legtickssize: int = 15,
                       colorbar: bool = True, 
                       vmin: Optional[float] = None, 
                       vmax: Optional[float] = None,
                       newlegendticks: Optional[List[str]] = None,
                       fontname: str = "Arial",
                       invertaxis: bool = True,
                       colorbar_orientation = 'vertical') -> Tuple[plt.Figure, np.ndarray]:
    """
    Creates a figure showing one or multiple channels of data with extensive customization options.

    Parameters
    ----------
    data : np.ndarray
        Numpy array containing the data to be plotted.
    num_rows : int, optional
        Number of rows in the subplot grid, by default 2.
    num_columns : int, optional
        Number of columns in the subplot grid, by default 2.
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height), by default (10, 10).
    label_name : Optional[str], optional
        Label for the colorbar legend, by default None.
    channel_names : Optional[List[str]], optional
        Labels for each plot, by default None.
    cmap : str, optional
        Matplotlib colormap name, by default 'viridis'.
    fontsize : int, optional
        Font size for the main figure, by default 12.
    legfontsize : int, optional
        Font size for the legend title, by default 15.
    legtickssize : int, optional
        Font size for the legend ticks, by default 15.
    colorbar : bool, optional
        If True, includes a colorbar legend, by default True.
    vmin : Optional[float], optional
        Minimum data value for colormap scaling, by default None.
    vmax : Optional[float], optional
        Maximum data value for colormap scaling, by default None.
    newlegendticks : Optional[List[str]], optional
        Custom legend ticks, by default None.
    fontname : str, optional
        Font name for the plot text, by default "Arial".
    invertaxis : bool, optional
        If True, inverts the x-axis, by default True.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        The created figure and array of axes.
    """ 
                
    import matplotlib as mpl
    if chanels_names is None:
        chanels_names = list(range(data.shape[0]))

    def set_ax(ax, data, cmaptxt, vmin, vmax, title, invertaxis):
        ax.imshow(data, cmap=cmaptxt, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontdict=fontmainfigure)
        if invertaxis: ax.invert_xaxis()
        ax.set_axis_off()
        return ax
        
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_columns, figsize = figsize)
    
    count = 0
    vars = chanels_names
    cmaptxt = plt.get_cmap(cmap)
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
            
    fontmainfigure = {'family': fontname, 'color': 'black', 
                      'weight': 'normal', 'size': fontsize }

    fontlegtick = {'family': fontname, 'color': 'black', 
                   'weight': 'normal', 'size': legtickssize}
    
    fontlegtitle = {'family': fontname, 'color':  'black', 
                    'weight': 'normal', 'size': legfontsize}
    
    for j in range(num_rows):
        for i in range(num_columns):
            if count < len(vars):

                if num_rows>1 and num_columns > 1:
                    ax[j,i] = set_ax(ax[j,i], data[count], cmaptxt, vmin, vmax, vars[count], invertaxis)
                elif (num_rows == 1 and num_columns > 1) or (num_rows > 1 and num_columns == 1):
                    ax[i] = set_ax(ax[i], data[count], cmaptxt, vmin, vmax, vars[count], invertaxis)
                else:
                    ax = set_ax(ax, data[count], cmaptxt, vmin, vmax, vars[count], invertaxis)
                    
                count +=1
            else:
                if num_rows>1:
                    ax[j,i].axis('off')
                else:
                    ax[i].axis('off')
    #cbar = plt.colorbar(data.ravel())
    #cbar.set_label('X+Y')
    #cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    if colorbar:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7]) if colorbar_orientation == 'vertival' else None
            
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, orientation=colorbar_orientation,
                    cax=cbar_ax, pad=0.15)
        
        cb.ax.tick_params(labelsize=legtickssize)
        if label_name is not None:
            cb.set_label(label=label_name, fontdict=fontlegtitle)
        if newlegendticks:
            cb.ax.get_yaxis().set_ticks([])
            for j, lab in enumerate(newlegendticks):
                cb.ax.text(vmax, (7.2 * j + 2) / (vmax+3), lab,
                           ha='left', va='center',fontdict=fontlegtick)

    return fig,ax


def plot_datacube(xr_dict, dates = True, variable = None, limit = None, nrows = None, ncols = None, **kwargs):
    from datetime import datetime
    
    imgs = []
    labels = []
    for k,v in xr_dict.items():
        if dates:
            date = datetime.strptime(k, '%Y%m%d') 
            labels.append(date.strftime("%Y-%m-%d"))
        else:
            labels.append(k)
        imgs.append(v[variable].values)
        if limit and len(labels)>limit:
            break

    return plot_multichanels(np.array(imgs), num_columns=ncols, num_rows=nrows, chanels_names=labels, **kwargs) 
