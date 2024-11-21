from spatialdata.gis_functions import add_2dlayer_toxarrayr
from spatialdata.soil_data import find_soil_textural_class_in_nparray

import numpy as np
from rosetta import SoilData, rosetta


## 'Sat. hydraulic conductivity, macropore, cm h-1'
def calculate_sks(sand: float = None, 
                  silt: float = None, 
                  clay: float = None, 
                  bulk_density: float = None, 
                  field_capacity: float = None, 
                  permanent_wp: float = None) -> float:
    """
    Calculate Saturated Hydraulic Conductivity (Ks) in cm/h based on soil properties.

    Parameters
    ----------
    sand : float, optional
        Percentage of sand in the soil (0-100).
    silt : float, optional
        Percentage of silt in the soil (0-100).
    clay : float, optional
        Percentage of clay in the soil (0-100).
    bulk_density : float, optional
        Bulk density of the soil in g/cm³.
    field_capacity : float, optional
        Field capacity of the soil in cm³/cm³.
    permanent_wp : float, optional
        Permanent wilting point of the soil in cm³/cm³.

    Returns
    -------
    float
        Estimated saturated hydraulic conductivity (Ks) in cm/h.

    Raises
    ------
    ValueError
        If not enough information is provided to calculate sand, silt, or clay percentages.
    """
    
    if sand is None and silt is not None and clay is not None:
        sand = 100 - silt - clay
    elif silt is None and sand is not None and clay is not None:
        silt = 100 - sand - clay 
    elif clay is None and sand is not None and silt is not None:
        clay = 100 - sand - silt  

    elif sand is None or silt is None or clay is None:
        raise ValueError("At least two of sand, silt, or clay percentages must be provided.")
    
    datinfo =  [i for i in [sand, silt,clay, bulk_density, field_capacity, permanent_wp] if i]
    
    soil_data = SoilData.from_array(
                [datinfo]
            )
    
    vangenuchten_pars, _, _ = rosetta(3, soil_data)

    ks = (10**vangenuchten_pars[0][-1]) # Convert log(Ks) to Ks in cm/day / 24

    ks = ks / 24  # Convert cm/day to cm/hour
    return ks



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


def slu1(clay:float, sand:float) -> float:
    """
    Calculate Stage 1 evaporation limit (slu), in mm.

    Parameters
    ----------
    clay : float
        Percentage of clay in the soil.
    sand : float
        Percentage of sand in the soil.

    Returns
    -------
    float
        The Stage 1 evaporation limit in millimeters.

    Raises
    ------
    ValueError
        If the clay or sand percentage is not in the range of 0 to 100.
    """
    if not (0 <= clay <= 100):
        raise ValueError("Clay percentage must be between 0 and 100.")
    if not (0 <= sand <= 100):
        raise ValueError("Sand percentage must be between 0 and 100.")
    

    if sand>=80:
        lu = 20-0.15*sand
    elif clay>=50:
        lu = 11 - 0.06*clay
    else:
        lu = 8-0.08*clay
    return lu


def get_layer_texture(soilxrdata_layer, texture_name = 'texture'):
    sand = soilxrdata_layer.sand.values*0.1 if np.nanmax(soilxrdata_layer.sand.values) > 300 else soilxrdata_layer.sand.values
    clay = soilxrdata_layer.clay.values*0.1 if np.nanmax(soilxrdata_layer.clay.values) > 300 else soilxrdata_layer.clay.values
    texturemap = find_soil_textural_class_in_nparray(sand, clay).astype(float)
    texturemap[texturemap == 0] = np.nan
    return add_2dlayer_toxarrayr(texturemap, soilxrdata_layer, variable_name=texture_name)