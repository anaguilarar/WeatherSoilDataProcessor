import numpy as np
import os
import pandas as pd

try:
    from DSSATTools.weather import Weather
    from DSSATTools.soil import SoilProfile, SoilLayer
except:
    from DSSATTools.weather import Weather
    from DSSATTools.soil import SoilProfile, SoilLayer
    
