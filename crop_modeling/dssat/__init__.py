import sys
import importlib.util
import os
dssattools_spec = importlib.util.find_spec("DSSATTools")

if dssattools_spec is not None:
    try:
        from DSSATTools.weather import Weather
    except:
        from DSSATTools.weather import Weather

else:
    try:
        from Py_DSSATTools.DSSATTools.weather import Weather
    except:
        from Py_DSSATTools.DSSATTools.weather import Weather
