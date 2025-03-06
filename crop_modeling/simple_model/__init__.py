
#https://github.com/ataraxno/SIMPLE_crop_model/blob/main/1_SIMPLE_implementation.ipynb
SOIL_PARAMETERS = {
    'AWC': 'water-holding capacity',
    'RCN': 'runoﬀ curve number', # donde
    'DDC': 'deep drainage coeﬃcient', ## SDUL ?
    'RZD': 'root zone depth' # dssat # 1500?
}

## drained upper limit (sdul)
RUNOFF_CURVES = (73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 68.0, 73.0, 68.0, 68.0, 73.0)

TEXTURE_CLASSES = ("clay", "silty clay", "sandy clay", "clay loam", "silty clay loam", "sandy clay loam", "loam", 
                    "silty loam", "sandy loam", "silt", "loamy sand", "sand", "unknown")

