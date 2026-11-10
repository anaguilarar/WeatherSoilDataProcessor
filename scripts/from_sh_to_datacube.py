import shutil
import argparse
from pathlib import Path
import sys

from aggeodata.transform.soil_cube import SoilDataCubeBuilder



NAME_MAP = {
    "bdod": "DAP_g_cm-3.tif",
    "clay": "Arcilla_%.tif",
    "sand": "Arena_%.tif",
    "silt": "SILT_%.tif",
    "phh2o": "pHAQ_un.tif",
    "cec": "CIC_cmol(+)_kg-1.tif",
    "ce": "CEAQ_dS_m-1.tif",
    "som": "MO_%.tif",
    "kdis": "KDIS_cmol(+)_kg-1.tif",
    "pdis": "PDIS_mg_kg-1.tif",
}

def main():
    
    p = argparse.ArgumentParser(description=__doc__, formatter_class= argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required= True)
    p.add_argument('--dst', required= True)
    p.add_argument('--fn', required = True)
    p.add_argument('--depths', default=["0-5", "5-15", "15-30"])
    
    
    args = p.parse_args()
    depths = args.depths
    
    for var, filename in NAME_MAP.items():
        for depth in depths:
            shutil.copy2(args.src, filename, args.dst/ f"{var}_{depth}cm_mean_30s.tif")
            
    builder = SoilDataCubeBuilder(data_folder = args.dst, variables = list(NAME_MAP), reference_variable = "sand", target_crs = "EPSG:4326")
    builder.build_and_save(output_path = args.dst,
                           filename = args.fn,
                           )
    
    
if __name__ == "__main__":
    
    main()
    
    