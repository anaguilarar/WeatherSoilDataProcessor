# WeatherSoilDataProcessor

**WeatherSoilDataProcessor** is a Python-based tool designed for downloading, processing, and analyzing historical weather and soil data. The repository enables users to create multi-temporal data cubes for weather variables such as precipitation, temperature, and solar radiation, as well as download and process soil data. With built-in support for interactive visualizations and data export 
capabilities, this tool is ideal for scientific and agricultural research applications.

## Features
* **Weather Data Downloads**: Automatically fetch historical weather data from CHIRPS and AgEra5.
* **Soil Data Downloads**: Automatically fetch and download soil data from SoilGrids.
* **DSSAT Format Conversion**: Convert processed weather and soil data into DSSAT-compatible format for easy integration into DSSAT crop modeling software.

## Installation

* Python 3.7+
* pip install -r requirements.txt


## How to Use
### Example: Generating Weather and Soil Data Cubes with DSSAT Export
This script generates multi-temporal weather and multi-depth soil data cubes and exports them in DSSAT-compatible format. To run the script, you need to provide a configuration file (in YAML format) that specifies the paths, variables, and settings for the process.
```bash
python create_crop_weather_dssat_files.py --config generate_dssat_files_per_aoi_and_group.yaml
```

Here’s an example of what your options.yaml file should include:
```yaml

SPATIAL_INFO:
  boundaries: "path/to/shapefile.shp"

WEATHER:
  paths:
    precipitation: "path/to/precipitation_data"
    srad: "path/to/solar_radiation_data"
    tmax: "path/to/max_temperature_data"
    tmin: "path/to/min_temperature_data"
  starting_date: "2001-01-01"
  ending_date: "2001-12-31"
  reference_variable: "precipitation"
  scale_factor: 10

SOIL:
  path: "path/to/soil_data"
  variables: ["sand", "clay", "silt"]
  depths: ["0-5","5-15","15-30","30-60"]
  crs_reference: "EPSG:4326"
  reference_variable: sand

ROI:
  path: "path/to/roi_shapefile.shp"
  roi_column: "region_column_name"

GENERAL:
  ncores: 4
  crs_reference: "EPSG:4326"

GROUPBY:
  variable: "texture"

DSSAT:
  variable_names:
    DATE: "date"
    TMIN: "tmin"
    SRAD: "srad"
    RAIN: "precipitation"
    TMAX: "tmax"
    LON: "x"
    LAT: "y"

PATHS:
  output_path: "path/to/export_data"

```

## Google Colab Examples

You can explore the functionality of this repository using Google Colab notebooks:

Example 1 [(weathersoildata_processor_example)](https://github.com/anaguilarar/WeatherSoilDataProcessor/blob/main/weathersoildata_processor_example.ipynb):
- Download weather data using CHIRPS and AgEra5.
- Process and analyze soil data from SoilGrids.
- Create multi-temporal weather data cubes.
- Export processed data into tabular formats.

Example 2 [(transform_datacube_to_dssat_files)](https://github.com/anaguilarar/WeatherSoilDataProcessor/blob/main/transform_datacube_to_dssat_files.ipynb):
- Convert weather and soil data into DSSAT-compatible formats.
- Export processed data into tabular formats.

