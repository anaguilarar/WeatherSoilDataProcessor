# WeatherSoilDataProcessor

**WeatherSoilDataProcessor** is a Python-based tool designed for downloading, processing, and analyzing historical weather and soil data. The repository enables users to create multi-temporal data cubes for weather variables such as precipitation, temperature, and solar radiation, as well as download and process soil data. With built-in support for interactive visualizations and data export 
capabilities, this tool is ideal for scientific and agricultural research applications.

## Features
* **Weather Data Downloads**: Automatically fetch historical weather data from CHIRPS and AgEra5.
* **Soil Data Downloads**: Automatically fetch and download soil data from SoilGrids.
* **DSSAT Format Conversion**: Convert processed weather and soil data into DSSAT-compatible format for easy integration into DSSAT crop modeling software.

## Installation

* Python 3.7+
* pip -r requirements.txt

## Google Colab Examples

You can explore the functionality of this repository using Google Colab notebooks:

Example 1 (weathersoildata_processor_example):
- Download weather data using CHIRPS and AgEra5.
- Process and analyze soil data from SoilGrids.
- Create multi-temporal weather data cubes.
- Export processed data into tabular formats.

Example 2 (transform_datacube_to_dssat_files):
- Convert weather and soil data into DSSAT-compatible formats.
- Export processed data into tabular formats.