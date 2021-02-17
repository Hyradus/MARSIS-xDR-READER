# SCRIPT for read MARSIS EDR, RDR and RAW files and export Images, Numpy Dumps and geopackages/shapefiles
### @author: Giacomo Nodjoumi - g.nodjoumi@jacobs-university.de

# README
________________________________________________________________________________
# Table of Contents

- [Pipeline/workflow description](#pipeline-workflow-description)
- [Differences between CLI script and notebook](#differences-between-cli-script-and-notebook)
- [CONDA environment](#conda-environment)
  * [Install anaconda](#install-anaconda)
  * [Create and activate the environment using the yml](#create-and-activate-the-environment-using-the-yml)
- [Script execution](#script-execution)
  * [Arguments that can be passed [ONLY CLI SCRIPT]](#arguments-that-can-be-passed--only-cli-script-)
    + [Output directory](#output-directory)
    + [Data directory](#data-directory)
    + [Driver for saving GIS files](#driver-for-saving-gis-files)
    + [Data record type](#data-record-type)
    + [Save images flag](#save-images-flag)
    + [Save numpy dumps](#save-numpy-dumps)
    + [Save SEG-Y](#save-seg-y)
    + [Update Database](#update-database)
  * [General example](#general-example)
    + [CLI script](#cli-script)
  * [Outputs:](#outputs-)
    + [GIS OUTPUTS](#gis-outputs)
    + [Image outputs](#image-outputs)
    + [SEG-Y outputs](#seg-y-outputs)
    + [Ingestion into postgres+postgis database](#ingestion-into-postgres-postgis-database)
    + [SEG-Y outpust](#seg-y-outpust)
  * [Test example](#test-example)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

________________________________________________________________________________
# Pipeline/workflow description

The script in brief:

* Ask user to provide arguments if not passed. e.g. data path, type of file, etc.
* Read all files available in the provided folder and export all flagged elements

** See example at the end of this readme**

The script can work both passing some/all arguments or none ***If NO argument is passed, defaults are used and interactively requested the others.***

# Differences between CLI script and notebook

The only difference is that arguments can only be passed to CLI script while notebook ask them interactively

# CONDA environment

To best use the script a conda environment configuration file is provided: ***MARSISv2.yml***

## Install anaconda

Installer of anaconda for different operating systems are provided on the official page. [Anaconda Installers](https://www.anaconda.com/products/individual)

To install conda on linux, download [this](https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh) file, 
with terminal move to the downloaded folder and run:
* `sudo chmod +x Anaconda3-VERSION-Linux-x86_64.sh` (replace VERSION with the proper filename)
* `sudo ./Anaconda3-VERSION-Linux-x86_64.sh`

## Create and activate the environment using the yml

Using the terminal, craate conda env using provided MARSISv2.yml:
* `conda env create -f MARSISv2.yml`
* `conda activate MARSISv2`

# Script execution

To execute, simple run the following code `python xDR|RAW-Reader.py`
It will ask every arguments.

## Arguments that can be passed [ONLY CLI SCRIPT]

### Output directory
`--wdir` path where all files are saved

### Data directory
`--ddir` path where are all files to be processed

### Driver for saving GIS files
`--drv` choose between GPKG, gpkg, SHP, shp

### Data record type
`--drt` insert choice between EDR, edr, RDR, rdr, RAW, raw

### Save images flag
`--sim` insert choice between Y,y,N,n

### Save numpy dumps
`--sdum` insert choice between Y,y,N,n

### Save SEG-Y
`--sgy` insert choice between Y,y,N,n

### Update Database
`--dbup` insert choice between Y,y,N,n

## General example

### CLI script
Just run `python xDR-RAW-Reader.py 

If NO argument is passed, the scripts ask interactively:
    
* Output Directory: path where all files are saved
* Data directory: path where are all files to be processed
* Driver for gis output: insert choice between GPKG, gpkg, SHP, shp
* Data record type: insert choice between EDR, edr, RDR, rdr, RAW, raw
* Flag for save images of both frequencies: insert choice between Y,y,N,n
* Flag for save numpy dump of both frequencies: insert choice between Y,y,N,n
* Flag for save seg-y of both frequencies: insert choice between Y,y,N,n
* Flag for ingest data into database Y,y,N,n
* Flag for select destination CRS

## Outputs:
### GIS OUTPUTS
It creates three different geopackages:
    - FULL with all orbits
    - North Pole with orbits from 65°->90° Latitude
    - South Pole with orbits from -65°->-90° Latitude
    
### Image outputs
As default it creates thre types of images for each frequency:
* Original image
* Normalized image
* Scaled image using sklearn MinMaxScaler

*Further image processing is in development*


### SEG-Y outputs
It export a seg-y file for each frequency of each image

### Ingestion into postgres+postgis database
Connection to database and ingestion parameters must be set into utils/DB_utils.py
**Provided configuration is just an example**

### SEG-Y outpust
It export a seg-y file for both frequency

Example of seg-y imported into opendTect
![alt text](Readme_images/segy_opendtect.jpg?raw=true "seg-y opendTect")

And 2d track, showing dt in a major scale
![alt text](Readme_images/segy_opendtect_2d_image.jpg?raw=true "seg-y opendTect")
## Test example

Here the example code shown in the image


![alt text](Readme_images/test.jpg?raw=true "Test")
