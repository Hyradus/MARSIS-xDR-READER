#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: MARSIS xDR-RAW DATA READER
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de

_________________________________ README _________________________________

## Create the environment using the yml

Using the terminal, move to the folder where is located MARSISv2.yml and run:
`conda env create -f MARSISv2.yml`

## Activate MARSISv2 environment

Just run:
`conda activate MARSISv2`

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

## General example

### CLI script
Just run `python xDR-RAW-Reader.py 

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

Created on Tue Jul 21 09:22:35 2020
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de
"""
from argparse import ArgumentParser
from tkinter import Tk,filedialog
import numpy as np
import os
import pathlib
import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
from statistics import mean
from shapely.geometry import LineString
from pyproj.crs import CRS
import itertools
import cv2 as cv
from numpy import frombuffer as fbuff
##### Import Utils
from utils.GenUtils import get_paths, make_folder, question
from utils.ReprojUtils import coordTransformer
from utils.DFUtils import DF_drop, geoDF2file, gdf_split, xDR_DF
from utils.SegyUtils import assemply_segy, save_segy
from utils.DBUtils import databaseUpdate
############################## GLOBAL VARIABLES #########################
N_pole_crs = CRS.from_proj4('+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs ')
S_pole_crs = CRS.from_proj4('+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs ')
marsORTHO = CRS.from_user_input('+proj=ortho +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs')
mars_sphere = CRS.from_proj4('+proj=longlat +R=3396190 +no_defs +type=crs')
marsPLNTC = CRS.from_proj4('+proj=longlat +a=3396190 +b=3376200 +no_defs')
marsPLANCE = CRS.from_user_input('GEOGCS["Mars 2000 planetocentric",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],PRIMEM["AIRY-0",180],UNIT["Degree",0.017453292519943295]]')
marsEQUI= CRS.from_user_input('PROJCS["Mars_Equidistant_Cylindrical",GEOGCS["Mars 2000",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Standard_Parallel_1",0],UNIT["Meter",1]]')
marsEQUI180 = CRS.from_user_input('PROJCS["Mars_Equidistant_Cylindrical",GEOGCS["Mars 2000",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",180],PARAMETER["Standard_Parallel_1",0],UNIT["Meter",1]]')
marsUTM = CRS.from_proj4('+proj=tmerc +lat_0=0 +lon_0=0 +k=0.9996 +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs')
marsMRCA = CRS.from_user_input('PROJCS["Mars_Mercator_AUTO",GEOGCS["Mars 2000",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Mercator"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Standard_Parallel_1",0],UNIT["Meter",1]]')
mars2000=CRS.from_wkt('GEOGCS["Mars 2000",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
#########################################################################



def DAT2FILE(image_dir, dump_dir, segy_dir, file_name, F, savedump, saveimg, savesegy, coordinates, dt, pp, scaler):
    
    import cv2 as cv
    
    for i in range(len(F)):
        if i == 0:
            freq='F1'
        else:
            freq='F2'
        img=np.array(F[i])            
        name = '/'+file_name+'_'+freq
        if saveimg in ['Y','y']:
            img=np.array(F[i])
            cv.imwrite(image_dir+name+'_original.png', img)
            # cv.imwrite(name+'original_255.png', img*255)
            if pp in ['Y','y']:
                imgNorm(img,image_dir,name)
                # imgDen(img_norm,name)
                imgScaler(img, image_dir,name)
        if savedump in ['Y','y']:
            name_dump = dump_dir+name+'_dump'
            np.save(name_dump, img)         
        
        if savesegy in ['Y','y']:
            xStart=coordinates[0][0]
            xEnd=coordinates[len(coordinates)-1][0]
            yStart=coordinates[0][1]
            yEnd=coordinates[len(coordinates)-1][1]
            segy = assemply_segy(img, xStart, xEnd, yStart, yEnd, dt , scaler)
            name_segy = segy_dir+name+'.sgy'
            save_segy(segy, name_segy)
                
            
    # imgEnh(imgs, basename)

def imgNorm(image, image_dir, name):
    import cv2 as cv
    image_norm= cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    name_norm=image_dir+name+'_normalized.png'
    cv.imwrite(name_norm,image_norm)
    return(image_norm)


def imgScaler(image, image_dir,name):
    from sklearn import preprocessing 
    min_max_scaler = preprocessing.MinMaxScaler()
    img_norm = (min_max_scaler.fit_transform(image)*255).astype(np.uint8)
    name_scal = image_dir+name+'_scaled.png'
    cv.imwrite(name_scal, img_norm)
    
def imgDen(image, name):
    import cv2 as cv
    image_den = cv.fastNlMeansDenoising((image).astype(np.uint8), None, 10,7,21)
    name_den=name+'_denoised.png'
    cv.imwrite(name_den, image_den)

def imgEnh(image, name):
    if isinstance(image, list):
        img_norm = []
        
        for im in image:
            img_norm.append(cv.normalize(im, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX))
                
        img_merge=(img_norm[0]+img_norm[1])*2
        cv.imwrite('Merged2.png',img_merge)
def raw_reader(file,offBytes,counter, FileRecord,Item, RecordBytes, precision):
    values = []
    
    for i in range(FileRecord):
        if 'S' in precision:
            values.append((fbuff(file, dtype=np.dtype(precision), count=1,
                                offset=offBytes+(RecordBytes*i))[0]).decode())
            
            
        else:
            values.append(fbuff(file, dtype=np.dtype(precision), count=1,
                                offset=Item+offBytes+(RecordBytes*i))[0])    
    return(values)


def RAW2GeoDF(xDR_df, xDR_gdf, xDrFile, ParamDF, def_crs):
    fname = pathlib.Path(xDrFile).name.split('.')[0]
    #F_NAME = re.split('_|\.', fname)[0:6]
    full_parameters = [] #All values for each parameter
    short_parameters = [] #Mean value for each parameter
    

    for i in range(len(ParamDF['NAME'])):
        #print(ParamDF['NAME'][i])
        offBytes=ParamDF['START_BYTES'][i]
        precision = ParamDF['DATA_TYPE'][i]
        Items = ParamDF['ITEMS'][i]
        ItemBytes=ParamDF['ITEM_BYTES'][i]
        file = open(xDrFile, 'rb')
        FileSize = pathlib.Path(xDrFile).stat().st_size
        RecordBytes=ParamDF['RECORD_BYTES'][0]
        FileRecord = int(FileSize/RecordBytes)
        f = file.read()
        if Items > 1 and 'S' not in precision:
            values=[]
            mean_values = []
            for l in range(Items):
                Item = l*ItemBytes
                val = np.array(raw_reader(f,offBytes,i,FileRecord,Item,RecordBytes, precision))
                values.append(val)
                mean_values.append(mean(val))
        else:
            Item=0
            values = np.array(raw_reader(f,offBytes,i,FileRecord,Item,RecordBytes, precision))
            if 'S' in precision:
                mean_values= values[0]
            else:
                mean_values=mean(values)
        full_parameters.append(values)
        short_parameters.append(mean_values)
        file_name=fname.split('.')[0]
        if ParamDF['NAME'][i] in ['SUB_SC_LONGITUDE','SUB_SC_LATITUDE']:
            min_val = min(values)
            max_val = max(values)
            short_parameters.append(min_val)
            short_parameters.append(max_val)
            #print(min_val, ' ', max_val)

    if DRTYPE in ['RDR', 'rdr','EDR','edr']:
        a, b,c,d = [11,17,29,30]
        F= [full_parameters[a], full_parameters[b]]
        coord = list(zip(full_parameters[c], full_parameters[d]))
        lat_mean = mean(full_parameters[d])
        dt_value = 7.14285714285714e-03
        scl = -10
        
        if lat_mean <=-65:
            segy_crs = S_pole_crs
        elif lat_mean >=65:
            segy_crs = N_pole_crs
        else:
            segy_crs = marsEQUI180
            
    elif DRTYPE in ['RAW','raw']:
        a, b,c,d = [9,11,21,22]
        F= [full_parameters[a], full_parameters[b]]
        coord = list(zip(full_parameters[c], full_parameters[d]))
        lat_mean = mean(full_parameters[d])
        scl = -1
        dt_value = 7.14285714285714e-03
        
        if lat_mean <=-65:
            segy_crs = S_pole_crs
        elif lat_mean >=65:
            segy_crs = N_pole_crs
        else:
            segy_crs = marsEQUI180

    segy_coord = coordTransformer(coord, def_crs, segy_crs)            
    DAT2FILE(IMG_DIR, DUMP_DIR, SEGY_DIR, file_name, F, SAVEDUMP, SAVEIMG, SAVESEGY,
             segy_coord, dt=dt_value ,pp='y',scaler=scl)
    
    #dst_crs = marsEQUI180
    dst_crs = DST_CRS
    # Create the linear geometry
    trans_coord = coordTransformer(coord, def_crs, dst_crs)
    track =  LineString(trans_coord)
    # create the geoSeries
    gser = gpd.GeoSeries(track)
    
    # Create a dataframe containing all values for each parameter
    proj4= dst_crs.to_proj4()#.to_wkt()
    meta = [fname, proj4]
    temp_full = meta+full_parameters
    series_full = pd.Series(temp_full, index=xDR_df.columns)
    xDR_df = xDR_df.append(series_full,ignore_index=True)
    # Create a temporary dataframe containing all mean values for each parameter
    temp_short = meta+short_parameters
    
    if DRTYPE in ['RDR','rdr']:
        aa, bb, cc, dd = [32,33,35,36]
    else:
        aa, bb, cc, dd = [22,23,25,26]
        
    short_cols = xDR_df.columns.tolist()
    short_cols.insert(aa,'c1min')
    short_cols.insert(bb,'c1max')
    short_cols.insert(cc,'c2min')
    short_cols.insert(dd,'c2max')
    
    series_short = pd.Series(temp_short, index=short_cols)
    df = pd.DataFrame(columns=short_cols).append(series_short, ignore_index=True)
    # Create the geodataframe containing all mean values and geometry for each parameter
    xDR_geodf = gpd.GeoDataFrame(df, crs = dst_crs, geometry=gser)
    return(xDR_geodf, xDR_df, coord, trans_coord, segy_coord)

def parallel_df(files, JOBS,FM_df, FM_gdf, GeomDF, def_crs):
    from joblib import Parallel, delayed
    results = Parallel (n_jobs=JOBS)(delayed(RAW2GeoDF)(FM_df, FM_gdf, files[i], GeomDF, def_crs)
                            for i in range(len(files)))
    return (results)

def chunk_creator(item_list, chunksize):
    it = iter(item_list)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            break
        yield chunk

def main():        
    # List all files
    all_filenames = get_paths(DATA_PATH, 'dat')
    
    # Check available resources
    import psutil
    avram=psutil.virtual_memory().total >> 3
    if avram > 31 and len(all_filenames) <5000:
        JOBS=psutil.cpu_count(logical=True)
    elif avram > 31 and len(all_filenames)>5000:
        JOBS=psutil.cpu_count(logical=True)
    elif avram <=31 and len(all_filenames)<5000:
        JOBS=psutil.cpu_count(logical=True)
    elif avram <= 31 and len(all_filenames) > 5000:
        JOBS=psutil.cpu_count(logical=False)
    
    # Create chunks for parallel processing
    filerange = len(all_filenames)
    chunksize = round(filerange/JOBS)
    if chunksize <1:
        chunksize=1
        JOBS = filerange
    chunks = []
    for c in chunk_creator(all_filenames, JOBS):
        chunks.append(c)
               
    # Load sub-functions
    if DRTYPE in ['RDR', 'rdr','EDR','edr']:
        from utils.DFUtils import xDR_params
        def_crs = marsPLNTC
        ParamDF=xDR_params()
    
    elif DRTYPE in ['RAW','raw']:
        from utils.DFUtils import RAW_params
        def_crs = marsPLNTC
        ParamDF = RAW_params()
        
    # Initialize dataframes
    xDR_df = xDR_DF(ParamDF)
    xDR_gdf = gpd.GeoDataFrame(xDR_df)
    results = []
    
    # Parallel processing
    with tqdm(total=len(all_filenames),
             desc = 'Generating files',
             unit='File') as pbar:
        
        for i in range(len(chunks)):
            files = chunks[i]    
            # print(files)
            chunk_results = parallel_df(files, JOBS,xDR_df, xDR_gdf, ParamDF, def_crs)
            
            for r in chunk_results:
                results.append(r)
                xDR_df = pd.concat([xDR_df, r[1]])
                xDR_gdf = pd.concat([xDR_gdf, r[0]])
            pbar.update(JOBS)

    # Sort dataframe elements by orbit number and drop incompatible Parameters from dataframe
    xDR_df = xDR_df.sort_values(by='name', ascending=True, ignore_index=True)    
    xDR_gdf = xDR_gdf.sort_values(by='name', ascending=True, ignore_index=True)    
    xDR_gdf = DF_drop(xDR_gdf, DRTYPE)
   
    if DRVR is None:
        print('All Done')
        xDR_gdf_NPole = None
        xDR_gdf_SPole = None
    else:        
        # Save geopackages     
        print('\nSaving Full geopackate')    
        # geoDF2file(xDR_gdf, 'Complete', marsEQUI180, SAVE_DIR, DRVR)
        geoDF2file(xDR_gdf, 'Complete', [DST_CRS, DST_CRS], SAVE_DIR, DRVR)
        print('\nSaving N-Pole geopackage')
        xDR_gdf_NPole = gdf_split(xDR_gdf, 65)
        geoDF2file(xDR_gdf_NPole, 'NPole', [DST_CRS, N_pole_crs], SAVE_DIR, DRVR)
        print('\nSaving S-Pole geopackage')
        xDR_gdf_SPole = gdf_split(xDR_gdf, -65)
        geoDF2file(xDR_gdf_SPole, 'SPole', [DST_CRS, S_pole_crs], SAVE_DIR, DRVR)
        print('All done')
        
        
    if DBUP in ['Y','y']:
        error = databaseUpdate(xDR_gdf)
        if error == False:
            print('Database updated')
        else:
            print(error)
        
    return(xDR_gdf, xDR_df, xDR_gdf_NPole, xDR_gdf_SPole)


if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument('--wdir', help='Output folder: ')
    parser.add_argument('--ddir', help='Input files folder: ')
    parser.add_argument('--drv', help='Driver to save files GPKG/gpkg or SHP/shp')
    parser.add_argument('--drt', help='Data type: EDR/RDR/FM/RAW')
    parser.add_argument('--sim', help='Save images?')
    parser.add_argument('--sdum', help='Save dumps?')
    parser.add_argument('--sgy', help='Save seg-y?')
    
    args = parser.parse_args()
    WORK_PATH = args.wdir
    DATA_PATH = args.ddir
    DRVR = args.drv
    DRTYPE = args.drt
    SAVEIMG=args.sim
    SAVEDUMP=args.sdum
    SAVESEGY=args.sgy

    ## PATHS
    if WORK_PATH is None:
        root = Tk()
        root.withdraw()
        WORK_PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Select output folder")
        print('Output folder:', WORK_PATH)
        
    if DATA_PATH is None:
        root = Tk()
        root.withdraw()
        DATA_PATH = filedialog.askdirectory(parent=root,initialdir=os.getcwd(),title="Select input files folder")
        print('Input files folder:', DATA_PATH)
    
            
    if DRTYPE is None:
        DRTYPE = question('Enter data type: EDR/RDR/RAW',['EDR','edr','RDR','rdr','RAW','raw'])
     
    if DRVR is None:
        qst = question('Save geopackagse/shapefiles',  ['Y','y','N','n'])
        if qst in ['Y','y']:
            DRVR= question('Enter output file type: GPKG/gpkg or SHP/shp', ['GPKG','gpkg','SHP','shp'])
    if DRVR in ['GPKG','gpkg']:
        SAVE_DIR = make_folder(WORK_PATH,'Geopackages')
    elif DRVR in ['SHP', 'shp']:
        SAVE_DIR = make_folder(WORK_PATH,'Shapefiles')
    else:
        DRVR = None

    ## IMAGES AND DUMPS
    if SAVEIMG is None:
        SAVEIMG = question('Save images?', ['Y','y','N','n'])
    if SAVEDUMP == None:
        SAVEDUMP = question('Dump arrays?', ['Y','y','N','n'])
    if SAVESEGY is None:
        SAVESEGY = question('Save Seg-y', ['Y','y','N','n'])
        
    ## FOLDERS        
    if SAVEIMG in ['Y', 'y']:
        IMG_DIR = make_folder(WORK_PATH,'Images')
    else:
        IMG_DIR=None
    if SAVEDUMP in ['Y', 'y']:
        DUMP_DIR = make_folder(WORK_PATH, 'Dumps')
    else:
        DUMP_DIR=None
    if SAVESEGY in ['Y', 'y']:
        SEGY_DIR = make_folder(WORK_PATH, 'Seg-y')
    else:
        SEGY_DIR = None
   
    DST_CRS = input('Destination CRS in wkt/proj4 format - Leave empty for Mars Ecquirectangular') or marsEQUI180

    
    DBUP = question('Save results into postgres database?',['Y','y','N','n'])
    
    # main()    
    xDR_gdf, xDR_df, xDR_gdf_NPole, xDR_gdf_SPole = main()
    
    
