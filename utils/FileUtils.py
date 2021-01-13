#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de



Created on Fri Dec 18 16:25:37 2020
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de
"""
import numpy as np
from utils.ImgUtils import imgNorm, imgDen, imgScaler
from utils.SegyUtils import assemply_segy, save_segy

def DAT2FILE(image_dir, dump_dir, segy_dir, file_name, F, savedump, saveimg, savesegy, coords, dt, pp, scaler):
    
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
                img_norm = imgNorm(img,image_dir,name)
                imgDen(img_norm,image_dir,name)
                imgScaler(img, image_dir,name)
        if savedump in ['Y','y']:
            name_dump = dump_dir+name+'_dump'
            np.save(name_dump, img)         
        
        if savesegy in ['Y','y']:
            xStart=coords[0][0]
            xEnd=coords[len(coords)-1][0]
            yStart=coords[0][1]
            yEnd=coords[len(coords)-1][1]
            segy = assemply_segy(img, xStart, xEnd, yStart, yEnd, dt , scaler)
            name_segy = segy_dir+name+'.sgy'
            save_segy(segy, name_segy)