#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de



Created on Wed Oct  7 20:46:26 2020
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de
"""
import numpy as np
from numpy import frombuffer as fbuff

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