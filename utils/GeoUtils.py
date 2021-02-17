#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de



Created on Fri Dec 18 16:24:36 2020
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de
"""
from pyproj import Transformer

def coordTransformer(coordinates, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs)
    coord_new=[]
    for crds in transformer.itransform(coordinates):
        coord_new.append(crds)
    return(coord_new)