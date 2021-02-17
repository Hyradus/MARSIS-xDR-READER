#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de



Created on Fri Dec 18 12:18:14 2020
@author: @author: Giacomo Nodjoumi g.nodjoumi@jacobs-unversity.de
"""

def databaseUpdate(xDR_gdf): #Function to insert dataframe columns into postgres + postgis database
    
    import psycopg2              
    connection = psycopg2.connect(host='localhost',port='5432',
                                  database="matisse_db",
                                  user="postgres",
                                  password="test")
    
    cursor = connection.cursor()
    
    for i, row in xDR_gdf.iterrows():
        geom = xDR_gdf['geometry'][i]
        geomwb = geom.wkb_hex
        data_dict = {'geom':geomwb,
                     'srid':49900,
                      'name':xDR_gdf['name'][i],
                     # 'name':'ciao12'+str(i),
                      'proj4':xDR_gdf['proj4'][i],
                      # 'proj4':'ciao',
                      'c1min':xDR_gdf['c1min'][i],
                      'c1max':xDR_gdf['c1max'][i],
                      'c2min':xDR_gdf['c2min'][i],
                      'c2max':xDR_gdf['c2max'][i]
                     # 'c1min':0,
                     # 'c1max':0,
                     # 'c2min':0,
                     # 'c2max':0
                     }
        data_sql = ('INSERT INTO geodata(geom, name, proj4, c1min, c1max,c2min,c2max)'
                    'VALUES (ST_SetSRID(%(geom)s::geometry, %(srid)s), %(name)s, %(proj4)s,'
                    '%(c1min)s, %(c1max)s, %(c2min)s, %(c2max)s)')
        try:
            # print(data_dict['proj4']) 
            cursor.execute(data_sql, data_dict) 
            connection.commit()
          
        except (Exception, psycopg2.Error) as error:
            print('Error: ', error)
            connection.rollback()        
            pass
        connection.commit()
        
    
    query = "select * from geodata"
    cursor.execute(query)
    data = cursor.fetchall()
    
    for i in range(len(data)):
        geofileid = data[i][0]
        name = data[i][1]
        obtyid = 1
        ftype = 1
        meta_sql = ('INSERT INTO observationmetadatafill(name, geofileid, observationtypeid,filetype)'
               'VALUES (%(name)s, %(geofileid)s, %(observationtypeid)s, %(filetype)s)')
        meta_dict = {'name':name, 'geofileid':geofileid, 'observationtypeid':obtyid,'filetype':ftype}
        try:
            cursor.execute(meta_sql,meta_dict)
            connection.commit()
            err = None
        except (Exception, psycopg2.Error) as error:
            
            err = error
            print(error)
            connection.rollback()
            pass
        connection.commit()
        
    connection.close()
    
    return(err)
    cursor.execute('TRUNCATE TABLE geodata;')
    cursor.execute('TRUNCATE TABLE observationmetadatafill;')
    connection.commit()
    
    
    # from shapely import wkb
    # cursor.execute('SELECT name, geom FROM geodata')
    # for name, geom_wkb in cursor:
    #     geom = wkb.loads(geom_wkb, hex=True)
    #     print('{0}:{1}'.format(name, geom.wkt))