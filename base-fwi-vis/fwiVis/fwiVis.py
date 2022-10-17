import s3fs
s3 = s3fs.S3FileSystem(anon=False)
from math import cos, asin, sqrt

import re
import numpy as np
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import os
import rioxarray as rio
import xarray as xr
import rasterio
import glob
from geocube.api.core import make_geocube
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point
import warnings
import folium
from folium import plugins
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

def st_avail(files, st_id_map, inter_type = "spline.HourlyFWIFromHourlyInterpContinuous"):
    file_inter = []
    for path in files:
        if inter_type in path:
            file_inter.append(path)

    df = []
    for i in file_inter:
        pt_1 = re.sub("veda-data-store-staging/EIS/other/station-FWI/20000101.20220925.hrlyInterp/FWI/", "", i)
        #pt_2 = re.sub(".spline.DailyFWIfromHourlyInterp.csv","",  pt_1)
        pt_2 = re.sub(("." + inter_type + ".csv"),"",  pt_1)
        pt_3 = pt_2.split("-")
        usaf = re.sub(r'[^0-9]', '',pt_3[0]) ## Sometimes ID had extra characters? 
        #print(usaf)
        wban = re.sub(r'[^0-9]', '',pt_3[1]) 
        #print(wban)
        
        st = st_id_map.loc[(st_id_map.USAF == usaf) | (st_id_map.WBAN == wban)]
        #print(st)
        if(st.empty):
            print("Empty Dataframe")
            break
        
        lat = st.LAT.iloc[0]
        lon = st.LON.iloc[0]

        df.append({
            "File_path": i,
            "Lat": lat,  
            "Lon": lon,
            "USAF": usaf,
            "WBAN": wban
        })
   

   
    return(pd.DataFrame(df))

def hour_fix (hr):
    small_hr = ["0","1","2","3","4","5","6","7","8","9"]
    append = ""
    
    if (hr in small_hr):
        append = "0"
    hr = append + hr
    
    return(hr)

## Function for parsing into datetime objects
# Takes a pandas dataframe of the same format as found: 
#'s3://veda-data-store-staging/EIS/other/station-FWI/20000101.20220925.hrlyInterp/FWI/727850-24157.spline.HourlyFWIFromHourlyInterpContinuous.csv'

def date_convert(dat_time):
    dat_time.HH = dat_time.HH.astype('int') # Drop #.0
    dat_time.HH = dat_time.HH.astype('str')
    dat_time['HH_format'] = dat_time.HH.apply(hour_fix)
    dat_time['time'] = pd.to_datetime(dat_time['YYYY'].astype(str) +"-"+  dat_time['MM'].astype("str") +"-"+  dat_time['DD'].astype("str") + " " + dat_time['HH_format'].astype('str'), format='%Y-%m-%d %H')

    return(dat_time)

def get_st(lat, lon, stations, flag_bad = True):
        
    st = stations.loc[(stations.Lat == lat) & (stations.Lon == lon)]
    #dat = pd.read_csv("s3://veda-data-store-staging/EIS/other/station-FWI/20000101.20220907.hrlyInterp/FWI/727970-94240.spline.DailyFWIfromHourlyInterp.csv")
    dat = pd.read_csv(("s3://" + st.File_path.iloc[0]), index_col = False)
    dat = date_convert(dat)
    
    if flag_bad:
        mask = dat['OBSMINUTEDIFFTEMP'].loc[dat.OBSMINUTEDIFFTEMP > 20]  ## Kluge. Basically, use this as a flag. 
        dat.iloc[mask.index, 4:-4 ] = np.nan
    
    return(dat)

def plot_st(lat, lon, stations, all_plot = True, times = "none_passed", flag_bad = True, time_start = "default", time_end = "default"):
    
    st = stations.loc[(stations.Lat == lat) & (stations.Lon == lon)]
    #dat = pd.read_csv("s3://veda-data-store-staging/EIS/other/station-FWI/20000101.20220907.hrlyInterp/FWI/727970-94240.spline.DailyFWIfromHourlyInterp.csv")
    dat = pd.read_csv(("s3://" + st.File_path.iloc[0]), index_col = False)
    
    
    if flag_bad:
        mask = dat['OBSMINUTEDIFFTEMP'].loc[dat.OBSMINUTEDIFFTEMP > 20]  ## Kluge. Basically, use this as a flag. 
        dat.iloc[mask.index, 3:-4 ] = np.nan
        
    if (times == "none_passed"):
        times = np.arange(np.datetime64('2000-01-01 12:00:00'),
                          np.datetime64('2022-09-24 20:00:00'), np.timedelta64(60, 'm')) # Generate timesereis on fly bc parsing files headache # Time shifted????
        dat["time"] = times
        
        if ((time_start != "default") & (time_end != "default")):
            
            dat = dat[(dat['time'] > time_start) & (dat['time'] < time_end )] # For "zooming" to particular time

    for col in dat.columns[4:-6]:
        plt.figure()
        plt.plot(dat["time"], dat[col])
        plt.xlabel('Year')
        plt.ylabel('')
        plt.title(col)
        plt.xticks(rotation = 45)

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav)) ## Returns in km

def closest(data, v):
    mn = min(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
    dist = distance(v['Lat'],v['Lon'], mn['Lat'], mn['Lon'])
    print("The closest station is", dist, "km away." )
    return mn

def closest_srch(data, v):
    mn = min(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
    dist = distance(v['Lat'],v['Lon'], mn['Lat'], mn['Lon'])
    mn = pd.DataFrame([mn])
    mn['dist_km'] = dist
    return mn

