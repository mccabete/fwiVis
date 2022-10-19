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
        
## Finding stations
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

### Loading Fire files
def load_file(date,layer='perimeter',handle_multi=False,
              only_lf=False,area_lim=5,show_progress=False):
    '''
    loads in snapshot file based on input date and layer
    
    INPUTS:
        
        date (str): string in the form YYYYMMDDAM (or PM)
        layer (str): either "perimeter", "fireline", or "newfirepix"
        handle_multi (bool): drop fire ids in snapshot data that have several polygons
                             associated with them. these are usually several close together
                             static fires that should be filtered out.
        only_lf (bool): only display fires with polygons > area_lim
        area_lim (int): value in km2 to use as lower threshold for largefire filter
        show_progress (bool): print out the file's date once it's loaded.
                              this is helpful when using the function in a loop.
       This function was authored by Eli. 
    
    '''
    
    base_path = '/projects/shared-buckets/gsfc_landslides/FEDSoutput-s3-conus/CONUS/2019/Snapshot/'
    full_path = os.path.join(base_path,date)
    try: 
        gdf = gpd.read_file(full_path,layer=layer)
        if show_progress:
            print(date,'loaded')
    except: 
        print('could not load file',full_path)
        return None
    
    if handle_multi:
        multi_geoms = gdf.loc[gdf.geometry.geometry.type=='MultiPolygon'].index
        gdf['NumPolygons'] = gdf.loc[multi_geoms,'geometry'].apply(lambda x: len(list(x)))
        too_many_polygons = gdf[gdf['NumPolygons']>4].index
        gdf.drop(too_many_polygons,inplace=True)
    
    if only_lf:
        gdf = gdf[gdf['farea']>area_lim]
    
    return gdf

def prep_gdf(date = '20191031PM',handle_multi=True,only_lf=True,area_lim=5, index ='fireID',inplace=True):
    gdf = load_file('20191031PM',handle_multi=True,only_lf=True,area_lim=5)
    gdf.set_index(index, inplace)
    print("Total number of fires:",len(gdf.index.unique()))
    print("These were the files availible at" + date + " and larger than", area_lim, "found at CONUS")
    
    gdf_test.to_crs('EPSG:4326')
    print("Setting projection to EPSG:4326")
    gdf_test['lon'] = gdf_test.centroid.x
    gdf_test['lat'] = gdf_test.centroid.y
    
    return(gdf)

def fire_search (gdf, stations, dist_max_km = 112.654): # ~ 70 miles distance
    
    st_dict = stations[['Lat', 'Lon']].to_dict('records')
    small_map = gdf[["lon", 'lat']]

    fire_st_coloc = []
    
    for i, val in enumerate(small_map.index):
        foi = {'Lat': small_map.lat.iloc[i], 'Lon': small_map.lon.iloc[i]}
        cls_st_srch = closest_srch(st_dict, foi)
        cls_st_srch['fireID'] = val
        fire_st_coloc.append([cls_st_srch.Lat[0], cls_st_srch.Lon[0], cls_st_srch.dist_km[0], cls_st_srch.Origin_lat[0], cls_st_srch.Origin_lon[0], cls_st_srch.fireID[0]])
        
    df = pd.DataFrame(fire_st_coloc, columns = ["Lat", "Lon", "dist_km","Origin_lat", "Origin_lon", "fireID"])
    #df = pd.DataFrame(fire_st_coloc)
    df_small = df[df.dist_km <= dist_max_km]
    return(df_small)