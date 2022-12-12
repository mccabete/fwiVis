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

def st_avail(files, st_id_map, inter_type = "spline.HourlyFWIFromHourlyInterpContinuous", path_s3 = "veda-data-store-staging/EIS/other/station-FWI/20000101.20220925.hrlyInterp/FWI/"):
    '''
   Takes a list of stations at a files path. Subsets by a specific interpolation type, and then parses the paths to get station ID's lat, and lon.  
    
    INPUTS:
        
        files (list[str]): A list of station data paths from a directory. 
        st_id_map (panadas DF): sheet that maps station ID's to lat and lon. Found at ref_data/isd-history.csv
        inter_type (str): Interpolation type. Options availible are described in the ref_data README. 
        path (str): The veda-data-store path to the FWI files. 
    
    '''
    print("Searching for availible stations at" + path_s3)
    file_inter = []
    for path in files:
        if inter_type in path:
            file_inter.append(path)

    df = []
    for i in file_inter:
        pt_1 = re.sub(path_s3, "", i)
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
    '''
    Appends a leading 0 to hours for date formatting 
    
    INPUTS:
        
        hr (str): An hour digit from 0-24
    
    '''
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
    '''
    Takes a data frame with the columns "YYYY", "MM", "DD", and "HH" and converts those columns into a single datetime column. 
    
    INPUTS:
        
        dat_time (DataFrame):  A data frame with the columns "YYYY", "MM", "DD", and "HH". Hours do not have leading zeros. 
    
    '''
    dat_time.HH = dat_time.HH.astype('int') # Drop #.0
    dat_time.HH = dat_time.HH.astype('str')
    dat_time['HH_format'] = dat_time.HH.apply(hour_fix)
    dat_time['time'] = pd.to_datetime(dat_time['YYYY'].astype(str) +"-"+  dat_time['MM'].astype("str") +"-"+  dat_time['DD'].astype("str") + " " + dat_time['HH_format'].astype('str'), format='%Y-%m-%d %H')

    return(dat_time)

def get_st(lat, lon, stations, flag_bad = True):
    '''
    Read in data from a station at a lat long. Optionally, set data where interpolation may be too far from data as NaN. 
    
    INPUTS:
        
        lat (str):  A  lattitude
        lon (str):  A longitude
        stations (DataFrame): a dataframe as outputted by st_avail. A dataframe with columns for station lat, lon, and ID. 
        flag_bad (bool): Filter out data where the difference in the observation and the interpolation is over 20. This could indicate that the interpolation is way off. Default to True. 
    
    '''
        
    st = stations.loc[(stations.Lat == lat) & (stations.Lon == lon)]
    #dat = pd.read_csv("s3://veda-data-store-staging/EIS/other/station-FWI/20000101.20220907.hrlyInterp/FWI/727970-94240.spline.DailyFWIfromHourlyInterp.csv")
    dat = pd.read_csv(("s3://" + st.File_path.iloc[0]), index_col = False)
    dat = date_convert(dat)
    
    if flag_bad:
        mask = dat['OBSMINUTEDIFFTEMP'].loc[dat.OBSMINUTEDIFFTEMP > 20]  ## Kluge. Basically, use this as a flag. 
        dat.iloc[mask.index, 4:-4 ] = np.nan
    
    return(dat)

def plot_st(lat, lon, stations, times = "none_passed", flag_bad = True, time_start = "default", time_end = "default"):
    '''
    Plots station variables for a given station. 
    
    INPUTS:
        
        lat (str):  A  lattitude
        lon (str):  A longitude
        stations (DataFrame): a dataframe as outputted by st_avail. A dataframe with columns for station lat, lon, and ID. 
        flag_bad (bool): Filter out data where the difference in the observation and the interpolation is over 20. This could indicate that the interpolation is way off. Default to True. 
        times (dateTime): A sereis of dateTimes, the time axis. Default to "none_passed", where function will generate one based on assumptions about when the record starts/ stops.
        time_start (dateTime): For subseting times in the plot to focus on. Mush either be default or changed with time_end. 
        time_end (dateTime): For subseting times in the plot to focus on. Mush either be default or changed with time_start.
    
    '''
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
    '''
    Finds distance in km between two points designated with lat and lon values 
    
    INPUTS:
        
        lat1 (str):  A  lattitude
        lon1 (str):  A longitude
        lat2 (str):  second lattitude
        lon2 (str):  second longitude 
    
    '''
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav)) ## Returns in km

def closest(data, v):
    '''
    Will return the shortests distance between one lat/lon point and a dictionary of several lat/lon points. 
    
    INPUTS:
        
        data (dict):  A  dictionary with "Lat" and "Lon" entries to compare to v
        v (dict):  A dictinary with a single "Lat" and "Lon" entry. 
    
    '''
    mn = min(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
    dist = distance(v['Lat'],v['Lon'], mn['Lat'], mn['Lon'])
    #this_station = data[(data['Lat'] == mn['Lat']) & (data['Lon'] == mn['Lon'])]
    #print("The closest station is", this_station["STATION NAME"]," USAF: ", this_station["USAF"],"located at Lat", mn['Lat']," and Lon",mn['Lon'],  "and is ",  dist, "km away." )
    print("The closest station is",  dist, "km away." )
    return mn

def closest_srch(data, v):
    '''
    Will return the shortests distance between one lat/lon point and a dictionary of several lat/lon points. Will return a DataFrame of the closest point with columns "Lat", "Lon", and "dist_km". 
    
    INPUTS:
        
        data (dict):  A  dictionary with "Lat" and "Lon" entries to compare to v
        v (dict):  A dictinary with a single "Lat" and "Lon" entry. 
    
    '''
    mn = min(data, key=lambda p: distance(v['Lat'],v['Lon'],p['Lat'],p['Lon']))
    dist = distance(v['Lat'],v['Lon'], mn['Lat'], mn['Lon'])
    mn = pd.DataFrame([mn])
    print("The closest station is", dist, "km away." )
    mn['dist_km'] = dist
    return mn

def station_lookup(Lat, Lon, stations = stations, id_key = st_id_map):
    station = stations
    #this_station = station[(round(station.Lat, 3) == Lat) & (round(station.Lon,3) == Lon)] ## Note_ rounding assumptions need to change if not coming from print statement
    this_station = station[(station.Lat) == Lat) & (station.Lon) == Lon)] 
    this_station_more_info = id_key.loc[id_key.USAF.astype(str) == this_station.USAF.iloc[0]]
    print("The closest station is ", this_station_more_info["STATION NAME"], " USAF: ", this_station["USAF"], "Located at ", Lat, ":", Lon)
def stations_dist(st_dict, fire_center):
    '''
    Will return the distances between one lat/lon point and a dictionary of several lat/lon points. 
    
    INPUTS:
        
        st_dict (dict):  A  dictionary with "Lat" and "Lon" entries to compare to fire_center
        fire_center (dict):  A dictinary with a single "Lat" and "Lon" entry. 
    
    '''
  
    
    distances = pd.Series(map(lambda p: distance(fire_center['Lat'],fire_center['Lon'],p['Lat'],p['Lon']), st_dict))
    st_df = pd.DataFrame.from_dict(st_dict)
    colname = "Dist_to_fire"
    st_df[colname] = distances
    st_df = st_df.sort_values(by = colname)
    

    return st_df

def all_stations_search(st_dict, fire_center, id_key = st_id_map, max_dist = np.nan): 
        '''
    Returns all stations, with descriptors, within a certain distance from a point
    
    INPUTS:
        
        st_dict (dict):  A  dictionary with "Lat" and "Lon" entries to compare to fire_center
        fire_center (dict):  A dictinary with a single "Lat" and "Lon" entry. 
        id_key (panadas DF): sheet that maps station ID's to lat and lon. Found at ref_data/isd-history.csv
        max_dist (float): Maximum distance in km from fire_center. Default of NaN will return the distance of each station in network. 
    
    '''
    
    st_df = stations_dist(st_dict, fire_center)

    id_key = id_key.rename(columns={"LAT": "Lat", "LON": "Lon"})
    names = pd.merge(st_df, id_key, on = ["Lat", "Lon"])
    
    if(not np.isnan(max_dist)):
        names = names[names.Dist_to_fire <= max_dist]
    
        
        
    return(names)

### Loading Fire files
def load_file(date,layer='perimeter',handle_multi=False,
              only_lf=False,area_lim=5,show_progress=False, year = "2019", path_region = "WesternUS"):
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
        year (str): Year that fires took place. Default to 2019. Availible options differ by path_region. 
        path_region (str): This constructs the path that the fires are stored in. WesternUS and CONUS availible. 
    
       This function was authored by Eli, and modified by Tess.  
    
    '''
    
    base_path = '/projects/shared-buckets/gsfc_landslides/FEDSoutput-s3-conus/' +path_region+ '/'+ year +'/Snapshot/'
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

def prep_gdf(date = '20191031PM',layer='perimeter', handle_multi=True,only_lf=True,area_lim=5, year = "2019", path_region = "WesternUS"):
    '''
    loads in snapshot file based on input date and layer, then preps it for "explore" by adding centriod data and converting datTime files to strings. 
    
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
        year (str): Year that fires took place. Default to 2019. Availible options differ by path_region. 
        path_region (str): This constructs the path that the fires are stored in. WesternUS and CONUS availible. 
    '''
    gdf = load_file( date, layer,  handle_multi,only_lf,area_lim, year, path_region)
    gdf.set_index('fireID',inplace=True)

### Get explore map to display lat and lon 
    gdf_test = gdf ## Default crs seems to be easting and westing in just US. reproject to lat vs lon

    gdf_test = gdf_test.to_crs('EPSG:4326')
    gdf_test['lon'] = gdf_test.centroid.x
    gdf_test['lat'] = gdf_test.centroid.y

    gdf_test["t"] = gdf_test["t"].astype("str")
    gdf_test["t_st"] = gdf_test["t_st"].astype("str")
    gdf_test["t_ed"] = gdf_test["t_ed"].astype("str")

    #gdf = load_file(date = '20191031PM',handle_multi=True,only_lf=True,area_lim=5)
    #gdf.set_index(index, inplace)
    #print("Total number of fires:",len(gdf.index.unique()))
    #print("These were the files availible at " + date + " and larger than", area_lim, " found at CONUS")
    
    #gdf_test = gdf.to_crs('EPSG:4326')
    #print("Setting projection to EPSG:4326")
    #gdf_test['lon'] = gdf.centroid.x
    #gdf_test['lat'] = gdf.centroid.y

    
    
    #print("Setting fireID as index")
    
    #gdf_test["t"] = gdf_test["t"].astype("str")
    #gdf_test["t_st"] = gdf_test["t_st"].astype("str")
    #gdf_test["t_ed"] = gdf_test["t_ed"].astype("str")
    
    return(gdf_test)
    

def fire_search(gdf, stations, dist_max_km = 112.654): # ~ 70 miles distance
    '''
    Will look for the closest station for a geoDataFrame of fires, and will return a dataframe with the columns "Lat", "Lon", "dist_km","Origin_lat", "Origin_lon",  and "fireID". "Origin_*" refers to the fires's centriod lat and lon. "Lat"/ "Lon" are the station' lat and lon. 
    
    INPUTS:
        
        gdf (GeoDataFrame):  A snapshot file.
        stations (DataFrame): a dataframe as outputted by st_avail. A dataframe with columns for station lat, lon, and ID. 
        dist_max_km (float): Maximum distance in km. 
    '''
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


def load_large_fire(fireID, year = "2019", path_region = "WesternUS"):
    '''
    loads in largefire file based on fireID and layer, then preps it for "explore" by adding centriod data. Currently limited to one year. 
    
    INPUTS:
        
        fireID (str): fireID offire of interest. Can be found in gdf files read in by prep_gdf and load_file. Can be selected interactivly form a gdf if use gdf.explore()
        year (str): Year that fires took place. Default to 2019. Availible options differ by path_region. 
        path_region (str): This constructs the path that the fires are stored in. WesternUS and CONUS availible. 
    '''
    lf_files = glob.glob('/projects/shared-buckets/gsfc_landslides/FEDSoutput-s3-conus/' + path_region +'/'+ year +'/Largefire/*' + fireID + '*') 
    lf_ids = list(set([file.split('Largefire/')[1].split('_')[0] for file in lf_files])) 
    largefire_dict = dict.fromkeys(lf_ids)
    
    for lf_id in lf_ids:
         most_recent_file = [file for file in lf_files if lf_id in file][-1]
         largefire_dict[lf_id] = most_recent_file
    
    gdf = pd.concat([gpd.read_file(file,layer='perimeter') for key, file in largefire_dict.items()], 
                   ignore_index=True)
    gdf = gdf.to_crs('EPSG:4326')
    gdf['lon'] = gdf.centroid.x
    gdf['lat'] = gdf.centroid.y
    return gdf

def fr_st_merge(gdf, dat, sub= True, sub_type = "exact", num_months = 1, custom_date = "NA"):
    '''
    INPUTS:
        gdf (GeoDataFrame): A largefile GeoDataFrame as read in by load_large_fire.
        dat (DataFrame): Station data, as read in by get_st
        sub (bool): Will subset the station data to dates overlapping the time when the fire is active (according to Yang Chen's algorithm, ~ 5 days past last burn)
        sub_type (str): "exact", "month_before", Subset station data to include leadup to fire
        num_months (int): If sub_type = "month_before", # months before start of fire. 
    '''
    
    if(sub):
        if(sub_type == "exact"):
            st_dat = dat[(dat.time >= min(gdf.t)) &  (dat.time <= max(gdf.t))]
        if(sub_type == "month_before"):
            start_date =  min(gdf.t)
            month_early = start_date - np.timedelta64(num_months, 'M')

            st_dat = dat[(dat.time >= month_early) &  (dat.time <= max(gdf.t))]
        if(sub_type == "custom"):
            start_date =  min(gdf.t)
            print("Subsetting to ", custom_date)
            st_dat = dat[(dat.time >= custom_date) &  (dat.time <= max(gdf.t))]
        
    else:
        st_dat = dat
   
    st_dat = st_dat.rename(columns = {"time":"t"}) 
    
    ## Put both into datetimes
    st_dat['t'] = st_dat['t'].astype('datetime64[ns]')
    gdf['t'] = gdf['t'].astype('datetime64[ns]')
    full = pd.merge(gdf,st_dat, on = "t", how = "outer")
    #full = full.sort_values(by = ['t']) ## Need to sort or timeseries jumps around
    #full['t'] = full['t'].astype('datetime64[ns]') 
    return(full)


def merge_with_fire(gdf_beachie, fireID = "NA", fire_name = "NA", foi_custom = np.nan ):
    '''
    Will take a large_fire object and will merge it with the closest weather station. 
    
    INPUTS:
        gdf_beachie (GeoDataFrame): A largefile GeoDataFrame as read in by load_large_fire.
        fireID (str): ID of fire for labeling. Optional. 
        fire_name (str): Name of fire for labeling. Optional. 
        foi_custom (dict): A dictionary object with "Lat" and "Lon". This function's default behavior is to find a wether station closest to the center of the fire. foi_custom allows a user to pass a differnt point. 
        
    '''
    ## Clean some columns
    gdf_beachie["fireID"] = fireID
    gdf_beachie["Fire_Name"] = fire_name
    gdf_beachie["timediff"]  = gdf_beachie.t.astype("datetime64[ns]") - pd.to_datetime(min(gdf_beachie.t))
    gdf_beachie['timediff'] = gdf_beachie.timediff.astype("int")
    
    ## Get Fire Lat Lon
    if(np.isnan(foi_custom)):
        foi = gdf_beachie
        foi = foi.rename(columns = {"lat": "Lat", "lon": "Lon"})
        foi = foi.iloc[0] # First element
        print(foi["Lat"])
        print(foi["Lon"])
    else:
        foi = foi_custom
        print(foi["Lat"])
        print(foi["Lon"])
        
    ## Look for closest station
    st_cls = fv.closest(st_dict, foi)
    closest(st_dict,foi)
    
    ## Get station Data
    st = fv.get_st(lat = st_cls["Lat"], lon = st_cls["Lon"], stations = stations)
    
    ## Merge with fire data
    gdf_beachie["t"] = gdf_beachie["t"].astype("datetime64[ns]")
    full_fr = fr_st_merge(gdf_beachie, st, sub = True, sub_type = "custom", custom_date = "2020-08-08", end_date = "2020-10-08")
    full_fr = full_fr.sort_values( by = "t")
    
    return(full_fr)

#def fire_quickplot(fireID):
#    fr = load_large_fire(fireID)
    