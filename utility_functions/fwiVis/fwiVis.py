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
import datetime
from datetime import timedelta
import requests
from bs4 import BeautifulSoup


def ca_prov():
    '''
    Read in Canadian territory/ province vectors from 2021 https://data.opendatasoft.com/explore/dataset/georef-canada-province%40public/export/?disjunctive.prov_name_en&sort=prov_name_en. 
    
    No inputs. 
    '''
    tmp = gpd.read_file(f"{os.path.abspath('contextual_data')}/Canadian_prov/georef-canada-province@public.geojson")
    tmp_names = gpd.read_file(f"{os.path.abspath('contextual_data')}/Canadian_prov/georef-canada-province@public.csv")

    #neon.DomainID
    tmp_names = tmp_names[['Official Name Province / Territory (English)', 'Official Name Province / Territory (French)']]
    tmp_names = tmp_names.rename(columns={'Official Name Province / Territory (English)':"prov_name_en", 
                                         'Official Name Province / Territory (French)': 'prov_name_fr' })

    tmp = tmp.merge(tmp_names, on = "prov_name_fr")
    #print(tmp)
    
    if(any(tmp.columns.isin(["prov_name_en_y"]))):
        tmp = tmp.rename(columns = {"prov_name_en_y" : "prov_name_en"})
    tmp = tmp[['prov_name_fr', 'prov_name_en', 'geometry']]
    return(tmp)


def gpd_read_file(filename, parquet=False, **kwargs):
    itry = 0
    maxtries = 5
    fun = gpd.read_parquet if parquet else gpd.read_file
    while itry < maxtries:
        try:
            dat = fun(filename, **kwargs)
            return dat
        except Exception as e:
            itry += 1
            print(f"Attempt {itry}/{maxtries} failed.")
            if not itry < maxtries:
                raise e

### Functions that take largefire perimeters that are different overlapping fire and merge them into one. 
def prep_fire_files(path, crs = "3571"):
    '''
    Read in cvs of previously output GeoDataFrame as a GeoDataFrame with valid geometry column. 
    '''
    fires = pd.read_csv(path)
    fires = fires.rename(columns={"geometry":"csv_geometry"})
    fires.t = fires.t.astype("str")
    fires.fireID  = fires.fireID.astype("str")
    #fires['csv_geometry'] =fires['csv_geometry'].apply(wkt.loads)
    fires_geom = gpd.read_file(path, GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")
    fires_geom.t = fires_geom.t.astype("str")
    fires_geom.fireID  = fires_geom.fireID.astype("str")
    fires = fires_geom[["fireID", "t", "geometry"]].merge(fires, on=["fireID", "t"], how = "left")
    fires = fires.set_crs(crs)
    return (fires)

def listFD(url, ext=''):
    '''
    NCCS retrival helper functions for getting gridded data
    '''
    page = requests.get(url).text
    #print(page)
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


def get_nccs_url(pattern, url = 'https://portal.nccs.nasa.gov/datashare/GlobalFWI/ForecastFWIEXPERIMENTAL/QuebecAllFires.Radius.25.km.247.biggestFires/GEOS-5/GEOS-5.IMERGEARLY/chicletDataNoSmoothing/', ext = 'csv'):    
    '''
    Function to search files at NCCS url. Searcehd for specific pattern in files listed at a path. Is used specifically for checing the files availible for specific fireIDs. 
    
    INPUTS:
    
        pattern (str) string in the file name, ie a fireID. 
        url (str) nccs url with multiple files.  
    '''
    file_list = []
    for file in listFD(url, ext):
        file_list.append(file)

    try_pd = pd.DataFrame(file_list, columns= ["urls"])
    size = try_pd[try_pd.urls.str.contains(pattern)].urls.values.size
    if(size == 0):
        print("No matches found to pattern. Returning None.")
        return(None)
    if(size >= 2):
        print("Multiple matches found:")
        print(try_pd[try_pd.urls.str.contains(pattern)].urls.values)
        raise ValueError()
    url = try_pd[try_pd.urls.str.contains(pattern)].urls.values[0]
    return(url)

def get_gridded_fwi(fireID):
    '''
    Function to extract FWI values from nccs. Returns a DF of FWI and FWI with a lead time of 8 days. 
    
    INPUTS:
    
        fireID (str) Id of fire.   
    '''
    # Get the URL for the file
    fireID = str(fireID)
    pattern = "FWI." + fireID
    url = get_nccs_url(pattern = pattern)
    
    if(url is not None):
        
        # Get the DF
        grid_FWI = pd.read_csv(url)
        # Change names
        grid_FWI = grid_FWI.rename(columns={'INITDATE': 't', 
                                 "0":"FWI",
                                 "1":"FWI_lead_1",
                                 "2":"FWI_lead_2",
                                 "3":"FWI_lead_3",
                                 "4":"FWI_lead_4",
                                 "5":"FWI_lead_5",
                                 "6":"FWI_lead_6",
                                 "7":"FWI_lead_7",
                                 "8":"FWI_lead_8"
                                })
        # Change dates
        grid_FWI.t = grid_FWI.t.astype("datetime64[ns]").dt.strftime('%Y-%m-%d 12:00:00')

        # return
        return(grid_FWI)
    else:
        return(None)

def chop_fires_at_end(df):
    '''
    Function to extract FWI values from nccs. Returns a DF of FWI and FWI with a lead time of 8 days. 
    
    INPUTS:
    
        df (DataFrame) Single-fireID df that will trim excess observations after the final fire growth increment. TO be used with groupby fireID.    
    '''
    if(len(df[df.n_newpixels > 0]) > 0 ): 
        final_t = df[df.n_newpixels > 0].t.max()
        df = df[df.t <= final_t]
    else:
        df = df[df.t == df.t.min()] ## If the fire had it's only growth increment during the 12 hour period we aren't looking at (ie spread in the AM overpass)
    return(df)
    
def just_the_igs(df,): ## Reminder that this will only acocunt for the earliest ignition from a multi-fire complex. 

    df["is_ig"] = False
    df.loc[df.t == df.t.min(), ['is_ig']] = True
    
    return(df)

def extinction(df, days_past = 1):
    df.loc[:, ['is_ext']] = False
    df.loc[:, ["t"]] = df.t.astype("datetime64[ns]")
    final_t = df[df.n_newpixels > 0].t.astype("datetime64[ns]").max()
    post_fire = final_t + timedelta(days = days_past)

    if (pd.isnull(np.datetime64(str(final_t)))):
        df.loc[(df.t == df.t.min()), ['is_ext']] = True
    else:
        df.loc[(df.t  > final_t)  & (df.t  <= post_fire), ['is_ext']] = True
    #print(final_t)
    #print(df.fireID.unique())
    #print(len(df[df["is_ext"] == True]))
    return(df)

def fix_the_first_spread_day(df, col = "spread_bool_100", thresh = 1):
    val = df.loc[df.is_ig].farea.iloc[0]
    if(val>= 1):
        df.loc[df.is_ig, [col]] = 1
    else:
        df.loc[df.is_ig, [col]] = 0
    return(df)

def formatting_read_in(path, start_time = '2023-05-31 00:00:00', end_time = '2023-09-15 12:00:00', gen_cols = True, merge_with_ciffc = False, merge_with_nbac = True): ## Does some of the read-in functions
    ### Read in
    path = os.path.abspath(path)
    fire3 = prep_fire_files(path)

    ## Cut out end of fires
    fire3 = fire3.sort_values(by = ["fireID", "t"])
    fire3 = fire3.groupby("fireID").apply(chop_fires_at_end).reset_index(drop = True)
    fire3 = fire3.drop_duplicates()

    ## only keep noon overpass times
    fire3 = fire3[~fire3.FWI.isna()]

    ## subset by start and end time
    fire3 = fire3[fire3.t >= start_time]
    fire3 = fire3[fire3.t <= end_time]

    ## exclude small fires under 4 km^2
    # ids = fire3.groupby("fireID").farea.max().reset_index()
    # small_fires = ids[ids.farea < 4]
    # fire3 = fire3[~fire3.fireID.isin(small_fires)]

    ## Subset to the province of quebec
    prov = ca_prov()
    prov = prov[prov.prov_name_en == "Quebec"]
    prov = prov.to_crs('EPSG:3571')
    fire3 = fire3.to_crs('EPSG:3571')
    fire3 = fire3.sjoin(prov)
    fire3 = fire3.drop(['prov_name_fr', 'prov_name_en', 'index_right'], axis = 1)
    fire3 = fire3[~fire3.geometry.isna()]

    if(merge_with_nbac != gen_cols):
        print("WARNING: Intersecting with CIFFC will subset fires, but will make the generation of some column variables different after the fact. It is recommended to generate variable columns BEFORE CIFFC spatial joins")

    if(gen_cols):
        fire3["farea_diff_stand"] = fire3.groupby("fireID").farea.diff()
        fire3 = fire3.groupby("fireID").apply(extinction).reset_index(drop = True)
        fire3 = fire3.groupby("fireID").apply(just_the_igs).reset_index(drop = True)
        
        fire3["corrected_flinelen"] = fire3.flinelen + 0.001
        ## Fireline 
        fire3["frac_change"] = fire3.groupby("fireID").farea.pct_change()
        fire3["spread_bool"] = fire3.frac_change > 0
        fire3["spread_bool"] = fire3["spread_bool"].astype("int")
        fire3["spread_bool_frac_05"] = fire3.frac_change > 0.05
        fire3["spread_bool_frac_05"] = fire3["spread_bool_frac_05"].astype("int")
        fire3["spread_bool_100"] = fire3.farea_diff_stand > 1
        fire3["spread_bool_100"] = fire3["spread_bool_100"].astype("int")
        fire3.loc[:, "GEOS5_IMERGEARLY"] = fire3['GEOS-5.IMERGEARLY']
        fire3 = fire3.groupby('fireID').apply(fix_the_first_spread_day).reset_index(drop = True)




    ## subset to the ciffc data
    if(merge_with_ciffc):
        print(" WARNING No longer merging with CIFFC due to exclusions of larger fires")
        num_fires_prev = len(fire3.fireID.unique())
        ciffc = pd.read_csv(os.path.abspath("contextual_data/CIFFC_data/ciffc_all_canada.csv"))
        ciffc = ciffc[ciffc.field_agency_code == "qc"]
        ciffc = gpd.GeoDataFrame(ciffc, geometry= gpd.points_from_xy(ciffc.field_longitude, ciffc.field_latitude), crs = "4326")
        ciffc = ciffc.to_crs("3571")
        ciffc["geometry_point"] = ciffc.geometry
        fire3 = fire3.sjoin(ciffc)
        num_fires_now = len(fire3.fireID.unique())
        print(f"fire3 df had {num_fires_prev} fires p, and { num_fires_now} post CIFFC ignition join")

    if(merge_with_nbac):
        nbac_only = gpd.read_file(f"{os.path.abspath('contextual_data')}/NBAC/nbac_2023_20240530.shp")
        nbac_only = nbac_only[nbac_only.ADMIN_AREA == 'QC']
        nbac_only  = nbac_only [(nbac_only.ADJ_HA >= 400) |(nbac_only.POLY_HA >= 400)]
        nbac_only = nbac_only[(nbac_only.HS_SDATE <= "2023-09-15" ) |  (nbac_only.AG_SDATE <= "2023-09-15")]
        fire3 = fire3.to_crs(nbac_only.crs)
        fire3_tmp = fire3.sjoin(nbac_only)
        fire3 = fire3[fire3.fireID.isin(fire3_tmp.fireID.unique())] ### Doing this instead of a normal sjoin becuase a fire lost a time-step because one time-step did not intersect with the nbac fire.
       #  fire3 = fire3.drop(['YEAR', 'NFIREID', 'BASRC', 'FIREMAPS', 'FIREMAPM', 'FIRECAUS',
       # 'HS_SDATE', 'HS_EDATE', 'AG_SDATE', 'AG_EDATE', 'CAPDATE', 'POLY_HA',
       # 'ADJ_HA', 'ADJ_FLAG', 'ADMIN_AREA', 'NATPARK', 'PRESCRIBED', 'VERSION',
       # 'GID'], axis=1)
        fire3 = fire3.to_crs("4326")
    

    ### Final sorting 
    fire3.t = fire3.t.astype("datetime64[ns]")
    fire3 = fire3.sort_values(by = ["fireID", "t"])
    
    return(fire3)



