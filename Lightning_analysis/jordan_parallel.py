import sys
import pandas as pd
import geopandas as gpd
from glob import glob
import time
from multiprocessing import Pool, cpu_count


csvs_regex = '/projects/2023_lightning_data/*.raw'

def read_gdf(filename):
    af_df = pd.read_csv(filename, names =["InterCloud",
                                                                                             "t", 
                                                                                             "lat", 
                                                                                             "lon",
                                                                                             "current_mag", 
                                                                                             "multiplicity_0", 
                                                                                             "accr", 
                                                                                             "error_elps", 
                                                                                             "num_station"])
    # convert to GeoDataFrame

    af_gdf = gpd.GeoDataFrame(
        af_df,
        geometry=gpd.points_from_xy(af_df.lon, af_df.lat),
        crs="EPSG:4326"
    )
    #gdf = gpd.read_file(filename, crs='epsg:4326')
    print(filename)
    return af_gdf


def main():

    csvs_list = glob(csvs_regex)
    print(len(csvs_list))
    start = time.time()
    print(start)

    with Pool(processes=(cpu_count()) - 20) as p:
        result = p.map(read_gdf, csvs_list)
    gdf = pd.concat(result)
    print(gdf.shape)
    gdf.to_file("/projects/old_shared/fire_weather_vis/Lightning_analysis/test_para/2023.gpkg")
    end = time.time()

    #Subtract Start Time from The End Time
    total_time = end - start
    print("\n"+ str(total_time))
    
if __name__ == "__main__":
    sys.exit(main())