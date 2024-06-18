These csvs are to tie lightning strikes to fires events in Canada 2023. They are a compilation of a few different datasets. 

1) FEDS dataset. Paper here https://doi-org.proxy-um.researchport.umd.edu/10.6084/m9.figshare.c.5601537.v1 . This was run over the Boreal region. https://github.com/Earth-Information-System/fireatlas


2) The Earth Networks ground sensor network.  ICD for this product can be found publicly accessible at https://get.earthnetworks.com/hubfs/Customer%20Success/ENTLN%20Lightning%20Data%20Feed%20v4%20ICD.pdf

3) The CIFFC data for the season. CIFFC releases updates on fires with a centroid, a start time, and a designation of human-casued or natural causes. 



Column Names and Definions: 

'fireID' -- ID of Fire Event. 
't' -- Time stamp
'geometry' -- perimeter of fire event @ t
'InterCloud' -- Is strike InterCloud? (Only 0 should be present)
'lt_lat' -- Latitude of lighting strike
'lt_lon' -- Longitude of lightning strike
'current_mag' -- lightning strike current, see ICD
'error_elps' -- lightning strike error ellipse, see ICD
'num_station' -- lightning strike number of stations, see ICD
'num_candidate_ig_strike' -- Number of strikes that are likely candidates for fire ignition. Within 20 days of fire start, and within 1500m^2 of first perimeter. 
'num_strikes_at_t_of_candidates' -- Number of strikes in total region in the time period when candidates were found. If two candidates were found at t_4 and t_10, this number will be the number of total stikes in the whole region (ex Quebec) from t_4 through t_10. 
'num_strikes_in_region_during_window' -- Number of strikes in total region in the 20 days before the start of a fire. 
'n_pixels' -- Cumulative number of pixels that made up fire until t, see FEDS paper. 
'n_newpixels' -- number of pixels that made up perimeter @ t, see FEDS paper.
'farea' -- Fire area km^2, see FEDS paper.
'fperim' -- Length of fire perimeters km, see FEDS paper.
'flinelen' -- Length of active fire line, subset of fire perimeter km, see FEDS paper.
'fire_duration' -- Number of days since first observation of fire. Fires with a single observation have a duration of zero, see FEDS paper.
'pixden' -- Number of pixels divided by area of perimeter
'meanFRP' -- Mean fire radiative power. The weighted sum of the fire radiative power detected at each new pixel, divided by the number of pixels. If no new pixels are detected, meanfrp is set to zero.
'fr_lon_centroid' -- y-th coordinate of fire perimeter centroid. 
'fr_lat_centroid' -- x-th coordinate of fire perimeter centroid.
'field_situation_report_date' -- CIFFC report date. 
'field_system_fire_cause' -- CIFFC Fire cause. 
