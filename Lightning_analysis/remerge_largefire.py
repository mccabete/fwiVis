def remerge_largefire(fires):
    '''
    If two final feds perimeters intersect spatialy, check if one ended before the other began. If yes, give it the ID of the earlier perimeter. 
    Note: Could optionally be spitting out time differences, or sorting by them. For Quebec, was a max of 22 days, min of 1 day. Theoretically, not sure why the 1 day perimeter wasn't merged by feds. 
    '''
    # Get an id, first/ last t per ID, and a geometry
    first_perims = fires[~fires.geometry.isnull()].groupby("fireID").t.min().reset_index()
    last_perims = fires[~fires.geometry.isnull()].groupby("fireID").t.max().reset_index()
    plot_last = fires.merge(last_perims, on = ["fireID", "t"], how = 'right')
    plot_first = fires.merge(first_perims, on = ["fireID", "t"], how = 'right')
    plot_last = plot_last[["fireID", "t", "geometry"]]
    plot_first = plot_first[["fireID", "t", "geometry"]]
                    
    # Check what perimeters spatially intersect into anouther one through time
    last_in_last = plot_last.sjoin(plot_last, how = "left", predicate = "intersects")
    lil = last_in_last.groupby(["fireID_left",]).t_right.min().reset_index()
    id_with_max_time = lil.merge(last_in_last[["t_right", "t_left", "geometry", "fireID_right", "fireID_left"]], on = ["t_right", "fireID_left"], how = "right")

    id_with_max_time = id_with_max_time.rename(columns={"fireID_right": "mergeID", 
                                     "fireID_left" :"fireID",
                                     "t_right":"mergeID_t",
                                     "t_left":"fireID_end_t"
    })

    id_with_max_time_check = id_with_max_time[id_with_max_time.fireID_end_t < id_with_max_time.mergeID_t]

    ### FireID the earlier perimeter that later perimeters are merged into. "mergeID" describes the merge-ey
    fireID_with_merge = id_with_max_time_check.groupby(["fireID"]).mergeID.unique().reset_index() 

    mergeID_with_fireID  = id_with_max_time_check.groupby(["mergeID"]).fireID.unique().reset_index()
                    
    # Check when the fireID and mergeID started/stopped. 
    get_merge_start = plot_first[["fireID", "t"]].rename(columns={"fireID":"mergeID", 
                                                           "t":"mergeID_start_t"})
    get_fireID_start =  plot_first[["fireID", "t"]].rename(columns={ "t":"fireID_start_t"})

    id_map = id_with_max_time_check.merge(get_merge_start, on = ["mergeID"])
    id_map = id_map.merge(get_fireID_start, on = ["fireID"])
    id_map = id_map[["fireID","fireID_start_t",  "fireID_end_t", "mergeID", "mergeID_start_t", "mergeID_t"]]
    id_map["time_diff_fireIDend_mergeIDstart"] = id_map.fireID_end_t.astype('datetime64[ns]') - id_map.mergeID_start_t.astype('datetime64[ns]') ## Negative means that mergeID started after fireID ended
    
    # Subset to IDs where one fire "ended" before the next fire began                
    only_IDs_with_negative_dates = id_map[id_map.time_diff_fireIDend_mergeIDstart.dt.days < 0]
    
    # Go through an reindex just the IDs that overlap in space but not time                
    fires["old_id"] = ""
    fire_ids = only_IDs_with_negative_dates.mergeID.unique() ## Gives the IDs of fires to be merged into anouther fire

    for i in fire_ids:
        min_t = id_with_max_time_check[id_with_max_time_check.mergeID == i].fireID_end_t.min()
        sm_id_map = id_with_max_time_check[(id_with_max_time_check.mergeID == i) & (id_with_max_time_check.fireID_end_t == min_t)]
        if(len(sm_id_map) == 0):
            print("ID", i, " only merges with self")
            fires["old_id"][fires.fireID == i] = i
        else:
            if(len(sm_id_map) != 1):
                print("There are two perimeters that intersect with ID",i, " that started at the same time.")
                print(sm_id_map)
                break

            fires["old_id"][fires.fireID == i] = i
            fires["fireID"][fires.fireID == i] = str(*sm_id_map.fireID.values)
                    
    return(fires)