#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:43:13 2020

@author: matthew
"""
#%%

def check_def_visible(ph_def, mask_def, ph_topo, ph_turb, threshold = 0.1):
    """A function to check if a (synthetic) deformation pattern is still visible
    over synthetic topo correlated and turbulent atmospheres.  
    
    Inputs:
        ph_def | r2 array | deformation phase
        mask_def | ? | maks showing where the deformation is - 1s where deforming, I think?)
        ph_topo | r2 array | topo correlated APS
        ph_turb | r2 array | turbulent APS
        threshold | float | sets the level at which deformation is not considered visible over topo and turb APS
                            bigger = more likely to accept, smaller = less (if 0, will never accept)
        
    Returns:
        
    History:
        2019/MM/DD | MEG | Written as part of f01_real_image_dem_data_vXX.py
        2019/11/06 | MEG | Extracted from sctipt and placed in synth_ts.py
    
    """
    import numpy as np
    import numpy.ma as ma  
    
    ph_def = ma.array(ph_def, mask = (1-mask_def))
    ph_atm =  ma.array((ph_turb + ph_topo), mask = (1-mask_def))
    
    var_def = ma.var(ph_def)                        # calculate variance of the deforming part of ph_def
    var_combined = ma.var(ph_def + ph_atm)          # calcualte variance of deforming area when atmosphere is added
    
    if (1+threshold) * var_def > var_combined:
        viable = True
    else:
        viable = False
    return viable



#%%

def def_and_dem_translate(dem_large, defo_source, mask_coh, threshold = 0.3, n_pixs=224, defo_fraction = 0.8):
    """  A function to take a dem, defo source and coherence mask, randomly traslate the defo and dem, 
    then put together and check if the deformation is still visible.  
    Inputs:
        dem_large | rank 2 masked array | height x width , the dem
        defo_source | rank 2 array | height x width
        n_pixs | int | output size in pixels.  Will be square
        threshold | decimal | e.g. if 0.2, and max abs deforamtion is 10, anything above 2 will be considered deformation.  
        defo_fraction | decimal | if this fraction of deformation is not in masked area, defo, coh mask, and water mask are deemed compatible
        
    Returns:
        ph_defo
        dem
        viable
        loc_list
        mask_coh_water
        mask_def
        
    History:
        2019/MM/DD | MEG | Written as part of f01_real_image_dem_data_vXX.py
        2019/11/06 | MEG | Extracted from sctipt and placed in synth_ts.py
        
                
        """
    
    import numpy as np
    import numpy.ma as ma
    from neural_network_functions import localise_data
    
    def random_crop_of_r2(r2_array, n_pixs=224, extend = False):
        """  Randomly select a subregion of a rank 2 array.  If the array is quite small, we can select areas
        of size n_pixs that go over the edge by padding the edge of the array first.  Works well with things like deformation 
        (which is centered in the r2_array so the near 0 edges can be interpolated), but poorly with things like a dem.  
        Inputs:
            r2 array | rank 2 array | e.g. a dem or deformation pattern of size 324x324
            n_pixs | int | side lenght of subregion.  E.g. 224 
            extend | boolean | if true, the padding described above is carried out.  
        
        2019/03/20 | update to also output the xy coords of where the centre of the old scene now is
        """
        ny_r2, nx_r2 = r2_array.shape
        if extend:
            r2_array = np.pad(r2_array, int(((2*n_pixs)-ny_r2)/2), mode = 'edge')                   # pad teh edges, so our crop can go off them a bit (helps to get signals near the middle out to the edge of the cropped region)
            x_start = np.random.randint(10,n_pixs)
            y_start = np.random.randint(10,n_pixs)                                                  # 10 ensures we keep some of the centre
            
        else:
            x_start = np.random.randint(0, (ny_r2 - n_pixs), 1)[0]                                #                   "                - x direction
            y_start = np.random.randint(0, (nx_r2 - n_pixs), 1)[0]                                # calcaulte a random crop of the dem - y direction
            
        r2_array_subregion = r2_array[y_start:(y_start+n_pixs), x_start:(x_start+n_pixs)]           # do the crop
        x_pos = np.ceil((r2_array.shape[1]/2) - x_start)                                                     # centre of def - offset
        y_pos = np.ceil((r2_array.shape[0]/2) - y_start)                                                     # 
        pos_xy = (int(x_pos), int(y_pos))                                                                       # ints as we're working with pixel numbers
        return r2_array_subregion, pos_xy
    
    # start the function
    #import ipdb; ipdb.set_trace()
    dem, _ = random_crop_of_r2(dem_large, n_pixs, extend = False)                          # random crop of the dem
    ph_defo, def_xy = random_crop_of_r2(defo_source, n_pixs, extend = True)                # random crop of the deformation
    mask_water = ma.getmask(dem).astype(int)                                               # convert the boolean mask to a more useful binary mask, 0 for visible

#    import pdb; pdb.set_trace()
    try:
        loc_list = localise_data(ph_defo, centre = def_xy)                                                        # xy coords of deformation for class label
        viable = True
    except:
        loc_list = []                                                                       # return empty so that function can still return variable
        viable = False
            
    # determine if the deformation is visible
    mask_def = np.where(np.abs(ph_defo) > (threshold * np.max(np.abs(ph_defo))), np.ones((n_pixs, n_pixs)), np.zeros((n_pixs, n_pixs)))     # anything above the threshold is masked - sort of looks like blotchy areas of incoherence            
    mask_coh_water = np.maximum(mask_water, mask_coh)                                                                                       # combine incoherent and water areas to one mask
    ratio_seen = 1 - ma.mean(ma.array(mask_coh_water, mask = 1-mask_def))                                                                   # create masked array of mask only visible where deforamtion is, then get mean of that area
    #ph_defo = ma.array(ph_defo, mask = mask_coh_water)                                                                                     # mask out where we won't have radar return
    if ratio_seen < defo_fraction:
        viable = False                                                      # update viable is now not viable (as the deformation is not visible)
    return ph_defo, dem, viable, loc_list, mask_coh_water, mask_def, mask_water


#%%

def create_random_defo_m(dem, dem_ll_extent, deformation_ll, defo_source_n,
                         min_deformation_size, max_deformation_size, asc_or_desc):
    """ Given a dem, create a random deformation pattern of acceptable magnitude.  
    Inputs:
        dem
        dem_ll_extent
        deformation_ll
        defo_source_n
        min_deformation_size
        max_deformation_size
        asc_or_dec | string | 'asc' or 'desc' or 'random'.  If set to 'random', 50% chance of each.  
    Returns:
        defo_m
    History:
        2020/08/12 | MEG | Written
    """
    import numpy as np
    import numpy.ma as ma
    
    from syinterferopy_functions import deformation_wrapper
    
    deformation_magnitude_acceptable = False    
    while deformation_magnitude_acceptable is False:
        # 0: Set the parameters randomly for the forward model of the source
        if defo_source_n == 1:
            source = 'dyke'
            source_kwargs = {'strike'    : np.random.randint(0, 359),
                             'top_depth' : 2000 * np.random.rand(),                     # units are metres.  
                             'length'    : 10000 * np.random.rand(),
                             'dip'       : np.random.randint(75, 90),
                             'opening'   : 0.1 + 0.6 * np.random.rand()}
            source_kwargs['bottom_depth'] = source_kwargs['top_depth'] + 6000 * np.random.rand()                # as depends on a value in the dict, has to be made separately.  
        elif defo_source_n == 2:
            source = 'sill'
            source_kwargs = {'strike'   : np.random.randint(0, 359),
                             'depth'    : 1500 + 2000 * np.random.rand(),
                             'width'    : 2000 + 4000 * np.random.rand(),
                             'length'   : 2000 + 4000 * np.random.rand(),
                             'dip'      : np.random.randint(0,5),
                             'opening'  : 0.5}
        elif defo_source_n == 3:
            source = 'mogi'
            source_kwargs = {'volume_change' : 1e6,                           
                           'depth'         : 2000}                                                 # both in metres
                        
        # 1:  Generate the deformation
        defo_m, _, _, _ = deformation_wrapper(dem, dem_ll_extent, deformation_ll, 
                                                     source, m_in_pix = 92.6, asc_or_desc = asc_or_desc,  **source_kwargs)
        # 2: Check that it is of acceptable size (ie not a tiny signal, and not a massive signal).  
        max_los_deformation = np.max(np.abs(ma.compressed(defo_m)))
        if (min_deformation_size < max_los_deformation) and (max_los_deformation < max_deformation_size):
            deformation_magnitude_acceptable = True
    
    return defo_m
            





#%%

def create_volcano_dems(volcano_csv_file, srtm_tiles_folder, void_fill = True, width_km = 20, water_mask_resolution = 'i', download = False):
    """ A function to create a DEM for each of the Smithsonians subaerial volcanoes.  
    
    Inputs:
        volcano_csv_file | string | path to csv file containing name and lat lon of each volcano in the Smithsonian database
        srtm_tiles_folder | string | path to folder where SRTM3 tiles are stored.  

        void_fill | boolean | if True, voids in the SRTM data are filled.  Very slow.  
        width | int | DEM width in km, default is 20.  
        water_mask_resolution |  c (crude), l (low), i (intermediate), h (high), f (full) .  More detailed = significantly slower.  
        
    Returns:
        volcanoes | list of dicts | each volcano is an entry in the list, and consists of a ditionary of name, lonlat, the dem (with water mask) and the lon lat extent of the dem.  
        
    History:
        2020/08/?? | MEG | Written
        2020/08/11 | MEG | add return of lon lat extent to each volcano dict, and write docs.  
    
    """
    #### begin imports/definitions
    pixs2deg = 1201
   
    
    def open_volcano_csv_file(volcano_csv_file):
        """ Conver the csv file to Python variables.  
        Inputs:
            volcano_csv_file | string | path to volcano csv file
        Returns:
            volcanoes | list | one entry for each volcano, each entry is a dictionary of info about that volcano (name string and lonlat tuple)
        History:
            2020/07/29 | MEG | Written    
        """
        import csv  
        with open(volcano_csv_file, 'r', encoding = "ISO-8859-1") as f:                             # open the csv file
          reader = csv.reader(f)
          volc_list = list(reader)                                          # list where each item is a row of the file?
        volcanoes = []
        for volc in volc_list:
            volc_dict = {}
            volc_dict['name'] = volc[0]
            volc_dict['lonlat'] = (float(volc[2]), float(volc[1]))
            volcanoes.append(volc_dict)
        return volcanoes
    
        
    def create_volcano_dem(volcano, srtm_tiles_folder, void_fill = True, width_km = 20, water_mask_resolution = 'i', download = False):
        """ Given a list that contains a ditionary about each volcano containing its name and lonlat,
        create a dem for there and add to the dictionary.  
        Inputs:
            volcano | dict | Contains name and lonlat of volcano.  
            srtm_tiles_folder | string | path to folder where SRTM3 tiles can be stored locally.  
            void_fill | boolean | Fill the odd void in the DEMs.  Slow!
            width_km | int | Dems are square with side of this length, in km.  
            water_mask_resolution | string | Resolution of vector coastlines: c l i h f   (ie coarse down to fine)
            download | boolean | If allowed to download tiles.  If all available tiles are stored locally, setting this to False will speed up the function.  
        Returns:
            volcano | dict | updates to now also contain 'dem'
            
        History:
            2020/08/?? | MEG | Written
            
        """
        import numpy as np
        import numpy.ma as ma
        from dem_tools_lib import SRTM_dem_make, water_pixel_masker, fill_gridata_voids
        from auxiliary_functions import crop_matrix_with_ll
            
        width_deg = width_km / 111                                                      # convert from km to degrees
        print(f"Creating a DEM for {volcano['name']}")    
        lon_e = int(np.ceil(volcano['lonlat'][0] + width_deg/2))                        # get the east west north south limits of the dem
        lon_w = int(np.floor(volcano['lonlat'][0] - width_deg/2))        
        lat_n = int(np.ceil(volcano['lonlat'][1] + width_deg/2))
        lat_s = int(np.floor(volcano['lonlat'][1] - width_deg/2))
        
        dem, lons, lats = SRTM_dem_make(lon_w, lon_e, lat_s, lat_n, water_mask_resolution = None,
                                        SRTM3_tiles_folder = srtm_tiles_folder, void_fill = False, download = download)                                  # make a large dem, but don't mask water bodies or void fill.  (because it's wasterful to do at a large size)
        dem_crop, ll_extent_crop = crop_matrix_with_ll(dem, (lons[0], lats[0]), pixs2deg, volcano['lonlat'], width_km)              # crop the dem, ll_extent_crop is [(lower left lon lat),(upper right lon lat)]
        if void_fill:
            dem_crop = fill_gridata_voids(dem_crop)                                                                                 # fill voids at the cropped scale.  
        water_mask =  water_pixel_masker(dem_crop, ll_extent_crop[0], ll_extent_crop[1], water_mask_resolution, verbose = True)     # mask the water bodes in the cropped dem
        volcano['dem'] = ma.array(dem_crop, mask = water_mask)                                                                      # record to dictionary.  
        volcano['ll_extent'] = ll_extent_crop                                                                                       # also record the lonlat of the lower left and upper right corners of the dem
        return volcano
    
    #### Begin function

    volcanoes_name_ll = open_volcano_csv_file(volcano_csv_file)                                                                          # open the CSV from the Smithsonian
    volcanoes = []                                                                          
    for volcano_name_ll in volcanoes_name_ll[0:10]:
        try:
            volcano = create_volcano_dem(volcano_name_ll, srtm_tiles_folder, void_fill, width_km, water_mask_resolution, download = download)         # create a dem etc. for one volcano
            #import ipdb; ipdb.set_trace()
            volcanoes.append(volcano)
        except:
            pass  
    return volcanoes


#%%

def normalise_m1_1(r2_array):
    """ Rescale a rank 2 array so that it lies within the range[-1, 1]
    """
    import numpy as np
    r2_array = r2_array - np.min(r2_array)
    r2_array = 2 * (r2_array/np.max(r2_array))
    r2_array -= 1
    return r2_array

