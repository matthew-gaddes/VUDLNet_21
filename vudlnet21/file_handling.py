#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:28:33 2021

@author: matthew
"""

def open_smithsonian_csv_file(smithsonian_csv_file, side_length = 40e3):
    """ Conver the csv file to a list of python dictionaries
    Inputs:
        smithsonian_csv_file | string | path to volcano csv file
        side_length | int | the side length of the DEM in metres.  e.g. 40000m = 40km
    Returns:
        volcanoes | list | one entry for each volcano, each entry is a dictionary of info about that volcano (name string and lonlat tuple)
    History:
        2020/07/29 | MEG | Written    
        2020/10/19 | MEG | Add option to set side length
    """
    import csv  
    with open(smithsonian_csv_file, 'r', encoding = "ISO-8859-1") as f:                             # open the csv file
      reader = csv.reader(f)
      volc_list = list(reader)                                          # list where each item is a row of the file?
    volcanoes = []
    for volc in volc_list:
        volc_dict = {}
        volc_dict['name'] = volc[0]
        volc_dict['centre'] = (float(volc[2]), float(volc[1]))
        volc_dict['side_length'] =  (side_length, side_length)
        volcanoes.append(volc_dict)
    return volcanoes



#%%


def open_all_volcnet_files(volcnet_dir, defo_sources):
    """ Given a directory of VOLCNET files, open all of them.  
    """
    import glob
    import numpy as np
    import numpy.ma as ma
    
    VolcNet_files = sorted(glob.glob(str(volcnet_dir / '*.pkl')))             #  get a list of the paths to all the VolcNet files       
    if len(VolcNet_files) == 0:
        raise Exception('No VolcNet files have been found.  Perhaps the path is wrong? Or perhaps you only want to use synthetic data?  In which case, this section can be removed.  Exiting...')
    
    X_1s = []
    Y_class_1s = []
    Y_loc_1s = []
    for VolcNet_file in VolcNet_files:
        X_1, Y_class_1, Y_loc_1 = open_VolcNet_file(VolcNet_file, defo_sources)
        X_1s.append(X_1)
        Y_class_1s.append(Y_class_1)
        Y_loc_1s.append(Y_loc_1)
    X = ma.concatenate(X_1s, axis = 0)
    Y_class = np.concatenate(Y_class_1s, axis = 0)
    Y_loc = np.concatenate(Y_loc_1s, axis = 0)
    
    return X, Y_class, Y_loc


#%%


def open_VolcNet_file(file_path, defo_sources):
    """A file to open a single VolcNet file and extrast the deformation source into a one hot encoded numpy array, 
    and the deforamtion location as a n_ifg x 4 array.  
    Ifgs are masked arrays, in m, with up as positive.  
    
    Inputs:
        file_path | string or Path | path to fie
        defo_sources | list of strings | names of deformation sources, should the same as the names used in VolcNet
    
    Returns:
        X | r4 masked array | ifgs, as above. ? x y x x n_channels  
        Y_class | r2 array | class labels, ? x n_classes
        Y_loc | r2 array | locations of signals, ? x 4 (as x,y, width, heigh)
    
    History:
        2020_01_11 | MEG | Written
    """
    import pickle
    import numpy as np
    import numpy.ma as ma
    
    # 0: Open the file    
    with open(file_path, 'rb') as f:                                                      # open the real data file
        ifgs = pickle.load(f)                                                                # this is a masked array of ifgs
        ifgs_dates = pickle.load(f)                                                          # list of strings, YYYYMMDD that ifgs span
        pixel_lons = pickle.load(f)                                                          # numpy array of lons of lower left pixel of ifgs
        pixel_lats = pickle.load(f)                                                          # numpy array of lats of lower left pixel of ifgs
        all_labels = pickle.load(f)                                                          # list of dicts of labels associated with the data.  e.g. deformation type etc.  
    f.close()        
    
    # 1: Initiate arrays
    n_ifgs = ifgs.shape[0]                                                                      # get number of ifgs in file
    X = ifgs                                                                                    # soft copy to rename
    Y_class = np.zeros((n_ifgs, len(defo_sources)))                                             # initiate
    Y_loc = np.zeros((n_ifgs, 4))                                                               # initaite

    # 2: Convert the deformation classes to a one hot encoded array and the locations to an array
    for n_ifg in range(n_ifgs):                                                                 # loop through the ifgs
        current_defo_source = all_labels[n_ifg]['deformation_source']                           # get the current ifgs deformation type/class label
        arg_n = defo_sources.index(current_defo_source)                                         # get which number in the defo_sources list this is
        Y_class[n_ifg, arg_n] = 1                                                               # write it into the correct position to make a one hot encoded list.  
        Y_loc[n_ifg, :] = all_labels[n_ifg]['deformation_location']                             # get the location of deformation.  
        
    return X, Y_class, Y_loc
