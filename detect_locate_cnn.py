#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:27:31 2020
Dependencies:
    step_01 | make volcano DEMs | Dependencies: SRTM_DEM_tools
    step_02 | create_random_synthetic_ifgs | Dependencies: SyInterferoPy
    step_03 | 

# open matlab defo data

# make synthetic data?

# open real data
# augment real data?

# rescale and bottlenecks.  

# train model
    train in different ways


# save weights file to website?
 get weights file from website with wget? 

# predict model

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




import numpy as np
import pickle
import sys
import glob
import os
from pathlib import Path
import shutil


#%% 0: Things to set

dependency_paths = {'syinterferopy_bin' : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'srtm_dem_tools_bin' : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/'}                                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools

SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                     'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                     'SRTM3_tiles_folder'       : './SRTM3/',                                   # folder for DEM tiles.  
                     'download'                 : True,                                         # If tile is not available locally, try to download it
                     'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                     'side_length'              : 40e3}                                         # the side length in metres of the DEM.  To allow for different crops of this, it should be somewhat bigger than 224 (the number of pixels) x 90 (pixel size) ~ 20e3

synthetic_ifgs_n_files  =  2                                                                      # we will generate this many files, each of n_ifgs (so, e.g. 2 files of 20 ifgs = 40 in total)
synthetic_ifgs_folder   = '01_github_example'
synthetic_ifgs_settings = {'defo_sources'           :  ['no_def', 'dyke', 'sill', 'mogi'],      # deformation patterns that will be included in the dataset.  
                           'n_ifgs'                 : 5,                                        # the number of synthetic interferograms to generate PER FILE
                           'n_pix'                  : 224,                                      # number of 3 arc second pixels (~90m) in x and y direction
                           'outputs'                : ['uuu', 'uud'],                           # channel outputs.  uuu = unwrapped across all 3
                           'intermediate_figure'    : False,                                    # if True, a figure showing the steps taken during creation of each ifg is displayed.  
                           'coh_scale'              : 5000,                                     # The length scale of the incoherent areas, in meters.  A smaller value creates smaller patches, and a larger one creates larger pathces.  
                           'coh_threshold'          : 0.7,                                      # if 1, there are no areas of incoherence, if 0 all of ifg is incoherent.  
                           'min_deformation'        : 0.05,                                     # deformation pattern must have a signals of at least this many metres.  
                           'max_deformation'        : 0.25,                                     # deformation pattern must have a signal no bigger than this many metres.  
                           'snr_threshold'          : 2.0,                                      # signal to noise ratio (deformation vs turbulent and topo APS) to ensure that deformation is visible.  A lower value creates more subtle deformation signals.
                           'turb_aps_mean'          : 0.02,                                     # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)
                           'turb_aps_length'        : 5000}                                     # turbulent APS will be correlated on this length scale, in metres.  
                           



              
#%% Import dependencies (paths set above)

sys.path.append(dependency_paths['syinterferopy_bin'])
sys.path.append(dependency_paths['srtm_dem_tools_bin'])

from dem_tools_lib import SRTM_dem_make_batch
from random_generation_functions import create_random_synthetic_ifgs


#%% 1: Create or load DEMs for the volcanoes to be used for synthetic data.  

np.random.seed(0)                                                                                           # 0 used in the example

volcanoes = open_smithsonian_csv_file('smithsonian_name_lat_lon.csv', side_length=SRTM_dem_settings['side_length'])

print('Quick fix to shorten the number of volcanoes')
volcanoes = volcanoes[:5]

try:
    print('Trying to open a .pkl of the DEMs... ', end = '')
    with open('volcano_dems.pkl', 'rb') as f:
        volcano_dems = pickle.load(f)
    f.close()
    print('Done.  ')

except:
    print('Failed.  Generating them from scratch, which can be slow.  ')
    del SRTM_dem_settings['side_length']                                                                # this key is no longer needed, so delete.  
    volcano_dems = SRTM_dem_make_batch(volcanoes, **SRTM_dem_settings)                                  # make the DEMS
    with open(f'volcano_dems.pkl', 'wb') as f:
        pickle.dump(volcano_dems, f)
    print('Saved the dems as a .pkl for future use.  ')



#%% 2: Create or load the synthetic interferograms.  

print('Determining if files containing the synthetic deformation patterns exist... ', end = '')
data_files = glob.glob(str(Path(f"./synthetic_data/{synthetic_ifgs_folder}/*.pkl")))             #
if len(data_files) == synthetic_ifgs_n_files:
    print(f"The correct number of files were found ({synthetic_ifgs_n_files}) so no new ones will be generated.  ")
else:
    print(f"{len(data_files)} files were found, but {synthetic_ifgs_n_files} were requested.  "
          f"The folder containing these (./synthetic_data/{synthetic_ifgs_folder})will be deleted, and the correct number of files generated.  ")
    answer = input("Do you wish to continue ('y' or 'n')?")
    if answer == 'n':
        sys.exit()
    elif answer == 'y':
        try:
            shutil.rmtree(str(Path(f"./synthetic_data/{synthetic_ifgs_folder}/")))
        except:
            pass
        os.mkdir(Path(f"./synthetic_data/{synthetic_ifgs_folder}"))
        for file_n in range(synthetic_ifgs_n_files):
            X_all, Y_class, Y_loc = create_random_synthetic_ifgs(volcano_dems, **synthetic_ifgs_settings)
            with open(Path(f'./synthetic_data/{synthetic_ifgs_folder}/data_file_{file_n}.pkl'), 'wb') as f:
                pickle.dump(X_all, f)
                pickle.dump(Y_class, f)
                pickle.dump(Y_loc, f)
            f.close()
            del X_all, Y_class, Y_loc
    else:
        print(f"Answer ({answer}) was not understood as either 'y' or 'n' so exiting to err on the side of caution")
        sys.exit()
    
        
    

import sys; sys.exit()
#%%


#     with open('volcano_dems.pkl', 'rb') as f:
#         volcano_dems = pickle.load(f)
#     f.close()
#     print('Done.  ')

# except:
#     print('Failed.  Generating them from scratch, which can be slow.  ')
#     del SRTM_dem_settings['side_length']                                                                # this key is no longer needed, so delete.  
#     volcano_dems = SRTM_dem_make_batch(volcanoes, **SRTM_dem_settings)                                  # make the DEMS
#     with open(f'volcano_dems.pkl', 'wb') as f:
#         pickle.dump(volcano_dems, f)
        
#     #also save the dict of settings as a text file
        
#     print('Saved the dems as a .pkl for future use.  ')


        

#%%

input("press enter to continue...")

print("continuing")




############################################################################


#%% Things to set

# General settings
threshold_m = 0.015                  # deformation above this level will be included in the localization box
snr_visible = 5.0                    # SNR of deformation signal over noise (topo. correlated and turbulent APS) required to ifg to be classsed as usable.  

defo_fraction = 0.8                     # this fraction of the deformation above the threshold must be on land (i.e. checked using the DEM)
turb_aps_mean = 2.5                        # max size of turbulent APS in cm
topo_aps_mean = 56.0                    # rad/km of delay for each sar acquisition 
topo_aps_var = 8.                       # rad/km variance for each sar acquisition




padding = 10


#pix_in_m = 92.6                    # pixels in m.  92.6m for a SRTM3 pixel on equator


dem_settings = {'volcano_csv_file' : '/home/matthew/university_work/02_neural_networks_python/Detect-Locate-CNN/smithsonian_name_lat_lon.csv',
                'srtm_tiles_folder' : '/home/matthew/university_work/data/SRTM/SRTM_3_tiles/',
                'void_fill' : True,
                'download' : True,
                'width_km' : 30,
                'water_mask_resolution' : 'f'}                      # c(rude), l(ow), i(termediate), h(igh), f(ull)


synth_data_settings = {'n_files' : 2,
                       'n_per_file' : 5,
                       'n_classes' : 4,                                          # no def, inflation, dyke mogi would be 4
                       'min_deformation' : 0.02,                                 # deformation of at least X m is required
                       'max_deformation' : 0.25,                                 # but deformation must not be larger than X m
                       'n_pix' : 224}                                           # number of pixels of square outputs
    





#%% Imports

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import pickle
import os
import sys


sys.path.append(dependency_paths['syinterferopy_bin'])
sys.path.append(dependency_paths['srtm_dem_tools_bin'])

#from aux_functions import create_random_defo_m, def_and_dem_translate, check_def_visible, normalise_m1_1
from syinterferopy_functions import coherence_mask, atmosphere_topo, atmosphere_turb

print('Temporary import of matrix_show for debugging.  ')
sys.path.append('/home/matthew/university_work/python_stuff/python_scripts/')
from small_plot_functions import matrix_show


#%% 01 Create of load dems for each volcano.  






import sys; sys.exit()



#%%
print("Trying to open a .pkl of volcano dems... ", end = '')
try:
    with open('volcano_dems.pkl', 'rb') as f:
        volcanoes = pickle.load(f)
    f.close()
    print('Success!')
except:
    print('Failed.  Starting to make these (~2 minutes per volcano = 8 hours)...', end = '')
    volcanoes = create_volcano_dems(**dem_settings)                                                 # open all subaerial volcanoes and make DEMS for them.  
    with open(f'volcano_dems.pkl', 'wb') as f:
        pickle.dump(volcanoes, f)
    f.close()
    print('Done!')

dem_settings['ny'], dem_settings['nx'] = volcanoes[0]['dem'].shape

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

#%%




#%% 03 Create synthetic interferograms using DEMS (units are metres at this point)


"""
Structure ideas:
    for file in files:
        generate n_in_file (create_random_synthetic_ifgs)           
        save n_in_file
"""
                                                 


import sys; sys.exit()
    

#%%


    
#%%
    
volcanoes = volcanoes
out_folder = './test'
outputs = ['uuu', 'uud']
n_files = synth_data_settings['n_files']
n_ifgs = synth_data_settings['n_ifgs']
n_pix = synth_data_settings['n_pix']
n_classes = synth_data_settings['n_classes']
intermediate_figure = True

from aux_functions import combine_signals


os.makedirs(out_folder)
for output_extension in outputs:                                               # loop through them
    os.makedirs(f"{out_folder}/{output_extension}")                                  # making them


for out_file_num in range(n_files):

    
    import sys; sys.exit()
    save_signals(X_all, Y_class, Y_loc)

#%%
            
        
    #     dem = ma.array(dem, mask=mask_coh_water)                                                # mask dem - water and also incoherent bits.  Suppose we don't need to mask incoherent bits, but done to make consistent.  
        
    
    #     # 1b: Check that still visible over noise (topo APS and turb APS)
        

            
    #     # 2: combine signals
    #     if viable_location and viable_snr:
    #         print(f"| Viable combination found ({count} attempts).  ")
    #         if np.random.rand() < (1/ synth_data_settings['n_classes']):                                                        # a certain fraction of the signals shouldn't contain deformation.  
    #             ph_all = ((4*np.pi)/s1_wav)*(APS_topo_m + APS_turb_m)                                                           # comnbine the signals in m, then convert to rads
    #             Y_class[succesful_generate,0] = 0                                                                               # label as no deformation
    #             Y_loc[succesful_generate,:] = np.array([0,0,0,0])                                                               # 0 0 0 0 for no deformaiton
    #         else:
    #             ph_all = ((4*np.pi)/s1_wav) * (APS_topo_m + APS_turb_m + defo_m)                                                 # combine the signals in m, then convert to rads.  Here we include deformation.  
    #             Y_class[succesful_generate,0] = defo_source_n                                                                    # write the label as a number
    #             Y_loc[succesful_generate,:] = np.array([loc_list[0][0],loc_list[0][1],loc_list[1][0],loc_list[1][1]])            # location of deformation 
            

    #         ph_all_wrap = (ph_all + np.pi) % (2 * np.pi ) - np.pi                       # wrap
    #         #1 Genreate SAR amplitude 
    #         look_az = heading - 90     
    #         look_in = 90 - incidence                                                 # 
    #         ls = LightSource(azdeg=look_az, altdeg=look_in)
    #         sar_amplitude = normalise_m1_1(ls.hillshade(dem))                                                                   # in range [-1 1]
            
    #         # make real and imaginary
    #         ifg_real = sar_amplitude * np.cos(ph_all_wrap)
    #         ifg_imaginary = sar_amplitude * np.sin(ph_all_wrap)
    #         ifg_real = ifg_real + sar_speckle_strength * np.random.randn(ifg_real.shape[0], ifg_real.shape[1])                 # add noise to real
    #         ifg_imaginary = ifg_imaginary + sar_speckle_strength * np.random.randn(ifg_real.shape[0], ifg_real.shape[1])         # and imaginary

    #         # make things rank 4
    #         ph_all = ma.expand_dims(ma.expand_dims(ph_all, axis = 0), axis = 3)
    #         dem = ma.expand_dims(ma.expand_dims(dem, axis = 0), axis = 3)
    #         ifg_real = ma.expand_dims(ma.expand_dims(ifg_real, axis = 0), axis = 3)
    #         ifg_imaginary = ma.expand_dims(ma.expand_dims(ifg_imaginary, axis = 0), axis = 3)
    #         ph_all_wrap = ma.expand_dims(ma.expand_dims(ph_all_wrap, axis = 0), axis = 3)
            

    #         X_1s = []                                                                                   # this list will store one visualisation of 3 channel data in each form (uuu, uud, rid, www, wwwd)
    #         X_1s.append(ma.concatenate((ph_all, ph_all, ph_all), axis = 3))                             # uuu
    #         X_1s.append(ma.concatenate((ph_all, ph_all, dem), axis = 3))                                # uud
    #         X_1s.append(ma.concatenate((ifg_real, ifg_imaginary, dem), axis = 3))                       # rid                                                                                     X_1s.append(ma.concatenate((ph_all_wrap, ph_all_wrap, ph_all_wrap), axis = 3))              # www
    #         X_1s.append(ma.concatenate((ph_all_wrap, ph_all_wrap, dem), axis = 3))                      #wwd
            
    #         nans_present_flag = 0                                                           #check if any have nans in them.  
    #         for X_1 in X_1s:
    #             nans_present_current = np.max(np.ravel(np.isnan(X_1s)).astype(int))          # nans flag as an int
    #             if nans_present_current == 1:
    #                 nans_present_flag = 1                                                   # if there are nans, set the flag to 1
    #             else:
    #                 pass                                                                    # if no nans, don't update the flag (leave at 1 if already encountered nans)
    #         nans_present_flag = bool(nans_present_flag)                              # convert from int to boolean
                    
    #         if nans_present_flag:
    #             pass                                                                    # do nothing and start te loop over if we have nans
    #         else:
    #             for i in range(len(X_1s)):
    #                 X_all[i][succesful_generate,: :, :] = X_1s[i]
    #             succesful_generate += 1
    #             print(f"Generated {succesful_generate} of {n_per_file} synthetic interferograms.  ")
    #     else:
            
   
    # # step back to loop through each file, not each ifg.   save what is required, not the most elegant way of doing this.  
    # if 'uuu' in outputs:
    #     with open(f'{out_folder}uuu/data_file_{out_file_num}.pkl', 'wb') as f:
    #         pickle.dump(X_all[0], f)
    #         pickle.dump(Y_class, f)
    #         pickle.dump(Y_loc, f)

    # if 'uud' in outputs:
    #     with open(f'{out_folder}uud/data_file_{out_file_num}.pkl', 'wb') as f:
    #         pickle.dump(X_all[1], f)
    #         pickle.dump(Y_class, f)
    #         pickle.dump(Y_loc, f)
            
    # if 'rid' in outputs:
    #     with open(f'{out_folder}rid/data_file_{out_file_num}.pkl', 'wb') as f:
    #         pickle.dump(X_all[2], f)
    #         pickle.dump(Y_class, f)
    #         pickle.dump(Y_loc, f)

    # if 'www' in outputs:
    #     with open(f'{out_folder}www/data_file_{out_file_num}.pkl', 'wb') as f:
    #         pickle.dump(X_all[3], f)
    #         pickle.dump(Y_class, f)
    #         pickle.dump(Y_loc, f)
            
    # if 'wwd' in outputs:
    #     with open(f'{out_folder}wwd/data_file_{out_file_num}.pkl', 'wb') as f:
    #         pickle.dump(X_all[4], f)
    #         pickle.dump(Y_class, f)
    #         pickle.dump(Y_loc, f)
    

    


                    
                


            
            
            
            # pick whether to include deformaion
                # if includes deformation:
                        #, generate with random parameters (but ensure signal is correct size)
                        # check visible within scene
            # write label
            # conver to different forms.  
                
    
    # save rank 4 ma to file (or files, with different formats in different files.)
    
#%%
test = {}
for output in outputs:
    test[output] = np.random.rand(10,2)
    
    
    
#%%

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt



X, Y = np.meshgrid(np.arange(0,mask_coh_water.shape[1]), np.arange(0,mask_coh_water.shape[0]))


fig, ax = plt.subplots()
#CS = ax.contourf(X, Y, mask_coh_water, alpha = 0.1)
#ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Simplest default with labels')
#ax.imshow(defo_m)


