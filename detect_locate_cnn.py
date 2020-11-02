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
import keras
from keras.applications.vgg16 import VGG16


#%% 0: Things to set

dependency_paths = {'syinterferopy_bin' : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'srtm_dem_tools_bin' : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/'}                                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools

SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                     'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                     'SRTM3_tiles_folder'       : './SRTM3/',                                   # folder for DEM tiles.  
                     'download'                 : True,                                         # If tile is not available locally, try to download it
                     'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                     'side_length'              : 40e3}                                         # the side length in metres of the DEM.  To allow for different crops of this, it should be somewhat bigger than 224 (the number of pixels) x 90 (pixel size) ~ 20e3

ifg_settings            = {'n_per_file'         : 10}                                            # number of ifgs per data file.  
synthetic_ifgs_n_files  =  6                                                                    # numer of files of synthetic data
synthetic_ifgs_folder   = '01_github_example'
synthetic_ifgs_settings = {'defo_sources'           : ['dyke', 'sill', 'no_def'],               # deformation patterns that will be included in the dataset.  
                           'n_ifgs'                 : ifg_settings['n_per_file'],               # the number of synthetic interferograms to generate PER FILE
                           'n_pix'                  : 224,                                      # number of 3 arc second pixels (~90m) in x and y direction
                           'outputs'                : ['uuu'],                                  # channel outputs.  uuu = unwrapped across all 3
                           'intermediate_figure'    : False,                                    # if True, a figure showing the steps taken during creation of each ifg is displayed.  
                           'coh_scale'              : 5000,                                     # The length scale of the incoherent areas, in meters.  A smaller value creates smaller patches, and a larger one creates larger pathces.  
                           'coh_threshold'          : 0.7,                                      # if 1, there are no areas of incoherence, if 0 all of ifg is incoherent.  
                           'min_deformation'        : 0.05,                                     # deformation pattern must have a signals of at least this many metres.  
                           'max_deformation'        : 0.25,                                     # deformation pattern must have a signal no bigger than this many metres.  
                           'snr_threshold'          : 2.0,                                      # signal to noise ratio (deformation vs turbulent and topo APS) to ensure that deformation is visible.  A lower value creates more subtle deformation signals.
                           'turb_aps_mean'          : 0.02,                                     # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)
                           'turb_aps_length'        : 5000}                                     # turbulent APS will be correlated on this length scale, in metres.  

real_ifg_settings       = {'augmentation_factor' : 2}                                           # factor to agument by.  E.g. if set to 10 and there are 30 data, there will be 300 augmented data.  
                           

cnn_settings = {'input_range' : {'min':0, 'max':255}}


              
#%% Import dependencies (paths set above)

sys.path.append(dependency_paths['syinterferopy_bin'])
sys.path.append(dependency_paths['srtm_dem_tools_bin'])

from dem_tools_lib import SRTM_dem_make_batch                                       # From SRTM dem tools
from random_generation_functions import create_random_synthetic_ifgs                # From SyInterferoPy
from detect_locate_plotting_functions import plot_data_class_loc_caller, open_pkl_and_plot                                # from this repo
from detect_locate_nn_functions import augment_data, choose_for_augmentation, merge_and_rescale_data                      # from this repo


#%% 1: Create or load DEMs for the volcanoes to be used for synthetic data.  
print("\nStep 01: Creating or loadings DEMs")

np.random.seed(0)                                                                                           # 0 used in the example

volcanoes = open_smithsonian_csv_file('./step_01_dem_data/smithsonian_name_lat_lon.csv', side_length=SRTM_dem_settings['side_length'])

print('Quick fix to shorten the number of volcanoes')
volcanoes = volcanoes[:5]

try:
    print('Trying to open a .pkl of the DEMs... ', end = '')
    with open('./step_01_dem_data/volcano_dems.pkl', 'rb') as f:
        volcano_dems = pickle.load(f)
    f.close()
    print('Done.  ')

except:
    print('Failed.  Generating them from scratch, which can be slow.  ')
    del SRTM_dem_settings['side_length']                                                                # this key is no longer needed, so delete.  
    volcano_dems = SRTM_dem_make_batch(volcanoes, **SRTM_dem_settings)                                  # make the DEMS
    with open(f'./step_01_dem_data/volcano_dems.pkl', 'wb') as f:
        pickle.dump(volcano_dems, f)
    f.close()
    print('Saved the dems as a .pkl for future use.  ')



#%% 2: Create or load the synthetic interferograms.  
print("\nStep 02: Creating or loading synthetic interferograms")

n_synth_data = ifg_settings['n_per_file'] * synthetic_ifgs_n_files

print('Determining if files containing the synthetic deformation patterns exist... ', end = '')
data_files = glob.glob(str(Path(f"./step_02_synthetic_data/{synthetic_ifgs_folder}/*.pkl")))             #
if len(data_files) == synthetic_ifgs_n_files:
    print(f"The correct number of files were found ({synthetic_ifgs_n_files}) so no new ones will be generated.  ")
else:
    print(f"{len(data_files)} files were found, but {synthetic_ifgs_n_files} were requested.  "
          f"The folder containing these (./step_02_synthetic_data/{synthetic_ifgs_folder})will be deleted, and the correct number of files generated.  ")
    answer = input("Do you wish to continue ('y' or 'n')?")
    if answer == 'n':
        sys.exit()
    elif answer == 'y':
        try:
            shutil.rmtree(str(Path(f"./step_02_synthetic_data/{synthetic_ifgs_folder}/")))
        except:
            pass
        os.mkdir(Path(f"./step_02_synthetic_data/{synthetic_ifgs_folder}"))
        for file_n in range(synthetic_ifgs_n_files):
            X_all, Y_class, Y_loc, Y_source_kwargs = create_random_synthetic_ifgs(volcano_dems, **synthetic_ifgs_settings)
            Y_class = keras.utils.to_categorical(Y_class, len(synthetic_ifgs_settings['defo_sources']), dtype='float32')          # convert to one hot encoding (from class labels)
            with open(Path(f'./step_02_synthetic_data/{synthetic_ifgs_folder}/data_file_{file_n}.pkl'), 'wb') as f:
                pickle.dump(X_all[synthetic_ifgs_settings['outputs'][0]], f)                                                    # usual output style is many channel formats in a dict, but we are only interesetd in the one we generated.  
                pickle.dump(Y_class, f)
                pickle.dump(Y_loc, f)
            f.close()
            del X_all, Y_class, Y_loc
    else:
        print(f"Answer ({answer}) was not understood as either 'y' or 'n' so exiting to err on the side of caution")
        sys.exit()


open_pkl_and_plot(f"step_02_synthetic_data/{synthetic_ifgs_folder}/data_file_0.pkl", n_data = 15, window_title ='Sample of synthetic data')                                        # open and plot the data in 1 file

     
#%% 3: Load the real data (and augment).  Note that these are in metres, and use one hot encoding for the class, and are masked arrays (incoherence and water are masked)
print("\nStep 03: Loading and augmenting the real interferograms.  ")

with open("step_03_real_data/real_data_class_locs_subset.pkl", 'rb') as f:                                                      # open the real data file
    X = pickle.load(f)                                                                                                          # this is a masked array
    Y_class = pickle.load(f)                                                                                                    # numpy array
    Y_loc = pickle.load(f)                                                                                                      # numpy array
f.close()    
plot_data_class_loc_caller(X, Y_class, Y_loc, source_names = ['dyke', 'sill', 'no def'], window_title = 'Real data')            # plot the data in it

print('Commented out the augmentation as a quick fix.  ')
# print(f"Starting to augment the real data...")
# n_augmented_files = int((X.shape[0] * real_ifg_settings['augmentation_factor']) / ifg_settings['n_per_file'])                   # get the number of real data, multiply by the augmentation factor, find how many files will be required as n data per file
# for n_augmented_file in range(n_augmented_files):                                                                               # loop through each file that is to be made
#     print(f'    File {n_augmented_file} of {n_augmented_files}...', end = '')  
#     X_sample, Y_class_sample, Y_loc_sample = choose_for_augmentation(X, Y_class, Y_loc, ifg_settings['n_per_file'])             # chose a subset of the data, and balance the classes.  
#     X_aug, Y_class_aug, Y_loc_aug = augment_data(X_sample, Y_class_sample, Y_loc_sample, ifg_settings['n_per_file'])            # do the augmentation

#     with open(f"./step_03_real_data/augmented/data_file_{n_augmented_file}.pkl", 'wb') as f:                                    # save the output as a pickle
#         pickle.dump(X_aug, f)
#         pickle.dump(Y_class_aug, f)
#         pickle.dump(Y_loc_aug, f)
#     f.close()
#     print('Done!')
# print('Done!')

open_pkl_and_plot("./step_03_real_data/augmented/data_file_0.pkl", n_data = 15, window_title = 'Sample of augmented real data')



#%% 4: Merge real and synthetic data, and rescale to desired range (e.g. [0, 1], [0, 255], [-125, 125] etc)

print("\nStep 04: Mergring the real and synthetic interferograms and rescaling to CNNs input range.")

synthetic_data_files = glob.glob(str(Path(f"./step_02_synthetic_data/{synthetic_ifgs_folder}/*.pkl")))             #
real_data_files = glob.glob(str(Path(f"./step_03_real_data/augmented//*.pkl")))             #
merge_and_rescale_data(synthetic_data_files, real_data_files, cnn_settings['input_range'])

#%% 5: Compute bottlenceck features

print("\nStep 05: Computing the bottleneck features.")
vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))

data_out_files = glob.glob(f'step_04_merged_rescaled_data/*.npz')           # get the files outputted by part 1

for file_n, data_out_file in enumerate(data_out_files):
    print(f'Bottlneck file {file_n}:')
    
    data_out_file = Path(data_out_file)                                               # convert to path object
    bottleneck_file_name = data_out_file.parts[-1].split('.')[0]                      # and get last part which is filename
    
    data = np.load(data_out_file)
    X = data['X']
    Y_class = data['Y_class']
    Y_loc = data['Y_loc']
    
    X_btln = vgg16_block_1to5.predict(X, verbose = 1)                                             # predict up to bottleneck    
    
    np.savez(f'step_05_bottleneck/{bottleneck_file_name}_bottleneck.npz', X = X_btln, Y_class = Y_class, Y_loc = Y_loc)                            #, source_names = source_names)  



#%%

import sys; sys.exit()


#%% 6: Train the CNN


#%% 7: Evaluate the CNN



#%%

import sys; sys.exit()


#% Things to set

augmentation_output = '/nfs/a1/homes/eemeg/2_neural_networks_python/3_python_defo_atm_data/02_augmented_real_data'
train_data_folder = '/nfs/a1/homes/eemeg/2_neural_networks_python/06_real_data'

source_names = ['dyke', 'sill', 'no def.']            
n_outfiles = 40

#% Open the npz of training data

npz_arrays = np.load(f'{train_data_folder}/training_data.npz')
#npz_arrays = np.load(f'{train_data_folder}/training_data_small.npz')
X_train = npz_arrays['X']
Y_class_train = npz_arrays['Y_class']
Y_loc_train = npz_arrays['Y_loc']

#% Choose a subset to be augmented that represents the samples evenly
# at this point data are in correct range, mean centered, and masked bits are 0

X_sample, Y_class_sample, Y_loc_sample = choose_for_augmentation(X_train, Y_class_train, Y_loc_train, 3)

plot_data_class_loc_caller(X_sample, classes=Y_class_sample, locs=Y_loc_sample, source_names = source_names)    


#% do data augmentation

for out_file_num in range(n_outfiles):
    print(f'File {out_file_num}...', end = '')
    X_sample, Y_class_sample, Y_loc_sample = choose_for_augmentation(X_train, Y_class_train, Y_loc_train, 3)
    X_aug, Y_class_aug, Y_loc_aug = augment_data(X_sample, Y_class_sample, Y_loc_sample)

    np.savez(f'{augmentation_output}/uuu/data_file_{out_file_num}.npz', X = X_aug, Y_class= Y_class_aug, Y_loc = Y_loc_aug)                            #, source_names = source_names)  

    print('Done!')

#plot_data_class_loc_caller(X_aug, classes=Y_class_aug, locs=Y_loc_aug, source_names = source_names)






#%% 

#%% 4: Rescale and merge the data


#%% 5: Compute bottlenecks

#%% 6: Train a model.  

#%%

import sys; sys.exit()

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


