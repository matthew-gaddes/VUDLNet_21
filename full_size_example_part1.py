#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: matthew

"""


import numpy as np
import numpy.ma as ma 
import pickle
import sys
import glob
import os
from pathlib import Path
import shutil
import pdb





#%% MEG debug import

sys.path.append("/home/matthew/university_work/python_stuff/python_scripts")
from small_plot_functions import matrix_show, quick_linegraph
import matplotlib.pyplot as plt




#%% 0: Things to set

dependency_paths = {#'syinterferopy'        : '/home/matthew/university_work/15_my_software_releases/SyInterferoPy-3.0.1/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'syinterferopy'         : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy',

                    'srtm_dem_tools'        : '/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-2.1.1/',                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools
                    'volcnet'               : '/home/matthew/university_work/13_volcnet',                                                   # Available from Github: https://github.com/matthew-gaddes/VolcNet
                    'deep_learning_tools'   : '/home/matthew/university_work/20_deep_learning_tools'}                                      # Available from Github: https://github.com/matthew-gaddes/Deep-Learning-Tools


project_outdir = Path('./')

#step 01 (creating DEMS):
SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                     'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                     'SRTM3_tiles_folder'       : './SRTM3/',                                   # folder for DEM tiles.  
                     'download'                 : True,                                         # If tile is not available locally, try to download it
                     'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                     'side_length'              : 40e3}                                         # the side length in metres of the DEM.  To allow for different crops of this, it should be somewhat bigger than 224 (the number of pixels) x 90 (pixel size) ~ 20e3
n_volcanoes_used = 512                                                                            # there are 512 volcanoes in the Smithsonian file, but to speed up making the DEMS not all of these need to be used.   

#step 02 (making 20000 synthetic interferograms):
ifg_settings            = {'n_per_file'         : 500}                                            # number of ifgs per data file.  
synthetic_ifgs_n_files  =  40                                                                   # numer of files of synthetic data

synthetic_ifgs_settings = {'defo_sources'           : ['dyke', 'sill', 'no_def'],               # deformation patterns that will be included in the dataset.  
                           'n_ifgs'                 : ifg_settings['n_per_file'],               # the number of synthetic interferograms to generate PER FILE
                           'n_pix'                  : 224,                                      # number of 3 arc second pixels (~90m) in x and y direction
                           'outputs'                : ['u'],                                    # channel outputs.  u = unwrapped.  
                           'intermediate_figure'    : False,                                    # if True, a figure showing the steps taken during creation of each ifg is displayed.  
                           'cov_coh_scale'          : 5000,                                     # The length scale of the incoherent areas, in meters.  A smaller value creates smaller patches, and a larger one creates larger pathces.  
                           'coh_threshold'          : 0.7,                                      # if 1, there are no areas of incoherence, if 0 all of ifg is incoherent.  
                           'min_deformation'        : 0.05,                                     # deformation pattern must have a signals of at least this many metres.  
                           'max_deformation'        : 0.25,                                     # deformation pattern must have a signal no bigger than this many metres.  
                           'snr_threshold'          : 2.0,                                      # signal to noise ratio (deformation vs turbulent and topo APS) to ensure that deformation is visible.  A lower value creates more subtle deformation signals.
                           'turb_aps_mean'          : 0.02,                                     # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)
                           'turb_aps_length'        : 5000}                                     # turbulent APS will be correlated on this length scale, in metres.  

#step 03 (load real data and augment):
volcnet_def_min = 0.05                                                                          # in m, real ifg must have a signal bigger than this to be classed as deformation.  
file_batch_size = 27                                                                            # number of files to open and merge at one time.  Depends on amount of RAM available. 
augmentation_factor = 2                                                                        # factor to augment by.  E.g. if set to 10 and there are 30 data, there will be 300 augmented data.
n_files_test = 2
                                                                                                   # v2 data has ? ifgs, so ? x 89 ~ 20000 etc.  

# step 04 (merge synthetic and real, and rescale to desired range)
cnn_settings = {'input_range'       : {'min':-1, 'max':1}}


print(f"\n\nSetting a random seed\n\n")   
np.random.seed(0)                                                                                           # 0 used in the example

              
#%% Import dependencies (paths set above)

from vudlnet21.file_handling import open_smithsonian_csv_file, merge_and_rescale_data, shuffle_data_pkls
from vudlnet21.plotting import plot_data_class_loc_caller, open_datafile_and_plot
from vudlnet21.augmentation import augment_data



sys.path.append(dependency_paths['srtm_dem_tools'])
from dem_tools_lib import SRTM_dem_make_batch                                       # From SRTM dem tools

sys.path.append(dependency_paths['syinterferopy'])
import syinterferopy
from syinterferopy.random_generation import create_random_synthetic_ifgs                # From SyInterferoPy


sys.path.append(dependency_paths['deep_learning_tools'])
import deep_learning_tools
#from deep_learning_tools.file_handling import 


sys.path.append(dependency_paths['volcnet'])
import volcnet
#from volcnet.plotting import volcnet_ts_visualiser
from volcnet.plotting import plot_volcnet_files_labels
from volcnet.aux import ll_2_pixel
from volcnet.labelling import label_volcnet_files
from volcnet.creating_ifgs import create_volcnet_ifgs





#%% 1: Create or load DEMs for the volcanoes to be used for synthetic data.  
print("\nStep 01: Creating or loadings DEMs")
volcanoes = open_smithsonian_csv_file(project_outdir / 'step_01_dem_data/smithsonian_name_lat_lon.csv', side_length=SRTM_dem_settings['side_length'])          # conver the csv file into a list with a dict for each volcanoe

print(f"{len(volcanoes)} volcanoes have been loaded from the .csv file from the Smithsonian, "                                                  # update terminal
      f"and the first {n_volcanoes_used} are being used.  ")
volcanoes = volcanoes[:n_volcanoes_used]                                                                                                        # crop number of volcanoes as required.  

try:
    print('Trying to open a .pkl of the DEMs... ', end = '')
    with open(project_outdir / 'step_01_dem_data/volcano_dems.pkl', 'rb') as f:
        volcano_dems = pickle.load(f)
    f.close()
    print('Done.  ')

except:
    print('Failed.  Generating them from scratch, which can be slow.  ')
    del SRTM_dem_settings['side_length']                                                                # this key is no longer needed, so delete.  
    volcano_dems = SRTM_dem_make_batch(volcanoes, **SRTM_dem_settings)                                  # make the DEMS
    with open(project_outdir / 'step_01_dem_data/volcano_dems.pkl', 'wb') as f:
        pickle.dump(volcano_dems, f)
    f.close()
    print('Saved the dems as a .pkl for future use.  ')



#%%  possibly use to edit synyhteic ifgs.  
# outdir = Path("/home/matthew/university_work/02_neural_network_python/03_VUDLNet_21_github_repo/02_full_size_example/step_02_synthetic_data_2")
# data_files = glob.glob(str(Path(project_outdir / "step_02_synthetic_data/*.pkl")))             #

# for data_file in data_files:
#         with open(data_file, 'rb') as f:                                                    # open the file
#             X = pickle.load(f)                                                              # and extract data (X) and labels (Y)
#             Y_class = pickle.load(f)
#             Y_loc = pickle.load(f)
            
        
#         with open(outdir / data_file.split('/')[-1], 'wb') as f:                                                    # open the file
#             pickle.dump(X[:,:,:,np.newaxis], f)
#             pickle.dump(Y_class, f)
#             pickle.dump(Y_loc, f)
    

# sys.exit()



#%% 2: Create or load the synthetic interferograms.  
print("\nStep 02: Creating or loading synthetic interferograms")

n_synth_data = ifg_settings['n_per_file'] * synthetic_ifgs_n_files

print('Determining if files containing the synthetic deformation patterns exist... ', end = '')
data_files = glob.glob(str(Path(project_outdir / "step_02_synthetic_data/*.pkl")))             #



if len(data_files) == synthetic_ifgs_n_files:
    print(f"The correct number of files were found ({synthetic_ifgs_n_files}) so no new ones will be generated.  "
          f"However, this doesn't guarantee that the files were made using the settings in synthetic_ifgs_settings.  Check synth_data_settings.txt to be sure.   ")
else:
    print(f"{len(data_files)} files were found, but {synthetic_ifgs_n_files} were requested.  "
          f"The directory containing these (./step_02_synthetic_data/)will be deleted, and the correct number of files generated.  ")
    #pdb.set_trace()
    answer = input("Do you wish to continue ('y' or 'n')?")
    if answer == 'n':
        sys.exit()
    elif answer == 'y':
        try:
            shutil.rmtree(str(project_outdir / "step_02_synthetic_data"))
        except:
            pass
        os.mkdir(project_outdir / "step_02_synthetic_data")
        for file_n in range(synthetic_ifgs_n_files):
            print(f"Generating file {file_n} of {synthetic_ifgs_n_files} files.  ")
            X_all, Y_class_categorical, Y_loc, Y_source_kwargs = create_random_synthetic_ifgs(volcano_dems, **synthetic_ifgs_settings)
            Y_class = np.zeros((Y_class_categorical.shape[0], len(synthetic_ifgs_settings['defo_sources'])), dtype='float32')                           # initiate to store one hot encodings.  
            for data_n, Y_class_one in enumerate(Y_class_categorical):                                                                                  # loop through categorical labels
                Y_class[data_n, int(Y_class_one[0])] = 1                                                                                                # and convert to one hot   
            with open(project_outdir / 'step_02_synthetic_data' / f'data_file_{file_n:05d}.pkl', 'wb') as f:
                pickle.dump(X_all[synthetic_ifgs_settings['outputs'][0]], f)                                                    # usual output style is many channel formats in a dict, but we are only interesetd in the one we generated.  
                pickle.dump(Y_class, f)
                pickle.dump(Y_loc, f)
            f.close()
            del X_all, Y_class, Y_loc
        with open(project_outdir / "step_02_synthetic_data" / "synth_data_settings.txt", 'w') as f:                              # output the settings as a text file so that we know how data were generated in the future.  
            print(f"Number of data per file : {ifg_settings['n_per_file']}" ,file = f)
            print(f"Number of files: {synthetic_ifgs_n_files}" ,file = f)
            for key in synthetic_ifgs_settings:
                print(f"{key} : {synthetic_ifgs_settings[key]}", file = f)
    else:
        print(f"Answer ({answer}) was not understood as either 'y' or 'n' so exiting to err on the side of caution")
        sys.exit()


open_datafile_and_plot(project_outdir / "step_02_synthetic_data" / "data_file_00000.pkl", n_data = 15, window_title ='01 Sample of synthetic data')                                        # open and plot the data in 1 file


#%%  Step 03: Compute the deformation labels, create the interferograms.   

print(f"\nStep 03: Creating labels for all possible VolcNet data.  ")

volcnet_files = sorted(glob.glob(dependency_paths['volcnet'] +  '/*.pkl'))            # get the paths to the volcnet file.  

# from operator import itemgetter
# print(f"only using some of the VolcNet files! ")
# volcnet_files = itemgetter(1,2,13)(volcnet_files)

labels_dyke, labels_sill, labels_atmo = label_volcnet_files(volcnet_files, def_min = volcnet_def_min)               # label all the VolcNet data - slow! 
plot_volcnet_files_labels(volcnet_files, labels_dyke, labels_sill, labels_atmo)                                     # figure with two subplots showing how many of each label etc.  
n_min_real = np.min([labels_dyke.shape[0], labels_sill.shape[0], labels_atmo.shape[0]])                             # get the minimum number of data in a class

np.random.shuffle(labels_dyke)                                                                                      # shuffle along first axis
np.random.shuffle(labels_sill)
np.random.shuffle(labels_atmo)

labels_dyke_crop = labels_dyke[:n_min_real, ]                                                                       # crop number of data so all the same size (the minimum)
labels_sill_crop = labels_sill[:n_min_real, ]
labels_atmo_crop = labels_atmo[:n_min_real, ]

plot_volcnet_files_labels(volcnet_files, labels_dyke_crop, labels_sill_crop, labels_atmo_crop)                      # same plot as before but for subset of data.  

labels_all = np.vstack((labels_dyke_crop, labels_sill_crop, labels_atmo_crop))                                      # combine to one array for the data that will be used (same number of data in each class)

args_sorted = np.argsort(labels_all[:,0])                                                                           # get the args to sort the first column (which is the volcnet file number)
labels_all = labels_all[args_sorted,]                                                                               # sort so that data are in order by VolcNet file.  

del labels_dyke, labels_sill, labels_atmo, n_min_real, labels_dyke_crop, labels_sill_crop, labels_atmo_crop, args_sorted


create_volcnet_ifgs(labels_all, volcnet_files, project_outdir / "step_03_labelled_volcnet_data" , n_data_per_file  = ifg_settings['n_per_file'],            # create the desired ifgs.  
                    ny = synthetic_ifgs_settings['n_pix'], nx = synthetic_ifgs_settings['n_pix'], volcnet_def_min = volcnet_def_min)

#open_datafile_and_plot(project_outdir / "step_03_labelled_volcnet_data" /  "data_file_unshuffled_00015.pkl", n_data = 15, window_title = '03 Sample of augmented real data')


unshuffled_files = sorted(glob.glob(str(project_outdir / "step_03_labelled_volcnet_data" / '*.pkl')))                                           # get the paths to the volcnet file.      
shuffle_data_pkls(unshuffled_files, file_batch_size, outdir = project_outdir / "step_03_labelled_volcnet_data" )                               # 2nd arg is number of files to be opened at once.  Big number needs lots of RAM
open_datafile_and_plot(project_outdir / "step_03_labelled_volcnet_data" /  "data_file_shuffled_00000.pkl", 
                       n_data = 45, window_title = '03 Sample of VolcNet data')        


# Separate some files for testing.  
shuffled_files = sorted(glob.glob(str(project_outdir / "step_03_labelled_volcnet_data" / '*.pkl')))            # get the paths to the  shuffled volcnet file.      
test_files = shuffled_files[-n_files_test:]
for test_file in test_files:
    parts = list(Path(test_file).parts)
    outfile = project_outdir / "step_03_labelled_volcnet_data_testing" / parts[-1]
    shutil.move(test_file, outfile)
open_datafile_and_plot(project_outdir / "step_03_labelled_volcnet_data_testing" /  "data_file_shuffled_00026.pkl", n_data = 180, window_title = '03 Sample of VolcNet testing data')        





#%% Step 04: Augment the real data.  

labelled_volcnet_files = sorted(glob.glob(str(project_outdir / "step_03_labelled_volcnet_data" /  f"*.pkl")))                                              # get the paths to each file of real data

file_n = 0
for labelled_volcnet_file in labelled_volcnet_files:
    
    print(f'    Opening and augmenting {labelled_volcnet_file} ... ', end = '')
    with open(labelled_volcnet_file, 'rb') as f:                                                      # open the real data file
        X = pickle.load(f)
        Y_class = pickle.load(f)
        Y_loc = pickle.load(f)
    
    X_aug, Y_class_aug, Y_loc_aug = augment_data(X, Y_class, Y_loc, n_data = X.shape[0] * augmentation_factor, rotate = False, translate=False)
    
    for i in range(augmentation_factor):
        with open(project_outdir / "step_04_augmented_labelled_volcnet_data" / f"data_file_shuffled_{file_n:05d}.pkl", 'wb') as f:                     # save the output as a pickle
            pickle.dump(X_aug[i * ifg_settings['n_per_file']: (i+1) * ifg_settings['n_per_file'], ], f)
            pickle.dump(Y_class_aug[i * ifg_settings['n_per_file']: (i+1) * ifg_settings['n_per_file'], ], f)
            pickle.dump(Y_loc_aug[i * ifg_settings['n_per_file']: (i+1) * ifg_settings['n_per_file'], ], f)
        file_n += 1
    print("done.  ")
    
open_datafile_and_plot(project_outdir / "step_04_augmented_labelled_volcnet_data" /  "data_file_shuffled_00000.pkl", n_data = 45, window_title = '04 Sample of augmented VolcNet data')        

#%% 5: Merge real and synthetic data, and rescale to desired range 



print("\nStep 04: Mergring the real and synthetic interferograms and rescaling to CNNs input range.")

synthetic_data_files = sorted(glob.glob(str(project_outdir / "step_02_synthetic_data" / f"*.pkl")))                       # get the paths to each file of synthetic data
real_data_files = sorted(glob.glob(str(project_outdir / "step_04_augmented_labelled_volcnet_data" /  f"*.pkl")))                                              # get the paths to each file of real data

if len(synthetic_data_files) > len(real_data_files):                                                                        # see if one is bigger than other
    synthetic_data_files = synthetic_data_files[:len(real_data_files)]                                                      # if so crop longest to make the same
else:
    real_data_files = real_data_files[:len(synthetic_data_files)]                                                           # or if other is longer.  

merge_and_rescale_data(synthetic_data_files, real_data_files, cnn_settings['input_range'], triplicate_channel = True)                                   # merge the real and synthetic data, and rescale it into the correct range for use with the CNN

open_datafile_and_plot(project_outdir / "step_05_merged_rescaled_data" / "data_file_00000.npz", n_data = 15, window_title = ' 05 Sample of merged and rescaled data')


#%% 5a: Rescale only the synthetic data

from vudlnet21.file_handling import rescale_data

synthetic_data_files = sorted(glob.glob(str(project_outdir / "step_02_synthetic_data" / f"*.pkl")))                       # get the paths to each file of synthetic data

rescale_data(synthetic_data_files, project_outdir / "step_05a_merged_rescaled_data_synth_only",
             cnn_settings['input_range'], triplicate_channel = True)
    
open_datafile_and_plot(project_outdir / "step_05a_merged_rescaled_data_synth_only" / "data_file_00000.npz", n_data = 15, window_title = ' 05 Sample of merged and rescaled data - synthetic only')    
    

