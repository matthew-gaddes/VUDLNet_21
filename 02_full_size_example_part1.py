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
import sys
sys.path.append("/home/matthew/university_work/python_stuff/python_scripts")
from small_plot_functions import matrix_show, quick_linegraph
import matplotlib.pyplot as plt




#%% 0: Things to set

dependency_paths = {'syinterferopy' : '/home/matthew/university_work/15_my_software_releases/SyInterferoPy-3.0.1/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    #'syinterferopy' : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy',

                    'srtm_dem_tools': '/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-2.1.1/',                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools
                    'volcnet'       : '/home/matthew/university_work/13_volcnet'}                                                   # Available from Github: https://github.com/matthew-gaddes/VolcNet


project_outdir = Path('./')

#step 01 (creating DEMS):
SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                     'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                     'SRTM3_tiles_folder'       : './SRTM3/',                                   # folder for DEM tiles.  
                     'download'                 : True,                                         # If tile is not available locally, try to download it
                     'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                     'side_length'              : 40e3}                                         # the side length in metres of the DEM.  To allow for different crops of this, it should be somewhat bigger than 224 (the number of pixels) x 90 (pixel size) ~ 20e3
n_volcanoes_used = 512                                                                            # there are 512 volcanoes in the Smithsonian file, but to speed up making the DEMS not all of these need to be used.   

#step 02 (making synthetic interferograms):
# test
# ifg_settings            = {'n_per_file'         : 10}                                            # number of ifgs per data file.  
# synthetic_ifgs_n_files  =  4                                                                   # numer of files of synthetic data

# real - 20,000 data    
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
def_min = 0.05                                                                            # in m, real ifg must have a signal bigger than this to be classed as deformation.  
def_max = 0.5
random_large_threshold = 0.1                                                                # this fraction of ifgs with deformation over the max are still included.  

    #real_ifg_settings       = {'augmentation_factor' : 116}                                           # factor to augment by.  E.g. if set to 10 and there are 30 data, there will be 300 augmented data.
                                                                                                   # v2 data has ? ifgs, so ? x 89 ~ 20000 etc.  

# step 04 (merge synthetic and real, and rescale to desired range)
cnn_settings = {'input_range'       : {'min':-1, 'max':1}}

   
np.random.seed(0)                                                                                           # 0 used in the example
              
#%% Import dependencies (paths set above)

from vudlnet21.file_handling import open_smithsonian_csv_file, open_all_volcnet_files
#from vudlnet21.neural_net import augment_data, choose_for_augmentation, merge_and_rescale_data, define_two_head_model, file_list_divider, file_merger
from vudlnet21.plotting import plot_data_class_loc_caller, open_datafile_and_plot

sys.path.append(dependency_paths['srtm_dem_tools'])
from dem_tools_lib import SRTM_dem_make_batch                                       # From SRTM dem tools

sys.path.append(dependency_paths['syinterferopy'])
import syinterferopy
from syinterferopy.random_generation import create_random_synthetic_ifgs                # From SyInterferoPy

sys.path.append(dependency_paths['volcnet'])
import volcnet
#from volcnet.plotting import volcnet_ts_visualiser
from volcnet.labelling import volcnet_labeller
from volcnet.aux import ll_2_pixel



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





#%% 3: Load the real data (and augment).  Note that these are in metres, and use one hot encoding for the class, and are masked arrays (incoherence and water are masked)
print("\nStep 03: Loading and augmenting the real interferograms from VolcNet.  ")
print("    Starting to open the real data....", end = '')



volcnet_files = sorted(glob.glob(dependency_paths['volcnet'] +  '/*.pkl'))            # get the paths to the volcnet file.  


# for volcnet_file in volcnet_files[1:2]:
#     print("TESTING - only using Campi Flegrei volcnet file.  ")
# for volcnet_file in volcnet_files[11:12]:
#     print("TESTING - only using Wolf volcnet file.  ")
file_n = 0
data_n = 0

X = ma.zeros((ifg_settings['n_per_file'], synthetic_ifgs_settings['n_pix'], synthetic_ifgs_settings['n_pix'], 1))        # initiate, rank 4 ready for Tensorflow.  
Y_class = np.zeros((ifg_settings['n_per_file'], 3))
Y_loc = np.zeros((ifg_settings['n_per_file'], 4))

for volcnet_file in volcnet_files[11:12]:
    print("TESTING - only using Sierra Negra 128 volcnet file.  ")
    
    # 1: Open the file
    with open(volcnet_file, 'rb') as f:
        displacement_r3 = pickle.load(f)
        tbaseline_info = pickle.load(f)
        persistent_defs = pickle.load(f)
        transient_defs = pickle.load(f)

    n_acq = len(tbaseline_info['acq_dates'])
    n_ifg = (n_acq*n_acq) - n_acq

    
    for acq_n1, acq_1 in enumerate(tbaseline_info['acq_dates']):                                                # 
        for acq_n2, acq_2 in enumerate(tbaseline_info['acq_dates']):
            if acq_1 == acq_2:                                                                                      # will just be zeros.  
                pass
            else:
                ifg = displacement_r3['cumulative'][acq_n2,] - displacement_r3['cumulative'][acq_n1,]                                               # make the ifg
                def_predicted, sources, def_location = volcnet_labeller(f"{acq_1}_{acq_2}", persistent_defs, transient_defs)                      # label the ifg
                
                
                
                if def_predicted < def_min:                                                                                     # if it doesn't have a deformation signal we're classing as visible.  
                    Y_class[data_n,] = np.array([0,0,1])                                                                        # this is the one hot encoding for no deformation.  
                    Y_loc[data_n,] = np.array([0,0,0,0])                                                                        # no def has no location.  
                    viable = True
                
                elif (def_predicted < def_max) or (def_predicted > def_max and np.random.rand() < random_large_threshold):
                    if sources[0] == 'dyke':
                        Y_class[data_n,] = np.array([1,0,0])                                                                        # this is the one hot encoding for dyke
                    elif sources[0] == 'sill':
                        Y_class[data_n,] = np.array([0,1,0])                                                                        # this is the one hot encoding for sill
                    viable = True
                    
                if viable:
                    # crop the ifg to the required size and add to X
                    pdb.set_trace()
                    ny, nx = ifg.shape
                    start_y = int((ny -  synthetic_ifgs_settings['n_pix']) / 2 )
                    start_x = int((ny -  synthetic_ifgs_settings['n_pix']) / 2 )            
                    X[data_n,] = ifg[start_y : (start_y + synthetic_ifgs_settings['n_pix']),
                                        start_x : (start_x + synthetic_ifgs_settings['n_pix']), np.newaxis]                                                               # make have one channel
                

                else:

                        
                    
                    loc = ll_2_pixel(def_location, displacement_r3['lons'], displacement_r3['lats'])                           # convert lon and lat of deformation to pixel number
                    x_centre = int(np.mean([np.max(loc[:,0]), np.min(loc[:,0])])) - start_x                                     # subtract start_x to incorporate cropping that was done before
                    x_half_width = int((np.max(loc[:,0]) - np.min(loc[:,0]))/2) 
                    y_centre = int(np.mean([np.max(loc[:,1]), np.min(loc[:,1])])) - start_y                                      # and the same for y
                    y_half_width = int((np.max(loc[:,1]) - np.min(loc[:,1]))/2)
                    Y_loc[data_n,] = np.array([x_centre, y_centre, x_half_width, y_half_width])                                                      

                    

                data_n += 1

                # When generated the required number per file, save the file.  
                if data_n == (ifg_settings['n_per_file'] - 0):                         
                    print(f"    Saving file {file_n}")
                    with open(project_outdir / "step_03_labelled_volcnet_data" / f"data_file_{file_n:05d}.pkl", 'wb') as f:                                        # save the output as a pickle
                        pickle.dump(X, f)
                        pickle.dump(Y_class, f)
                        pickle.dump(Y_loc, f)
                    file_n += 1                                                                                                                 # advance to next file
                    data_n = 0                                                                                                                  # initiate for next file        
                    X = ma.zeros((ifg_settings['n_per_file'], synthetic_ifgs_settings['n_pix'], synthetic_ifgs_settings['n_pix'], 1))        # initiate for next file
                    Y_class = np.zeros((ifg_settings['n_per_file'], 3))
                    Y_loc = np.zeros((ifg_settings['n_per_file'], 4))
                


#%%

from vudlnet21.plotting import plot_data_class_loc_caller, open_datafile_and_plot
open_datafile_and_plot(project_outdir / "step_03_labelled_volcnet_data" /  "data_file_00000.pkl", n_data = 15, window_title = '03 Sample of augmented real data')
            
sys.exit()



#%%



if volcnet_version == 'v1':
    with open(volcnet_dir / "v1_manually_split" / "training_data.pkl", 'rb') as f:                     # open only the train data
        data_dict = pickle.load(f)                                                                     # 
    X = data_dict['X_m']                                                                                # this is a masked array of ifgs, units are metres
    Y_class = data_dict['Y_class']
    Y_loc = data_dict['Y_loc']

elif volcnet_version == 'v2':
    X, Y_class, Y_loc = open_all_volcnet_files(volcnet_dir / "v2_all_data", synthetic_ifgs_settings['defo_sources'])
else:
    raise Exception("There are currently only versions 'v1' and 'v2' of volcnet, but you have requested {volcnet_version} so exiting.  ")


plot_data_class_loc_caller(X[:30,], Y_class[:30,], Y_loc[:30,], source_names = ['dyke', 'sill', 'no def'], window_title = '02 Sample of Real data')         # plot the data in it (note that this can be across multiople windows)        
print('Done.  ')


n_augmented_files = int((X.shape[0] * real_ifg_settings['augmentation_factor']) / ifg_settings['n_per_file'])                   # detemine how many files will be needed, given the agumentation factor.  

print('    Determining if files containing the augmented real data exist.')
real_augmented_files = glob.glob(str(project_outdir / 'step_03_real_data' / 'augmented' / "*.pkl"))             #
if len(real_augmented_files) == n_augmented_files:
    print(f"    The correct number of augmented real data files were found ({n_augmented_files}) so no new ones will be generated.  "
          f"However, this doesn't guarantee that the files were made using the current real data.  ")
else:
    print(f"    {len(real_augmented_files)} files were found, but {n_augmented_files} were requested.  "
          f"The folder containing these (./step_03_real_data/augmented) will be deleted, and the correct number of files generated.  ")
    answer = input("Do you wish to continue ('y' or 'n')?")
    if answer == 'n':
        sys.exit()
    elif answer == 'y':
        try:
            shutil.rmtree(str(project_outdir / "step_03_real_data" / "augmented"))
        except:
            pass
        os.mkdir((project_outdir / "step_03_real_data" /"augmented"))

        print(f"    There are {X.shape[0]} real data and the augmentation factor is set to {real_ifg_settings['augmentation_factor']}.  ")
        print(f"    With {ifg_settings['n_per_file']} data per file, the nearest integer number of files is {n_augmented_files}.  ")
        for n_augmented_file in range(n_augmented_files):                                                                               # loop through each file that is to be made
            print(f'    File {n_augmented_file} of {n_augmented_files}...', end = '')  
            X_sample, Y_class_sample, Y_loc_sample = choose_for_augmentation(X, Y_class, Y_loc,                                         # make a new selection of the data with balanced classes
                                                                              n_per_class = int(X.shape[0] / Y_class.shape[1]))          # set it so there are as many per class as there are (on average) for the real data.  
            X_aug, Y_class_aug, Y_loc_aug = augment_data(X_sample, Y_class_sample, Y_loc_sample,                                        # augment the sample of real data
                                                          n_data = ifg_settings['n_per_file'])                                           # make as many new data as are set to be in a single file.  
        
            with open(project_outdir / "step_03_real_data" / "augmented" / f"data_file_{n_augmented_file:05d}.pkl", 'wb') as f:                                        # save the output as a pickle
                pickle.dump(X_aug, f)
                pickle.dump(Y_class_aug, f)
                pickle.dump(Y_loc_aug, f)
            f.close()
            print('Done!')
        print('Done!')

open_datafile_and_plot(project_outdir / "step_03_real_data" / "augmented" / "data_file_00000.pkl", n_data = 15, window_title = '03 Sample of augmented real data')



#%% 4: Merge real and synthetic data, and rescale to desired range (e.g. [0, 1], [0, 255], [-125, 125] etc)

print("\nStep 04: Mergring the real and synthetic interferograms and rescaling to CNNs input range.")

synthetic_data_files = glob.glob(str(project_outdir / "step_02_synthetic_data" / f"*.pkl"))                       # get the paths to each file of synthetic data
real_data_files = glob.glob(str(project_outdir / "step_03_real_data" / "augmented" / f"*.pkl"))                                              # get the paths to each file of real data
merge_and_rescale_data(synthetic_data_files, real_data_files, project_outdir, cnn_settings['input_range'])                                   # merge the real and synthetic data, and rescale it into the correct range for use with the CNN

open_datafile_and_plot(project_outdir / "step_04_merged_rescaled_data" / "data_file_00000.npz", n_data = 15, window_title = ' 04 Sample of merged and rescaled data')
