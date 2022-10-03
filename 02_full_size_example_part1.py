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

def initialise_arrays(n_data, ny, nx,n_channels):
    """
    """
    X = ma.zeros((n_data, ny, nx, n_channels))               # initialise, rank 4 ready for Tensorflow, last dimension is being used for different crops.  
    Y_class = np.zeros((n_data, 3))                                                                             # initialise, doesn't need another dim as label is the same regardless of the crop.  
    Y_loc = np.zeros((n_data, 4, 9))                                                                            # initialise
    return X, Y_class, Y_loc



#%% MEG debug import

sys.path.append("/home/matthew/university_work/python_stuff/python_scripts")
from small_plot_functions import matrix_show, quick_linegraph
import matplotlib.pyplot as plt




#%% 0: Things to set

dependency_paths = {#'syinterferopy' : '/home/matthew/university_work/15_my_software_releases/SyInterferoPy-3.0.1/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'syinterferopy' : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy',

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
random_no_def_accept = 0.1                                                                # this fraction of no deforamtion ifgs will be accepted.  
    #real_ifg_settings       = {'augmentation_factor' : 116}                                           # factor to augment by.  E.g. if set to 10 and there are 30 data, there will be 300 augmented data.
                                                                                                   # v2 data has ? ifgs, so ? x 89 ~ 20000 etc.  

# step 04 (merge synthetic and real, and rescale to desired range)
cnn_settings = {'input_range'       : {'min':-1, 'max':1}}



print(f"\n\nSetting a random seed\n\n")   
np.random.seed(0)                                                                                           # 0 used in the example



              
#%% Import dependencies (paths set above)

from vudlnet21.file_handling import open_smithsonian_csv_file, open_all_volcnet_files
#from vudlnet21.neural_net import augment_data, choose_for_augmentation, merge_and_rescale_data, define_two_head_model, file_list_divider, file_merger
from vudlnet21.plotting import plot_data_class_loc_caller, open_datafile_and_plot
from vudlnet21.augmentation import rescale_timeseries, random_cropping

sys.path.append(dependency_paths['srtm_dem_tools'])
from dem_tools_lib import SRTM_dem_make_batch                                       # From SRTM dem tools

sys.path.append(dependency_paths['syinterferopy'])
import syinterferopy
from syinterferopy.random_generation import create_random_synthetic_ifgs                # From SyInterferoPy

sys.path.append(dependency_paths['volcnet'])
import volcnet
#from volcnet.plotting import volcnet_ts_visualiser
from volcnet.labelling import label_volcnet_files, label_volcnet_ifg
from volcnet.plotting import plot_volcnet_files_labels
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




#%%  4: Compute the deformation labels.  



print(f"\nStep 03: Creating labels for all possible VolcNet data.  ")


volcnet_files = sorted(glob.glob(dependency_paths['volcnet'] +  '/*.pkl'))            # get the paths to the volcnet file.  

print(f"only using first 4 files! ")
volcnet_files = volcnet_files[:4]

labels_dyke, labels_sill, labels_atmo = label_volcnet_files(volcnet_files, def_min = 0.05)

plot_volcnet_files_labels(volcnet_files, labels_dyke, labels_sill, labels_atmo)



#%%


#%% quick plot.  

sys.exit()

for key, value in data_indexs.items():
    f, axes = plt.subplots(1,2, figsize = (16,8))
    f.suptitle(key)
    matrix_show(value[0], ax = axes[0], fig = f)
    matrix_show(value[1], ax = axes[1], fig = f)
    axes[0].set_title('def_magnitudes')
    axes[1].set_title('labels')
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')




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

data_indexs = []                                                                                                                 # initialise
X, Y_class, Y_loc = initialise_arrays(ifg_settings['n_per_file'], synthetic_ifgs_settings['n_pix'], synthetic_ifgs_settings['n_pix'], 9)                # initiliase for next file.      

for volcnet_file in volcnet_files:
# for volcnet_file in volcnet_files[11:12]:
#     print("TESTING - only using Sierra Negra 128 volcnet file.  ")
# for volcnet_file in volcnet_files[10:]:
#     print("TESTING - Using only some volcnet files.  ")
    

    
    print(f"Opening file: {volcnet_file.split('/')[-1]}")
    # 1: Open the file
    with open(volcnet_file, 'rb') as f:
        displacement_r3 = pickle.load(f)
        tbaseline_info = pickle.load(f)
        persistent_defs = pickle.load(f)
        transient_defs = pickle.load(f)

    n_acq, ny, nx = displacement_r3['cumulative'].shape
    n_ifg = (n_acq*n_acq) - n_acq
    print(f"The interferograms are of size: {displacement_r3['mask'].shape}")
   
    if (nx < synthetic_ifgs_settings['n_pix']) or (ny < synthetic_ifgs_settings['n_pix']):
        if ny/nx < 1:                                                                           # this is less than 1 if the image is wider than tall.  
            rescale_factor = (1.4 * synthetic_ifgs_settings['n_pix']) /ny                                                           #rescale to ensure y is large enough to be cropped down to 224    
        else:
            rescale_factor = (1.4 * synthetic_ifgs_settings['n_pix']) / nx
            
        displacement_r3 = rescale_timeseries(displacement_r3, rescale_factor)
        print(f"The interferograms have been interpolated to size: {displacement_r3['mask'].shape}")
        
        
    
    
    for acq_n1, acq_1 in enumerate(tbaseline_info['acq_dates']):                                                # 
        for acq_n2, acq_2 in enumerate(tbaseline_info['acq_dates']):
            if acq_1 == acq_2:                                                                                                          # will just be zeros so ignore.   
                pass
            else:
                ifg = displacement_r3['cumulative'][acq_n2,] - displacement_r3['cumulative'][acq_n1,]                                    # make the ifg between the two acquisitions.  
                def_predicted, sources, def_location = volcnet_labeller(f"{acq_1}_{acq_2}", persistent_defs, transient_defs)             # label the ifg, def_location is still in terms of lon and lat
                def_loc_pixels = ll_2_pixel(def_location, displacement_r3['lons'], displacement_r3['lats'])                             # convert the location label from lon lat to pixels (x then y)
                
                if (np.abs(def_predicted) < def_min):
                    if (np.random.rand() < random_no_def_accept):                                                                                              # if it doesn't have a deformation signal we're classing as visible.  
                        X[data_n,] = random_cropping(ifg, synthetic_ifgs_settings['n_pix'], None)    
                        data_n += 1
                        Y_class[data_n,] = np.array([0,0,1])                                                                                 # this is the one hot encoding for no deformation.  
                        Y_loc[data_n,] = np.repeat(np.array([0,0,0,0])[:,np.newaxis], axis = -1, repeats = 9)                                                                                 # no def has no location.  
                        data_indexs.append((file_n, data_n, 2))
                        print(f"file: {file_n} {data_n}: {acq_1} and {acq_2} : no deformation.  ")
                
                elif (np.abs(def_predicted) < def_max) or (np.abs(def_predicted) > def_max and np.random.rand() < random_large_threshold):
                    ifg_cropped, Y_loc_cropped = random_cropping(ifg, synthetic_ifgs_settings['n_pix'], def_loc_pixels)    
                    # from vudlnet21.aux import add_square_plot
                    # f, axes = plt.subplots(1,9)
                    # for i in range(9):
                    #     axes[i].imshow(ifg_cropped[:,:,i])
                    #     add_square_plot(Y_loc[i, 0] - Y_loc[i, 2], Y_loc[i, 0] + Y_loc[i, 2],
                    #                     Y_loc[i, 1] - Y_loc[i, 3], Y_loc[i, 1] + Y_loc[i, 3], axes[i], colour = 'k')
                    X[data_n,] = ifg_cropped
                    Y_loc[data_n] = Y_loc_cropped.T
                    
                    data_n += 1
                    if sources[0] == 'dyke':
                        Y_class[data_n,] = np.array([1,0,0])                                                                             # this is the one hot encoding for dyke
                        data_indexs.append((file_n, data_n, 0))
                        print(f"file: {file_n} {data_n}: {acq_1} and {acq_2} : dyke ({def_predicted:.2f} m of deformation).  ")
                    elif sources[0] == 'sill':
                        Y_class[data_n,] = np.array([0,1,0])                                                                             # this is the one hot encoding for sill
                        data_indexs.append((file_n, data_n, 1))
                        print(f"file: {file_n} {data_n}: {acq_1} and {acq_2} : sill ({def_predicted:.2f} m of deformation).  ")
                        
                # When generated the required number per file, save the file.  
                if data_n == (ifg_settings['n_per_file'] - 1):                         
                    print(f"    Saving file {file_n}")
                    with open(project_outdir / "step_03_labelled_volcnet_data" / f"data_file_{file_n:05d}.pkl", 'wb') as f:                     # save the output as a pickle
                        pickle.dump(X, f)
                        pickle.dump(Y_class, f)
                        pickle.dump(Y_loc, f)
                    file_n += 1                                                                                                                 # advance to next file
                    data_n = 0                                                                                                                  # initiate for next file        
                    X, Y_class, Y_loc = initialise_arrays(ifg_settings['n_per_file'], 
                                                          synthetic_ifgs_settings['n_pix'], synthetic_ifgs_settings['n_pix'], 9)                # initiliase for next file.  

if data_n != (ifg_settings['n_per_file'] - 1):                         
    print(f"    Saving file {file_n} (part complete final file)")
    with open(project_outdir / "step_03_labelled_volcnet_data" / f"data_file_{file_n:05d}.pkl", 'wb') as f:                     # save the output as a pickle
        pickle.dump(X[:data_n, ], f)                        # crop in first dim
        pickle.dump(Y_class[:data_n, ], f)
        pickle.dump(Y_loc[:data_n, ], f)        

                

# convert the data index from a list of tuples to an array.  
data_index = np.zeros((len(data_indexs), 3))                                                # 0 for dyke, 1 for sill, 2 for no def.  
for n, data_index_n in enumerate(data_indexs):
    data_index[n, :] = np.array([data_index_n[0], data_index_n[1], data_index_n[2]])


sys.exit()


#%% Real data balance classes, merge, shuffle.  


#%%

from vudlnet21.plotting import plot_data_class_loc_caller, open_datafile_and_plot
open_datafile_and_plot(project_outdir / "step_03_labelled_volcnet_data" /  "data_file_00000.pkl", n_data = 15, window_title = '03 Sample of augmented real data')
            
sys.exit()



#%% 4: Merge real and synthetic data, and rescale to desired range (e.g. [0, 1], [0, 255], [-125, 125] etc)

print("\nStep 04: Mergring the real and synthetic interferograms and rescaling to CNNs input range.")

synthetic_data_files = glob.glob(str(project_outdir / "step_02_synthetic_data" / f"*.pkl"))                       # get the paths to each file of synthetic data
real_data_files = glob.glob(str(project_outdir / "step_03_real_data" / "augmented" / f"*.pkl"))                                              # get the paths to each file of real data
merge_and_rescale_data(synthetic_data_files, real_data_files, project_outdir, cnn_settings['input_range'])                                   # merge the real and synthetic data, and rescale it into the correct range for use with the CNN

open_datafile_and_plot(project_outdir / "step_04_merged_rescaled_data" / "data_file_00000.npz", n_data = 15, window_title = ' 04 Sample of merged and rescaled data')
