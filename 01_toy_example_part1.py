#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import numpy.ma as ma 
import pickle
import sys
import glob
import os
from pathlib import Path
import shutil

# import tensorflow as tf
# import tf.keras as keras
# from keras import backend as K
# from keras import losses, optimizers
# from keras.applications.vgg16 import VGG16
# from keras.utils.vis_utils import plot_model
# from keras.models import Model, load_model
# from keras.layers import Input
# K.clear_session()                                                                                               # makes nameing models easier 




#%% 0: Things to set

dependency_paths = {#'syinterferopy_bin' : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'syinterferopy_bin' : '/home/matthew/university_work/15_my_software_releases/SyInterferoPy-2.2.0/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    #'srtm_dem_tools_bin' : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/'}                                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools
                    'srtm_dem_tools_bin' : '/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-2.1.1/'}                                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools


project_outdir = Path('./01_toy_example')

#step 01 (creating DEMS):
SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                     'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                     'SRTM3_tiles_folder'       : './SRTM3/',                                   # folder for DEM tiles.  
                     'download'                 : True,                                         # If tile is not available locally, try to download it
                     'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                     'side_length'              : 40e3}                                         # the side length in metres of the DEM.  To allow for different crops of this, it should be somewhat bigger than 224 (the number of pixels) x 90 (pixel size) ~ 20e3
n_volcanoes_used = 512                                                                            # there are 512 volcanoes in the Smithsonian file, but to speed up making the DEMS not all of these need to be used.   

#step 02 (making synthetic interferograms):
ifg_settings            = {'n_per_file'         : 50}                                            # number of ifgs per data file.  
#ifg_settings            = {'n_per_file'         : 5}                                            # number of ifgs per data file.  
synthetic_ifgs_n_files  =  13                                                                    # numer of files of synthetic data


synthetic_ifgs_settings = {'defo_sources'           : ['dyke', 'sill', 'no_def'],               # deformation patterns that will be included in the dataset.  
                           'n_ifgs'                 : ifg_settings['n_per_file'],               # the number of synthetic interferograms to generate PER FILE
                           'n_pix'                  : 224,                                      # number of 3 arc second pixels (~90m) in x and y direction
                           'outputs'                : ['uuu'],                                  # channel outputs.  uuu = unwrapped across all 3
                           'intermediate_figure'    : False,                                    # if True, a figure showing the steps taken during creation of each ifg is displayed.  
                           'cov_coh_scale'          : 5000,                                     # The length scale of the incoherent areas, in meters.  A smaller value creates smaller patches, and a larger one creates larger pathces.  
                           'coh_threshold'          : 0.7,                                      # if 1, there are no areas of incoherence, if 0 all of ifg is incoherent.  
                           'min_deformation'        : 0.05,                                     # deformation pattern must have a signals of at least this many metres.  
                           'max_deformation'        : 0.25,                                     # deformation pattern must have a signal no bigger than this many metres.  
                           'snr_threshold'          : 2.0,                                      # signal to noise ratio (deformation vs turbulent and topo APS) to ensure that deformation is visible.  A lower value creates more subtle deformation signals.
                           'turb_aps_mean'          : 0.02,                                     # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)
                           'turb_aps_length'        : 5000}                                     # turbulent APS will be correlated on this length scale, in metres.  

#step 03 (load real data and augment):
VolcNet_path = Path('/home/matthew/university_work/02_neural_networks_python/08_VolcNet')
real_ifg_settings       = {'augmentation_factor' : 3}                                           # factor to auument by.  E.g. if set to 10 and there are 30 data, there will be 300 augmented data.  

# step 04 (merge synthetic and real, and rescale to desired range)
cnn_settings = {'input_range'       : {'min':-1, 'max':1}}

   
np.random.seed(0)                                                                                           # 0 used in the example
              
#%% Import dependencies (paths set above)

sys.path.append(dependency_paths['syinterferopy_bin'])
sys.path.append(dependency_paths['srtm_dem_tools_bin'])

from dem_tools_lib import SRTM_dem_make_batch                                       # From SRTM dem tools
from random_generation_functions import create_random_synthetic_ifgs                # From SyInterferoPy

from vudlnet21.neural_net import open_VolcNet_file, augment_data, choose_for_augmentation, merge_and_rescale_data

from vudlnet21.plotting import plot_data_class_loc_caller, open_datafile_and_plot



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

VolcNet_files = sorted(glob.glob(str(VolcNet_path / '*.pkl')))             #  get a list of the paths to all the VolcNet files       
if len(VolcNet_files) == 0:
    raise Exception('No VolcNet files have been found.  Perhaps the path is wrong? Or perhaps you only want to use synthetic data?  In which case, this section can be removed.  Exiting...')

X_1s = []
Y_class_1s = []
Y_loc_1s = []
for VolcNet_file in VolcNet_files:
    X_1, Y_class_1, Y_loc_1 = open_VolcNet_file(VolcNet_file, synthetic_ifgs_settings['defo_sources'])
    X_1s.append(X_1)
    Y_class_1s.append(Y_class_1)
    Y_loc_1s.append(Y_loc_1)
X = ma.concatenate(X_1s, axis = 0)
Y_class = np.concatenate(Y_class_1s, axis = 0)
Y_loc = np.concatenate(Y_loc_1s, axis = 0)
del X_1s, Y_class_1s, Y_loc_1s, X_1, Y_class_1, Y_loc_1
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

synthetic_data_files = glob.glob(str(project_outdir / "step_02_synthetic_data" / "*.pkl"))                       # get the paths to each file of synthetic data
real_data_files = glob.glob(str(project_outdir / "step_03_real_data" / "augmented" / "*.pkl"))                                              # get the paths to each file of real data
merge_and_rescale_data(synthetic_data_files, real_data_files, project_outdir, cnn_settings['input_range'])                                   # merge the real and synthetic data, and rescale it into the correct range for use with the CNN

open_datafile_and_plot(project_outdir / "step_04_merged_rescaled_data" / "data_file_00000.npz", n_data = 15, window_title = ' 04 Sample of merged and rescaled data')
