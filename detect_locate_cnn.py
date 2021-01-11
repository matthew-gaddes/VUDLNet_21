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
import numpy.ma as ma 
import pickle
import sys
import glob
import os
from pathlib import Path
import shutil
import keras
from keras import backend as K
from keras import losses, optimizers
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
K.clear_session()                                                                                               # makes nameing models easier 




#%% 0: Things to set

dependency_paths = {'syinterferopy_bin' : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'srtm_dem_tools_bin' : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/'}                                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools
#step 01 (creating DEMS):
SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                     'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                     'SRTM3_tiles_folder'       : './SRTM3/',                                   # folder for DEM tiles.  
                     'download'                 : True,                                         # If tile is not available locally, try to download it
                     'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                     'side_length'              : 40e3}                                         # the side length in metres of the DEM.  To allow for different crops of this, it should be somewhat bigger than 224 (the number of pixels) x 90 (pixel size) ~ 20e3
n_volcanoes_used = 5                                                                            # there are 512 volcanoes in the Smithsonian file, but to speed up making the DEMS not all of these need to be used.   

#step 02 (making synthetic interferograms):
ifg_settings            = {'n_per_file'         : 50}                                            # number of ifgs per data file.  
synthetic_ifgs_n_files  =  13                                                                    # numer of files of synthetic data
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

#step 03 (load real data and augment):
VolcNet_path = Path('/home/matthew/university_work/02_neural_networks_python/08_VolcNet')
real_ifg_settings       = {'augmentation_factor' : 3}                                           # factor to auument by.  E.g. if set to 10 and there are 30 data, there will be 300 augmented data.  

# step 04 (merge synthetic and real, and rescale to desired range)
cnn_settings = {'input_range'       : {'min':0, 'max':255}}

# step 05 (compute bottleneck features): 
    
# step 06 (train the fully connected part of the network)
cnn_settings['n_files_train']     = 22                                              # the number of files that will be used to train the network
cnn_settings['n_files_validate']  = 2                                               # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
cnn_settings['n_files_test']      = 2                                               # the number of files held back for testing.  
fc_loss_weights = [0.05, 0.95]                                                      # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
n_epochs_fc = 10                                                                    # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)

# step 07 (fine-tune the 5th block and the fully connected part of the network):
block5_loss_weights = [0.05, 0.95]                                                  # as per fc_loss_weights, but by changing these more emphasis can be placed on either the clasification or localisation loss.  
#block5_lr = 1.5e-8                                                                  # a pretty important parameter.  We have to set a learning rate manually as an adaptive approach (e.g. NADAM) will be high initially, and therefore make large updates that will wreck the model (as we're just fine-tuning a model so have something good to start with)
block5_lr = 1.0e-6                                                                  # a pretty important parameter.  We have to set a learning rate manually as an adaptive approach (e.g. NADAM) will be high initially, and therefore make large updates that will wreck the model (as we're just fine-tuning a model so have something good to start with)
n_epochs_block5 = 10                                                                 # the number of epochs to fine-tune for (ie. the number of times all the training data are passed through the model)

   
np.random.seed(0)                                                                                           # 0 used in the example
              
#%% Import dependencies (paths set above)

sys.path.append(dependency_paths['syinterferopy_bin'])
sys.path.append(dependency_paths['srtm_dem_tools_bin'])

from dem_tools_lib import SRTM_dem_make_batch                                       # From SRTM dem tools
from random_generation_functions import create_random_synthetic_ifgs                # From SyInterferoPy

from detect_locate_nn_functions import augment_data, choose_for_augmentation, merge_and_rescale_data, train_double_network, define_two_head_model, file_list_divider, file_merger, open_VolcNet_file
from detect_locate_plotting_functions import plot_data_class_loc_caller, open_datafile_and_plot, custom_training_history                                



#%% 1: Create or load DEMs for the volcanoes to be used for synthetic data.  
print("\nStep 01: Creating or loadings DEMs")
volcanoes = open_smithsonian_csv_file('./step_01_dem_data/smithsonian_name_lat_lon.csv', side_length=SRTM_dem_settings['side_length'])          # conver the csv file into a list with a dict for each volcanoe

print(f"{len(volcanoes)} volcanoes have been loaded from the .csv file from the Smithsonian, "                                                  # update terminal
      f"and the first {n_volcanoes_used} are being used.  ")
volcanoes = volcanoes[:n_volcanoes_used]                                                                                                        # crop number of volcanoes as required.  

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
    print(f"The correct number of files were found ({synthetic_ifgs_n_files}) so no new ones will be generated.  "
          f"However, this doesn't guarantee that the files were made using the settings in synthetic_ifgs_settings.  Check synth_data_settings.txt to be sure.   ")
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
            print(f"Generating file {file_n} of {synthetic_ifgs_n_files} files.  ")
            X_all, Y_class, Y_loc, Y_source_kwargs = create_random_synthetic_ifgs(volcano_dems, **synthetic_ifgs_settings)
            Y_class = keras.utils.to_categorical(Y_class, len(synthetic_ifgs_settings['defo_sources']), dtype='float32')          # convert to one hot encoding (from class labels)
            with open(Path(f'./step_02_synthetic_data/{synthetic_ifgs_folder}/data_file_{file_n}.pkl'), 'wb') as f:
                pickle.dump(X_all[synthetic_ifgs_settings['outputs'][0]], f)                                                    # usual output style is many channel formats in a dict, but we are only interesetd in the one we generated.  
                pickle.dump(Y_class, f)
                pickle.dump(Y_loc, f)
            f.close()
            del X_all, Y_class, Y_loc
        with open(f"./step_02_synthetic_data/{synthetic_ifgs_folder}/synth_data_settings.txt", 'w') as f:                              # output the settings as a text file so that we know how data were generated in the future.  
            print(f"Number of data per file : {ifg_settings['n_per_file']}" ,file = f)
            print(f"Number of files: {synthetic_ifgs_n_files}" ,file = f)
            for key in synthetic_ifgs_settings:
                print(f"{key} : {synthetic_ifgs_settings[key]}", file = f)
    else:
        print(f"Answer ({answer}) was not understood as either 'y' or 'n' so exiting to err on the side of caution")
        sys.exit()


open_datafile_and_plot(f"step_02_synthetic_data/{synthetic_ifgs_folder}/data_file_0.pkl", n_data = 15, window_title ='01 Sample of synthetic data')                                        # open and plot the data in 1 file



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
real_augmented_files = glob.glob(str(Path(f"./step_03_real_data/augmented/*.pkl")))             #
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
            shutil.rmtree(str(Path(f"./step_03_real_data/augmented/")))
        except:
            pass
        os.mkdir((Path(f"./step_03_real_data/augmented/")))

        print(f"    There are {X.shape[0]} real data and the augmentation factor is set to {real_ifg_settings['augmentation_factor']}.  ")
        print(f"    With {ifg_settings['n_per_file']} data per file, the nearest integer number of files is {n_augmented_files}.  ")
        for n_augmented_file in range(n_augmented_files):                                                                               # loop through each file that is to be made
            print(f'    File {n_augmented_file} of {n_augmented_files}...', end = '')  
            X_sample, Y_class_sample, Y_loc_sample = choose_for_augmentation(X, Y_class, Y_loc,                                         # make a new selection of the data with balanced classes
                                                                              n_per_class = int(X.shape[0] / Y_class.shape[1]))          # set it so there are as many per class as there are (on average) for the real data.  
            X_aug, Y_class_aug, Y_loc_aug = augment_data(X_sample, Y_class_sample, Y_loc_sample,                                        # augment the sample of real data
                                                          n_data = ifg_settings['n_per_file'])                                           # make as many new data as are set to be in a single file.  
        
            with open(f"./step_03_real_data/augmented/data_file_{n_augmented_file}.pkl", 'wb') as f:                                        # save the output as a pickle
                pickle.dump(X_aug, f)
                pickle.dump(Y_class_aug, f)
                pickle.dump(Y_loc_aug, f)
            f.close()
            print('Done!')
        print('Done!')

open_datafile_and_plot("./step_03_real_data/augmented/data_file_0.pkl", n_data = 15, window_title = '03 Sample of augmented real data')



#%% 4: Merge real and synthetic data, and rescale to desired range (e.g. [0, 1], [0, 255], [-125, 125] etc)

from detect_locate_nn_functions import merge_and_rescale_data

print("\nStep 04: Mergring the real and synthetic interferograms and rescaling to CNNs input range.")

synthetic_data_files = glob.glob(str(Path(f"./step_02_synthetic_data/{synthetic_ifgs_folder}/*.pkl")))                       # get the paths to each file of synthetic data
real_data_files = glob.glob(str(Path(f"./step_03_real_data/augmented//*.pkl")))                                              # get the paths to each file of real data
merge_and_rescale_data(synthetic_data_files, real_data_files, cnn_settings['input_range'])                                   # merge the real and synthetic data, and rescale it into the correct range for use with the CNN

open_datafile_and_plot("./step_04_merged_rescaled_data/data_file_0.npz", n_data = 15, window_title = ' 04 Sample of merged and rescaled data')

#%% 5: Compute bottlenceck features

print("\nStep 05: Computing the bottleneck features.")
print('commented for speed')
vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))          # load the first 5 (convolutional) blocks of VGG16 and their weights.  
data_out_files = sorted(glob.glob(f'step_04_merged_rescaled_data/*.npz'))                           # get a list of the files output by step 05 (augmented real data and synthetic data mixed and rescaed to correct range, with 0s for masked areas.  )

for file_n, data_out_file in enumerate(data_out_files):                                             # loop through each of the step 05 files.  
    print(f'Bottlneck file {file_n}:')    
    data_out_file = Path(data_out_file)                                                                                     # convert to path 
    bottleneck_file_name = data_out_file.parts[-1].split('.')[0]                                                            # and get last part which is filename    
    data = np.load(data_out_file)                                                                                           # load the numpy file
    X = data['X']                                                                                                           # extract the data for it
    Y_class = data['Y_class']                                                                                               # and class labels.  
    Y_loc = data['Y_loc']                                                                                                   # and location labels.  
    X_btln = vgg16_block_1to5.predict(X, verbose = 1)                                                                       # predict up to bottleneck    
    np.savez(f'step_05_bottleneck/{bottleneck_file_name}_bottleneck.npz', X = X_btln, Y_class = Y_class, Y_loc = Y_loc)     # save the bottleneck file, and the two types of label.  


#%% 6: Train the fully connected part of the CNN

print('\nStep 06: Training the fully connected part of the CNN using the bottleneck features.')

# 6.1 deal with files
data_files = sorted(glob.glob(f'step_04_merged_rescaled_data/*npz'), key = os.path.getmtime)                  # make list of data files
bottleneck_files = sorted(glob.glob(f'step_05_bottleneck/*npz'), key = os.path.getmtime)                      # and make a list of bottleneck files (ie files that have been passed through the first 5 blocks of vgg16)

if len(data_files) < (cnn_settings['n_files_train'] + cnn_settings['n_files_validate'] + cnn_settings['n_files_test']):
    raise Exception(f"There are {len(data_files)} data files, but {cnn_settings['n_files_train']} have been selected for training, "
                    f"{cnn_settings['n_files_validate']} for validation, and {cnn_settings['n_files_test']} for testing, "
                    f"which sums to greater than the number of data files.  Perhaps adjust the number of files used for the training stages? "
                    f"For now, exiting.")

data_files_train, data_files_validate, data_files_test = file_list_divider(data_files, cnn_settings['n_files_train'], cnn_settings['n_files_validate'], cnn_settings['n_files_test'])                              # divide the files into train, validate and test
bottleneck_files_train, bottleneck_files_validate, bottleneck_files_test = file_list_divider(bottleneck_files, cnn_settings['n_files_train'], cnn_settings['n_files_validate'], cnn_settings['n_files_test'])      # also divide the bottleneck files

X_validate, Y_class_validate, Y_loc_validate      = file_merger(data_files_validate)                             # Open all the validation data to RAM
X_validate_btln, Y_class_validate, Y_loc_validate = file_merger(bottleneck_files_validate)                       # Open the validation data bottleneck features to RAM
X_test, Y_class_test, Y_loc_test                  = file_merger(data_files_test)                                 # Open the test data to RAM
X_test_btln, Y_class_test_btln, Y_loc_test_btln   = file_merger(bottleneck_files_test)                           # Open the test data bottleneck features to RAM

print(f"    There are {len(data_files)} data files.  {len(data_files_train)} will be used for training,"         # print to terminal status on how many files will be used etc.  
      f"{len(data_files_validate)} for validation, and {len(data_files_test)} for testing.  ")

# 6.2 define, compile, and train the model
vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                        # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define out own )
fc_model_input = Input(shape = vgg16_block_1to5.output_shape[1:])                                                 # the input to the fully connected model must be the same shape as the output of the 5th block of vgg16
output_class, output_loc = define_two_head_model(fc_model_input, len(synthetic_ifgs_settings['defo_sources']))    # build the full connected part of the model, and get the two model outputs
vgg16_2head_fc = Model(inputs=fc_model_input, outputs=[output_class, output_loc])                                 # define the model.  Input is the shape of vgg16 block 1 to 5 output, and there are two outputs (hence list)                                
plot_model(vgg16_2head_fc, to_file=f'step_06_train_fully_connected_model/vgg16_2head_fc.png',                     # also plot the model.  This funtcion is known to be fragile due to Graphviz dependencies.  
           show_shapes = True, show_layer_names = True)

loss_class = losses.categorical_crossentropy                                                                      # good loss to use for classification problems, may need to switch to binary if only two classes though?
loss_loc = losses.mean_squared_error                                                                              # loss for localisation
opt_used = optimizers.Nadam(clipnorm = 1., clipvalue = 0.5)                                                       # adam with Nesterov accelerated gradient
vgg16_2head_fc.compile(optimizer = opt_used, loss=[loss_class, loss_loc],                                         # compile the model
                       loss_weights = fc_loss_weights, metrics=['accuracy'])                                      # accuracy is useful to have on the terminal during training

vgg16_2head_fc,  metrics_class_fc, metrics_localisation_fc, metrics_combined_loss_fc = train_double_network(vgg16_2head_fc, bottleneck_files_train,
                                                                                                            n_epochs_fc, ['class_dense3_loss', 'loc_dense6_loss'],
                                                                                                            X_validate_btln, Y_class_validate, Y_loc_validate, len(synthetic_ifgs_settings['defo_sources']))

custom_training_history(metrics_class_fc, n_epochs_fc, title = 'Fully connected classification training')          # plot of the training process for classification
custom_training_history(metrics_localisation_fc, n_epochs_fc, title = 'Fully connected localisation training')     # plot of the training process for localisation
vgg16_2head_fc.save_weights(f'step_06_train_fully_connected_model/vgg16_2head_fc.h5')                              # save the weights of the model we have trained

# 6.3 Test the model
Y_class_test_cnn, Y_loc_test_cnn = vgg16_2head_fc.predict(X_test_btln, verbose = 1)                                # forward pass of the testing data bottleneck features through the fully connected part of the model
plot_data_class_loc_caller(X_test, classes = Y_class_test, classes_predicted = Y_class_test_cnn,                    # plot all the testing data
                           locs = Y_loc_test, locs_predicted = Y_loc_test_cnn, 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = 'Testing data (after step 06)')


#%% Step 07: Fine-tune the 5th convolutional block and the fully connected network.  

vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define out own )
output_class, output_loc = define_two_head_model(vgg16_block_1to5.output, len(synthetic_ifgs_settings['defo_sources']))          # build the fully connected part of the model, and get the two model outputs

vgg16_2head = Model(inputs=vgg16_block_1to5.input, outputs=[output_class, output_loc])                                           # define the full model
vgg16_2head.load_weights(f'step_06_train_fully_connected_model/vgg16_2head_fc.h5', by_name = True)                               # load the weights for the fully connected part which were trained in step 06 (by_name flag so that it doesn't matter that the models are different sizes))

for layer in vgg16_2head.layers[:15]:                                                                                             # freeze blocks 1-4 (ie, we are only fine tuneing the 5th block and the fully connected part of the network)
    layer.trainable = False    

#block5_optimiser = optimizers.RMSprop(lr=block5_lr)                                      
block5_optimiser = optimizers.SGD(lr=block5_lr, momentum=0.9)                                            # set the optimizer used in this training part.  Note have to set a learning rate manualy as an adaptive one (eg Nadam) would wreck model weights in the first few passes before it reduced.  
vgg16_2head.compile(optimizer = block5_optimiser, metrics=['accuracy'],                                  # recompile as we've changed which layers can be trained/ optimizer etc.  
                    loss=[loss_class, loss_loc], loss_weights = block5_loss_weights)                                  

try:
    plot_model(vgg16_2head, to_file='vgg16_2head.png', show_shapes = True, show_layer_names = True)      # try to make a graphviz style image showing the complete model 
except:
    print(f"Failed to create a .png of the model, but continuing anyway.  ")                               # this can easily fail, however, so simply alert the user and continue.  


print('\n\nFine-tuning the 5th convolutional block and the fully connected network.')
vgg16_2head, metrics_class_5th, metrics_localisation_5th, metrics_combined_loss_5th = train_double_network(vgg16_2head, data_files_train,
                                                                                                           n_epochs_block5, ['class_dense3_loss', 'loc_dense6_loss'],
                                                                                                           X_validate, Y_class_validate, Y_loc_validate, len(synthetic_ifgs_settings['defo_sources']))

custom_training_history(metrics_class_5th, n_epochs_block5, title = '5th block classification training')
custom_training_history(metrics_localisation_5th, n_epochs_block5, title = '5th block localisation training')

vgg16_2head.save(f'step_07_train_full_model/01_vgg16_2head_block5_trained.h5')
np.savez(f'step_07_train_full_model/training_history.npz', metrics_class_fc = metrics_class_fc,
                                                         metrics_localisation_fc = metrics_localisation_fc,
                                                         metrics_combined_loss_fc = metrics_combined_loss_fc,
                                                         metrics_class_5th = metrics_class_5th,
                                                         metrics_localisation_5th = metrics_localisation_5th,
                                                         metrics_combined_loss_5th = metrics_combined_loss_5th)




#%% Step 08: Test with synthetic and real data

print('\n\nStep 08: Forward pass of the testing data through the network:')

Y_class_test_cnn, Y_loc_test_cnn = vgg16_2head.predict(X_test[:,:,:,:], verbose = 1)                                    # predict class labels


plot_data_class_loc_caller(X_test, classes = Y_class_test, classes_predicted = Y_class_test_cnn,                    # plot all the testing data
                           locs = Y_loc_test, locs_predicted = Y_loc_test_cnn, 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = 'Testing data (after step 07)')