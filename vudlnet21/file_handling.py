#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:28:33 2021

@author: matthew
"""

import pdb

#%%


def shuffle_data_pkls(unshuffled_files, file_batch_size, outdir):
    """ Given a direcotry of unshuffled data that is split into files that cannot all be loaded into RAM at once, shuffle by opening 
    a subset of them at a time and shuffling them.  
    
    For a more complete shuffle, this function can be called multiple times on the results of itself.  
    
    Inputs:
        unshuffld_files | list of strings | paths to unshuffled files.  
        file_batch_size | int | number of files in one batch.  Bigger is better, but will eventually max out machine RAM.  
        outdir | pathlib Path | our directory for shuffled files.  
        
    Returns:
        shuffled files.  
        
    History:
        2022_10_11 | MEG | Written.  
        
    
    
    """
    import random
    import numpy as np
    import numpy.ma as ma
    import pickle
    import os

    
    random.shuffle(unshuffled_files)                                                            # shuffle the items in the list (this doesn't actually shuffle the data in the files though, so still called unshuffled)

    if len(unshuffled_files) % file_batch_size != 0:
        raise Exception(f"This function has not been tested with batch sizes that do not divide exactly into the number of files.  "
                        f"({len(unshuffled_files)} were detected, batches were {file_batch_size}, producing a {int(len(unshuffled_files) / file_batch_size)} batches and a remainder of {len(unshuffled_files) % file_batch_size})  "
                        f"Exiting.")
    else:
        n_batches = int(len(unshuffled_files) / file_batch_size)
    
    # get some info about the data by opening the first file.  
    with open(unshuffled_files[0], 'rb') as f:                                                    # open the file
        X = pickle.load(f)
    n_per_file, ny, nx, _ = X.shape
    
    # loop through opening file_batch_size files at once. (e.g. 10 files are opened and the contents shuffled)
    file_n = 0
    for i in range(n_batches):
        unshuffled_batch_files = unshuffled_files[i * file_batch_size: (i+1)*file_batch_size]                           # 
    
        X = ma.zeros((file_batch_size * n_per_file, ny, nx, 1))               # initialise, rank 4 ready for Tensorflow, last dimension is being used for different crops.  
        Y_class = np.zeros((file_batch_size * n_per_file, 3))                                                                             # initialise, doesn't need another dim as label is the same regardless of the crop.  
        Y_loc = np.zeros((file_batch_size * n_per_file, 4))                                                                            # initialise
        
        for file_n_batch, unshuffled_batch_file in enumerate(unshuffled_batch_files):
            with open(unshuffled_batch_file, 'rb') as f:                                                    # open the file
                print(f"Opening file {unshuffled_batch_file}.")
                X[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ]  = pickle.load(f)                                                              # and extract data (X) and labels (Y)
                Y_class[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ] = pickle.load(f)
                Y_loc[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ] = pickle.load(f)
            os.remove(unshuffled_batch_file)
        
        # do the shuffling        
        args = np.arange(0, X.shape[0])
        random.shuffle(args)
        X = X[args,]
        Y_class = Y_class[args,]
        Y_loc = Y_loc[args,]
        
        # save parts of the large shuffled array into separate files.  
        for file_n_batch in range(file_batch_size):
            print(f"    Saving shuffled file {file_n}")
            with open(outdir / f"data_file_shuffled_{file_n:05d}.pkl", 'wb') as f:                     # save the output as a pickle
                pickle.dump(X[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ], f)
                pickle.dump(Y_class[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ], f)
                pickle.dump(Y_loc[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ], f)
            file_n += 1
           



#%%


def rescale_data(data_files, outdir, output_range = {'min':0, 'max':225}, triplicate_channel = False):
    """ Given a list of synthetic data files and real data files (usually the augmented real data),
    
    Inputs:
        data_files           | list of Paths or string | locations of the .pkl files containing the masked arrays
        outdir               | pathlib Path            | directory to save numpy files.  
        output_range         | dict                    | min and maximum of each channel in each image.  Should be set to suit the CNN being used.  
        triplicate_channel   | Boolean                  | Data are saved as rank 4 (n_data, ny, nx, n_chanels), and this option repeates the channel data 3 times. 
                                                          Useful if we have one channel data but want to use it with an RGB triple channel model.  
    Returns:
        .npz files in step_04_merged_rescaled_data
    History:
        2020_10_29 | MEG | Written
        2021_01_06 | MEG | Fix bug in that mixed but not rescaled data was being written to the numpy arrays.  
        2022_10_17 | MEG | Add option to repeat data across 3 channels.  

    """
    import pickle
    import numpy as np
    import numpy.ma as ma
    
    from deep_learning_tools.data_handling import custom_range_for_CNN
    
    n_files = len(data_files)        
    out_file = 0
    for n_file in range(n_files):
        print(f'    Opening file {n_file}... ', end = '')
        with open(data_files[n_file], 'rb') as f:                                                      # open the real data file
            X = pickle.load(f)
            Y_class = pickle.load(f)
            Y_loc = pickle.load(f)
        f.close()    
        

        mix_index = np.arange(0, X.shape[0])                                                        # mix them, get a lis of arguments for each data 
        np.random.shuffle(mix_index)                                                                # shuffle the arguments
        X = X[mix_index,]                                                                           # reorder the data using the shuffled arguments
        Y_class = Y_class[mix_index]                                                                # reorder the class labels
        Y_loc = Y_loc[mix_index]                                                                    # and the location labels

        X_rescale = custom_range_for_CNN(X, output_range)                                           # resacle the data from metres/rads etc. to desired input range of cnn (e.g. [0, 255]), and convert to numpy array
        
        if triplicate_channel:
            X_rescale = np.repeat(X_rescale, repeats = 3, axis = -1)                                 # repeat the channels (last dimension) three times.  
                
        data_mid = int(X_rescale.shape[0] / 2)                                                                                                                          # data before this number in one file, and after in another
        np.savez(outdir / f"data_file_{out_file:05d}.npz", X = X_rescale[:data_mid,:,:,:], Y_class= Y_class[:data_mid,:], Y_loc = Y_loc[:data_mid,:])           # save the first half of the data
        out_file += 1                                                                                                                                                   # after saving once, update
        print('Done.  ')
        
      
  


#%%


def merge_and_rescale_data(synthetic_data_files, real_data_files, output_range = {'min':0, 'max':225},
                           triplicate_channel = False):
    """ Given a list of synthetic data files and real data files (usually the augmented real data),
    
    Inputs:
        synthetic_data_files | list of Paths or string | locations of the .pkl files containing the masked arrays
        reak_data_files      | list of Paths or string | locations of the .pkl files containing the masked arrays
        output_range         | dict                    | min and maximum of each channel in each image.  Should be set to suit the CNN being used.  
        triplicate_channel   \ Boolean                  | Data are saved as rank 4 (n_data, ny, nx, n_chanels), and this option repeates the channel data 3 times. 
                                                          Useful if we have one channel data but want to use it with an RGB triple channel model.  
    Returns:
        .npz files in step_04_merged_rescaled_data
    History:
        2020_10_29 | MEG | Written
        2021_01_06 | MEG | Fix bug in that mixed but not rescaled data was being written to the numpy arrays.  
        2022_10_17 | MEG | Add option to repeat data across 3 channels.  

    """
    import pickle
    import numpy as np
    import numpy.ma as ma
    
    from deep_learning_tools.data_handling import custom_range_for_CNN
    
    def data_channel_checker(X, n_cols = None, window_title = None):
        """ Plot some of the data in X.   All three channels are shown.  
        """
        import matplotlib.pyplot as plt        
        if n_cols == None:                                              # if n_cols is None, we'll plot all the data
            n_cols = X.shape[0]                                         # so n_cols is the number of data
            plot_args = np.arange(0, n_cols)                            # and we'll be plotting each of them
        else:
            plot_args = np.random.randint(0, X.shape[0], n_cols)        # else, pick some at random to plot
        f, axes = plt.subplots(3,n_cols)
        if window_title is not None:
            f.canvas.set_window_title(window_title)
        for plot_n, im_n in enumerate(plot_args):                           # loop through each data (column)               
            axes[0, plot_n].set_title(f"Data: {im_n}")
            for channel_n in range(3):                                      # loop through each row
                axes[channel_n, plot_n].imshow(X[im_n, :,:,channel_n])
                if plot_n == 0:
                    axes[channel_n, plot_n].set_ylabel(f"Channel {channel_n}")

    if len(synthetic_data_files) != len(real_data_files):
        raise Exception('This funtion is only designed to be used when the number of real and synthetic data files are the same.  Exiting.  ')

    n_files = len(synthetic_data_files)        
    out_file = 0
    for n_file in range(n_files):
        print(f'    Opening and merging file {n_file} of each type... ', end = '')
        with open(real_data_files[n_file], 'rb') as f:                                                      # open the real data file
            X_real = pickle.load(f)
            Y_class_real = pickle.load(f)
            Y_loc_real = pickle.load(f)
        f.close()    
        
        with open(synthetic_data_files[n_file], 'rb') as f:                                                      # open the synthetic data file
            X_synth = pickle.load(f)
            Y_class_synth = pickle.load(f)
            Y_loc_synth = pickle.load(f)
        f.close()    

        X = ma.concatenate((X_real, X_synth), axis = 0)                                             # concatenate the data
        Y_class = ma.concatenate((Y_class_real, Y_class_synth), axis = 0)                           # and the class labels
        Y_loc = ma.concatenate((Y_loc_real, Y_loc_synth), axis = 0)                                 # and the location labels
        
        mix_index = np.arange(0, X.shape[0])                                                        # mix them, get a lis of arguments for each data 
        np.random.shuffle(mix_index)                                                                # shuffle the arguments
        X = X[mix_index,]                                                                           # reorder the data using the shuffled arguments
        Y_class = Y_class[mix_index]                                                                # reorder the class labels
        Y_loc = Y_loc[mix_index]                                                                    # and the location labels

        X_rescale = custom_range_for_CNN(X, output_range)                                           # resacle the data from metres/rads etc. to desired input range of cnn (e.g. [0, 255]), and convert to numpy array
        
        if triplicate_channel:
            X_rescale = np.repeat(X_rescale, repeats = 3, axis = -1)                                 # repeat the channels (last dimension) three times.  
                
        data_mid = int(X_rescale.shape[0] / 2)                                                                                                                          # data before this number in one file, and after in another
        np.savez(f'step_05_merged_rescaled_data/data_file_{out_file:05d}.npz', X = X_rescale[:data_mid,:,:,:], Y_class= Y_class[:data_mid,:], Y_loc = Y_loc[:data_mid,:])           # save the first half of the data
        out_file += 1                                                                                                                                                   # after saving once, update
        np.savez(f'step_05_merged_rescaled_data/data_file_{out_file:05d}.npz', X = X_rescale[data_mid:,:,:,:], Y_class= Y_class[data_mid:,:], Y_loc = Y_loc[data_mid:,:])           # save the second half of the data
        out_file += 1                                                                                                                                                   # and after saving again, update again.  
        print('Done.  ')
        
      
    
      

#%%


def file_merger(files): 
    """Given a list of files, open them and merge into one array.  
    Inputs:
        files | list | list of paths to the .npz files
    Returns
        X | r4 array | data
        Y_class | r2 array | class labels, ? x n_classes
        Y_loc | r2 array | locations of signals, ? x 4 (as x,y, width, heigh)
    History:
        2020/10/?? | MEG | Written
        2020/11/11 | MEG | Update to remove various input arguments
    
    """
    import numpy as np
    import numpy.ma as ma
    import pickle
    from pathlib import Path
    
    # def open_synthetic_data_npz(name_with_path):
    #     """Open a file data file """  
    #     data = np.load(name_with_path)
    #     X = data['X']
    #     Y_class = data['Y_class']
    #     Y_loc = data['Y_loc']
    #     return X, Y_class, Y_loc

    n_files = len(files)
    
    for i, file in enumerate(files):
        file = Path(file)
        if file.suffix == '.pkl':                                                          # if it's a .pkl
            with open(file, 'rb') as f:                                                    # open the file
                X_batch = pickle.load(f)                                                              # and extract data (X) and labels (Y)
                Y_class_batch = pickle.load(f)
                Y_loc_batch = pickle.load(f)
            f.close()
        elif file.suffix == '.npz':                                                        # if it's a npz
            data = np.load(file)                                                           # load it
            X_batch = data['X']                                                                       # and extract data (X) and labels (Y)
            Y_class_batch = data['Y_class']
            Y_loc_batch = data['Y_loc']
        
        
        # X_batch, Y_class_batch, Y_loc_batch = open_synthetic_data_npz(file)
        if i == 0:
            n_data_per_file = X_batch.shape[0]
            X = ma.zeros((n_data_per_file * n_files, X_batch.shape[1], X_batch.shape[2], X_batch.shape[3]))      # initate array, rank4 for image, get the size from the first file
            Y_class = ma.zeros((n_data_per_file  * n_files, Y_class_batch.shape[1]))                              # should be flexible with class labels or one hot encoding
            Y_loc = ma.zeros((n_data_per_file * n_files, 4))                                                     # four columns for bounding box
            
        
        X[i*n_data_per_file:(i*n_data_per_file)+n_data_per_file,:,:,:] = X_batch
        Y_class[i*n_data_per_file:(i*n_data_per_file)+n_data_per_file,:] = Y_class_batch
        Y_loc[i*n_data_per_file:(i*n_data_per_file)+n_data_per_file,:] = Y_loc_batch
    
    return X, Y_class, Y_loc 

#%%

def file_list_divider(file_list, n_files_train, n_files_validate, n_files_test):
    """ Given a list of files, divide it up into training, validating, and testing lists.  
    Inputs
        file_list | list | list of files
        n_files_train | int | Number of files to be used for training
        n_files_validate | int | Number of files to be used for validation (during training)
        n_files_test | int | Number of files to be used for testing
    Returns:
        file_list_train | list | list of training files
        file_list_validate | list | list of validation files
        file_list_test | list | list of testing files
    History:
        2019/??/?? | MEG | Written
        2020/11/02 | MEG | Write docs
        """
    file_list_train = file_list[:n_files_train]
    file_list_validate = file_list[n_files_train:(n_files_train+n_files_validate)]
    file_list_test = file_list[(n_files_train+n_files_validate) : (n_files_train+n_files_validate+n_files_test)]
    return file_list_train, file_list_validate, file_list_test


#%%

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


