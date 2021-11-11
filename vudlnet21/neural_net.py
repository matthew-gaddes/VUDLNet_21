#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:07 2020

@author: matthew
"""


#%%
import tensorflow as tf

class numpy_files_sequence(tf.keras.utils.Sequence):                                                                                  # inheritance not tested like ths.  
    """A data generator for use with .npz files that contains X, Y_class, and Y_loc.  Can be used with either training, validation, or testing data.  
    Key methods:
            __len__                 to get the number of batches to pass all the data through (i.e. for one epoch)                        
            __getitem__             to get a batch of data.  
    If built correctly, it should guarantee that each sample is only used once per epoch.  
    """
        
    def __init__(self, file_list, batch_size):                                          # constructor
        """
        Inputs:
            file_list | list of strings or paths | locations of numpy files of data.  
            batch_size | int | number of data for each batch.  Note tested if larger than the number of data in a single file.  
        """
        self.file_list = file_list
        self.batch_size = batch_size

    def __len__(self):                                                      # number of batches in an epoch
        """As one large file (e.g. 1000 data) can't be used as a batch on a GPU (but maybe on a CPU?), we will break each 
        file into n_batches_per_file.  Therefore, the total number of batches (per epoch) will be n_files x n_batches"""
        
        import numpy as np
        n_files = len(self.file_list)                                                           # get the number of data files.  
        n_data_per_file = np.load(self.file_list[0])['X'].shape[0]                              # get the number of data in a file (assumed to be the same for all files)
        n_batches_per_file = int(np.ceil(n_data_per_file / self.batch_size))                    # the number of batches required to cover every data in the file.  
        n_batches = n_files * n_batches_per_file
        return n_batches

    def __getitem__(self, idx):                                             # iterates over the data and returns a complete batch, index is a number upto the number of batches set by __len__, with each number being used once but in a random order.  
        
        import numpy as np
        # repeat of __len__ to get info about batch sizes etc, probably a better way to do this.  
        n_files = len(self.file_list)                                                      # get the number of data files.  
        n_data_per_file = np.load(self.file_list[0])['X'].shape[0]                         # get the number of data in a file (assumed to be the same for all files)
        n_batches_per_file = int(np.ceil(n_data_per_file / self.batch_size))               # the number of batches required to cover every data in the file.  
        n_batches = n_files * n_batches_per_file
        
        # deal with files and batches (convert idx to a file number and batch number).  
        n_file, n_batch = divmod(idx, n_batches_per_file)                                   # idx tells us which batch (of the total number of batches from __len__), but that needs mapping to a file, and to which batch in that file.  
                                                                                            # divmod returns the quotient (file_n), and the remainder (batch from that file)
        data = np.load(self.file_list[n_file])                                              # load the correct numpy file.  
        X = data['X']                                                                       # eOpen the correct data file
        Y_class = data['Y_class']
        Y_loc = data['Y_loc']
        
        X_batch = X[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]
        Y_class = Y_class[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]
        Y_loc = Y_loc[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]
        
        return X_batch, [Y_class, Y_loc]



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

#%%


def define_two_head_model(model_input, n_class_outputs = 3):
    """ Define the two headed model that we have designed to performed classification and localisation.  
    Inputs:
        model_input | tensorflow.python.framework.ops.Tensor | The shape of the tensor that will be input to our model.  Usually the output of VGG16 (?x7x7x512)  Nb ? = batch size.  
        n_class_output | int | For a one hot encoding style output, there must be as many neurons as classes
    Returns:
        output_class |tensorflow.python.framework.ops.Tensor | The shape of the tensor output by the classifiction head.  Usually ?x3
        output_loc | tensorflow.python.framework.ops.Tensor | The shape of the tensor output by the localisation head.  Usually ?x4
    History:
        2020_11_11 | MEG | Written
        2021_08_24 | MEG | Remove a layer of 1024 neurons.  
    """
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    
    vgg16_block_1to5_flat = Flatten(name = 'vgg16_block_1to5_flat')(model_input)                              # flatten the model input (ie deep representation turned into a column vector)

    # 1: the clasification head
    x = Dropout(0.2, name='class_dropout1')(vgg16_block_1to5_flat)
    x = Dense(256, activation='relu', name='class_dense1')(x)                                                 # add a fully connected layer
    x = Dropout(0.2, name='class_dropout2')(x)
    x = Dense(128, activation='relu', name='class_dense2')(x)                                                 # add a fully connected layer
    output_class = Dense(n_class_outputs, activation='softmax',  name = 'class_dense3')(x)                  # and an ouput layer with 7 outputs (ie one per label)
    
    # 2: the localization head
    x = Dense(2048, activation='relu', name='loc_dense1')(vgg16_block_1to5_flat)                                                 # add a fully connected layer
    x = Dense(1024, activation='relu', name='loc_dense2')(x)                                                 # add a fully connected layer
    x = Dropout(0.2, name='loc_dropout1')(x)
    x = Dense(512, activation='relu', name='loc_dense4')(x)                                                 # add a fully connected layer
    x = Dense(128, activation='relu', name='loc_dense5')(x)                                                 # add a fully connected layer
    output_loc = Dense(4, name='loc_dense6')(x)        
    
    return output_class, output_loc
    


    


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
    
    def open_synthetic_data_npz(name_with_path):
        """Open a file data file """  
        data = np.load(name_with_path)
        X = data['X']
        Y_class = data['Y_class']
        Y_loc = data['Y_loc']
        return X, Y_class, Y_loc

    n_files = len(files)
    
    for i, file in enumerate(files):
        X_batch, Y_class_batch, Y_loc_batch = open_synthetic_data_npz(file)
        if i == 0:
            n_data_per_file = X_batch.shape[0]
            X = np.zeros((n_data_per_file * n_files, X_batch.shape[1], X_batch.shape[2], X_batch.shape[3]))      # initate array, rank4 for image, get the size from the first file
            Y_class = np.zeros((n_data_per_file  * n_files, Y_class_batch.shape[1]))                              # should be flexible with class labels or one hot encoding
            Y_loc = np.zeros((n_data_per_file * n_files, 4))                                                     # four columns for bounding box
            
        
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

def merge_and_rescale_data(synthetic_data_files, real_data_files, outdir, output_range = {'min':0, 'max':225}):
    """ Given a list of synthetic data files and real data files (usually the augmented real data),
    
    Inputs:
        synthetic_data_files | list of Paths or string | locations of the .pkl files containing the masked arrays
        real_data_files      | list of Paths or string | locations of the .pkl files containing the masked arrays
        outdir               | pathlib path | path to directory to save the .npz files to
        output_range         | dict                    | min and maximum of each channel in each image.  Should be set to suit the CNN being used.  
    Returns:
        .npz files in step_04_merged_rescaled_data
    History:
        2020_10_29 | MEG | Written
        2021_01_06 | MEG | Fix bug in that mixed but not rescaled data was being written to the numpy arrays.  
        2021_09_23 | MEG | add option to save to a directory.  

    """
    import pickle
    import numpy as np
    import numpy.ma as ma
    
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

        X_rescale = custom_range_for_CNN(X, output_range, mean_centre = False)                      # resacle the data from metres/rads etc. to desired input range of cnn (e.g. [0, 255]), and convert to numpy array
                
        data_mid = int(X_rescale.shape[0] / 2)                                                                                                                          # data before this number in one file, and after in another
        np.savez(outdir / "step_04_merged_rescaled_data" / f"data_file_{out_file:05d}.npz", X = X_rescale[:data_mid,:,:,:], Y_class= Y_class[:data_mid,:], Y_loc = Y_loc[:data_mid,:])           # save the first half of the data
        out_file += 1                                                                                                                                                   # after saving once, update
        np.savez(outdir / "step_04_merged_rescaled_data" / f"data_file_{out_file:05d}.npz", X = X_rescale[data_mid:,:,:,:], Y_class= Y_class[data_mid:,:], Y_loc = Y_loc[data_mid:,:])           # save the second half of the data
        out_file += 1                                                                                                                                                   # and after saving again, update again.  
        print('Done.  ')
        
        

#%%

def custom_range_for_CNN(r4_array, min_max, mean_centre = False):
    """ Rescale a rank 4 array so that each channel's image lies in custom range
    e.g. input with range of (-5, 15) is rescaled to (-125 125) or (-1 1) for use with VGG16.  
    Designed for use with masked arrays.  
    Inputs:
        r4_array | r4 masked array | works with masked arrays?  
        min_max | dict | 'min' and 'max' of range desired as a dictionary.  
        mean_centre | boolean | if True, each image's channels are mean centered.  
    Returns:
        r4_array | rank 4 numpy array | masked items are set to zero, rescaled so that each channel for each image lies between min_max limits.  
    History:
        2019/03/20 | now includes mean centering so doesn't stretch data to custom range.  
                    Instead only stretches until either min or max touches, whilst mean is kept at 0
        2020/11/02 | MEG | Update so range can have a min and max, and not just a range
        2021/01/06 | MEG | Upate to work with masked arrays.  Not test with normal arrays.
    """
    import numpy as np
    import numpy.ma as ma
    
    if mean_centre:
        im_channel_means = ma.mean(r4_array, axis = (1,2))                                                  # get the average for each image (in all thre channels)
        im_channel_means = expand_to_r4(im_channel_means, r4_array[0,:,:,0].shape)                                                   # expand to r4 so we can do elementwise manipulation
        r4_array -= im_channel_means                                                                        # do mean centering    


    im_channel_min = ma.min(r4_array, axis = (1,2))                                         # get the minimum of each image and each of its channels
    im_channel_min = expand_to_r4(im_channel_min, r4_array[0,:,:,0].shape)                  # exapnd to rank 4 for elementwise applications
    r4_array -= im_channel_min                                                              # set so lowest channel for each image is 0
    
    im_channel_max = ma.max(r4_array, axis = (1,2))                                         # get the maximum of each image and each of its channels
    im_channel_max = expand_to_r4(im_channel_max, r4_array[0,:,:,0].shape)              # make suitable for elementwise applications
    r4_array /= im_channel_max                                                              # should now be in range [0, 1]
    
    r4_array *= (min_max['max'] - min_max['min'])                                           # should now be in range [0, new max-min]
    r4_array += min_max['min']                                                              # and now in range [new min, new max]        
    r4_nparray = r4_array.filled(fill_value = 0)                                            # convert to numpy array, maksed incoherent areas are set to zero.  
    

    return r4_nparray    

#%%

def expand_to_r4(r2_array, shape = (224,224)):
    """
    Calcaulte something for every image and channel in rank 4 data (e.g. 100x224x224x3 to get 100x3)
    Expand new rank 2 to size of original rank 4 for elemtiwise operations
    """
    import numpy as np
    
    r4_array = r2_array[:, np.newaxis, np.newaxis, :]
    r4_array = np.repeat(r4_array, shape[0], axis = 1)
    r4_array = np.repeat(r4_array, shape[1], axis = 2)
    return r4_array



#%%

def augment_data(X, Y_class, Y_loc, n_data = 500):
    """ A function to augment data and presserve the location label for any deformation.  
    Note that n_data is not particularly intelligent as many more data may be generated,
    and only n_data returned, so even if n_data is low, the function can still be slow.  
    Inputs:
        X           | rank 4 array | data.  
        Y_class     | rank 2 array | One hot encoding of class labels
        Y_loc       | rank 2 array | locations of deformation
        n_data      | int |
    Returns:
        X_aug           | rank 4 array | data.  
        Y_class_aug     | rank 2 array | One hot encoding of class labels
        Y_loc_aug       | rank 2 array | locations of deformation
    History:
        2019/??/?? | MEG | Written
        2020/10/29 | MEG | Write the docs.  
        2020_01_11 | MEG | Major rewrite to speed things up.  
    """
    import numpy as np
    import numpy.ma as ma

    flips = ['none', 'up_down', 'left_right', 'both']                                       # the three possible types of flip

    # 0: get the correct nunber of data    
    n_ifgs = X.shape[0]
    data_dict = {'X'       : X,                                                      # package the data and labels together into a dict
                 'Y_class' : Y_class,
                 'Y_loc'   : Y_loc}
    
    if n_ifgs < n_data:                                                                                 # if we have fewer ifgs than we need, repeat them  
        n_repeat = int(np.ceil(n_data / n_ifgs))                                                        # get the number of repeats needed (round up and make an int)
        data_dict['X'] = ma.repeat(data_dict['X'], axis = 0, repeats = n_repeat)
        data_dict['Y_class'] = np.repeat(data_dict['Y_class'], axis = 0, repeats = n_repeat)
        data_dict['Y_loc'] = np.repeat(data_dict['Y_loc'], axis = 0, repeats = n_repeat)
    
    data_dict = shuffle_arrays(data_dict)                                                       # shuffle (so that these aren't in the order of the class labels)    
    for key in data_dict:                                                                       # then crop them to the correct number
        data_dict[key] = data_dict[key][:n_data,]
                
    X_aug = data_dict['X']                                                      # and unpack as this function doesn't use dictionaries
    Y_class_aug = data_dict['Y_class']
    Y_loc_aug = data_dict['Y_loc']

    # 1: do the flips        
    for data_n in range(n_data):
        flip = flips[np.random.randint(0, len(flips))]                                                       # choose a flip at random
        if flip != 'none':
            X_aug[data_n:data_n+1,], Y_loc_aug[data_n:data_n+1,] = augment_flip(X_aug[data_n:data_n+1,], Y_loc_aug[data_n:data_n+1,], flip)          # do the augmentaiton via one of the flips.  
        
    # 2: do the rotations
    X_aug, Y_loc_aug = augment_rotate(X_aug, Y_loc_aug)                # rotate 
    
    # 3: Do the translations
    X_aug, Y_loc_aug = augment_translate(X_aug,  Y_loc_aug, max_translate = (20,20)) 
        
    return X_aug, Y_class_aug, Y_loc_aug                    # return, and select only the desired data.  
    
    # # do all possible fips - now have 4x as much data
    
    # X_aug = [X]                                                                     # make the original data an entry in a list
    # Y_loc_aug = [Y_loc]                                                             # ditto
    # for flip in flips:
    #     X_temp, Y_loc_temp = augment_flip(X, Y_loc, flip)                           # do the augmentaiton via one of the flips.  
    #     X_aug.append(X_temp)                                                        # append to the lis of data
    #     Y_loc_aug.append(Y_loc_temp)                                                # also append the new locations (as they change with flips)
    # X_aug = ma.vstack(X_aug)                                                        # convert from a list back to one big array (ie stack along the first axis).  If we had n data, we now have 4n
    # Y_loc_aug = np.vstack(Y_loc_aug)                                                # and for locations
    # Y_class_aug = np.tile(Y_class, (4,1))                                           # classes don't change with flips etc. so can just be repeated.  
        
    # # rotate
    # X_aug = [X_aug]                                                                # convert back to an entry in a list
    # Y_loc_aug = [Y_loc_aug]                                                        # same for locations
    # for random_rotate in range(3):                                                 # 4n data will again become 16n data, as we will do 3 sets of rotations and keep the originals.  
    #     X_temp, Y_loc_temp = augment_rotate(X_aug[0], Y_loc_aug[0])                # rotate 4n data to make another 4n of data
    #     X_aug.append(X_temp)
    #     Y_loc_aug.append(Y_loc_temp)
    # X_aug = ma.vstack(X_aug)                                                       # 4 entries of 4n data are converted to 16n data
    # Y_loc_aug = np.vstack(Y_loc_aug)                                               # same for locations
    # Y_class_aug = np.tile(Y_class_aug, (4,1))                                      # classes don't change so can just be copied

    # # translate
    # X_aug = [X_aug]                                                                # now have 16n data, which becomes first item in list
    # Y_loc_aug = [Y_loc_aug]
    # for random_translate in range(3):                                              # 16n data will again become 64n data, as we will do 3 sets of translations and keep the originals.  
    #     X_temp, Y_loc_temp = augment_translate(X_aug[0],  Y_loc_aug[0], max_translate = (20,20)) 
    #     X_aug.append(X_temp)
    #     Y_loc_aug.append(Y_loc_temp)
    # X_aug = ma.vstack(X_aug)                                                        # 4 entries of 16n data are converted to 64n data
    # Y_loc_aug = np.vstack(Y_loc_aug)
    # Y_class_aug = np.tile(Y_class_aug, (4,1))    
    
    
    # # shuffle along the first axis (otherwise with small values of n_data, we just get the first (and original) data back.  )
    # data_dict = {'X'       : X_aug,                                                      # package the data and labels together into a dict
    #              'Y_class' : Y_class_aug,
    #              'Y_loc'   : Y_loc_aug}
    
    # data_dict_shuffled = shuffle_arrays(data_dict)                                          # shuffle (so that these aren't in the order of the class labels)
    
    # X_aug = data_dict_shuffled['X']                                                      # and unpack as this function doesn't use dictionaries
    # Y_class_aug = data_dict_shuffled['Y_class']
    # Y_loc_aug = data_dict_shuffled['Y_loc']
    
    
    

#%%

def augment_flip(X, Y_loc, flip):
    """A function to flip data horizontally or vertically 
    and apply the same transformation to the location label
    Inputs:
        X | r4 array | samples x height x width x channels
        Y_loc | r2 array | samples X 4
        flip | string | determines which way to flip.  
    """
    import numpy as np
    import numpy.ma as ma
    
    Y_loc_flip = np.copy(Y_loc)                              # make a copy of the location labels
    
    if flip is 'up_down':                                       # conver the string input to a value
        X_flip = X[:,::-1,:,:]                              # reverse in dim 2 which is y
        Y_loc_flip[:,1] = X.shape[1] - Y_loc_flip[:,1]
    elif flip is 'left_right':
        X_flip = X[:,:,::-1,:]                              # reverse in dim 3 which is x
        Y_loc_flip[:,0] = X.shape[2] - Y_loc_flip[:,0]      # flipping horizontally
    elif flip is 'both':
        X_flip = X[:,::-1,::-1,:]                              # reverse in dim 2 (y) and dim 3 (x)
        Y_loc_flip[:,1] = X.shape[1] - Y_loc_flip[:,1]
        Y_loc_flip[:,0] = X.shape[2] - Y_loc_flip[:,0]      # flipping horizontally
    else:
        raise Exception("'flip' must be either 'up_down', 'left_right', or 'both'.  ")
    
    return X_flip, Y_loc_flip





def augment_rotate(X, Y_loc):
    """ Rotate data and the label.  Angles are random in range [0 360], and different for each sample.  
    Note: Location labels aren't rotated!  Assumed to be roughly square.  
    Inputs:
        X | r4 array | samples x height x width x channels
        Y_loc | r2 array | samples X 4
    Returns:
        
    """
    import numpy as np
    import numpy.ma as ma
    
    def rot(image, xy, angle):
        """Taken from stack exchange """
        from scipy.ndimage import rotate
        im_rot = rotate(image,angle, reshape = False, mode = 'nearest') 
        org_center = (np.array(image.shape[:2][::-1])-1)/2.
        rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        return im_rot, new+rot_center

    X_rotate = ma.copy(X)
    Y_loc_rotate = ma.copy(Y_loc)
    rotate_angles_deg = np.random.randint(0, 360, X.shape[0])

    for n_ifg, ifg in enumerate(X):                                                                     #loop through each ifg
        ifg_rot, xy_rot = rot(ifg, Y_loc_rotate[n_ifg,:2], rotate_angles_deg[n_ifg])
        X_rotate[n_ifg,:,:,:] = ifg_rot
        Y_loc_rotate[n_ifg, :2] = xy_rot

    return X_rotate, Y_loc_rotate

def augment_translate(X, Y_loc, max_translate = (20,20)):
    """
    Inputs:
        max_translate | tuple | max x translation, max y translation
    """
    import numpy as np
    import numpy.ma as ma
    
    n_pixs = X.shape[1]                                                             # normally 224

    X_translate = ma.copy(X)
    Y_loc_translate = ma.copy(Y_loc)
    x_translations = np.random.randint(0, 2*max_translate[0], X.shape[0])           # translations could be + or - max translation, but everything is positive when indexing arrays so double the max translation
    y_translations = np.random.randint(0, 2*max_translate[1], X.shape[0])
    
    Y_loc_translate[:,0] -= x_translations - max_translate[0]                                  # these are the x centres
    Y_loc_translate[:,1] -= y_translations - max_translate[1]                                # these are the y centres
    
    
    for n_ifg, ifg in enumerate(X):                                                                                                                         #loop through each ifg, but ma doesn't have a pad (ie can't pad masked arrays)
        ifg_large_data = np.pad(ma.getdata(ifg), ((max_translate[1],max_translate[1]),(max_translate[0], max_translate[0]), (0,0)), mode = 'edge')          # padding the data  (y then x then channels)
        ifg_large_mask = np.pad(ma.getmask(ifg), ((max_translate[1],max_translate[1]),(max_translate[0], max_translate[0]), (0,0)), mode = 'edge')          # padding the mask (y then x then channels)
        ifg_large = ma.array(ifg_large_data, mask=ifg_large_mask)                                                                                           # recombining the padded mask and data to make an enlarged masked array
        ifg_crop = ifg_large[y_translations[n_ifg]:y_translations[n_ifg]+n_pixs,x_translations[n_ifg]:x_translations[n_ifg]+n_pixs, :]                      # crop from the large ifg back to the original resolution
        X_translate[n_ifg,:,:,:] = ifg_crop                                                                                                                 # append result to big rank 4 of ifgs

    return X_translate, Y_loc_translate



#%%

def choose_for_augmentation(X, Y_class, Y_loc, n_per_class):
    """A function to randomly select only some of the data, but in  fashion that ensures that the classes are balanced 
    (i.e. there are equal numbers of each class).  Particularly useful if working with real data, and the classes
    are usually very unbalanced (lots of no_def lables, normally).  
    Inputs:
        X           | rank 4 array | data.  
        Y_class     | rank 2 array | One hot encoding of class labels
        Y_loc       | rank 2 array | locations of deformation
        n_per_class | int | number of data per class. e.g. 3
    Returns:
        X_sample           | rank 4 array | data.  
        Y_class_sample     | rank 2 array | One hot encoding of class labels
        Y_loc_sample       | rank 2 array | locations of deformation
    History:
        2019/??/?? | MEG | Written
        2019/10/28 | MEG | Update to handle dicts
        2020/10/29 | MEG | Write the docs.  
        2020/10/30 | MEG | Fix bug that was causing Y_class and Y_loc to become masked arrays.  
        
    """
    import numpy as np
    import numpy.ma as ma
    
    n_classes = Y_class.shape[1]                                                         # only works if one hot encoding is used
    X_sample = []
    Y_class_sample = []
    Y_loc_sample = []
    
    for i in range(n_classes):                                                              # loop through each class
        args_class = np.ravel(np.argwhere(Y_class[:,i] != 0))                               # get the args of the data of this label
        args_sample = args_class[np.random.randint(0, len(args_class), n_per_class)]        # choose n_per_class of these (ie so we always choose the same number from each label)
        X_sample.append(X[args_sample, :,:,:])                                              # choose the data, and keep adding to a list (each item in the list is n_per_class_label x ny x nx n chanels)
        Y_class_sample.append(Y_class[args_sample, :])                                      # and class labels
        Y_loc_sample.append(Y_loc[args_sample, :])                                          # and location labels
    
    X_sample = ma.vstack(X_sample)                                                          # maskd array, merge along the first axis, so now have n_class x n_per_class of data
    Y_class_sample = np.vstack(Y_class_sample)                                              # normal numpy array, note that these would be in order of the class (ie calss 0 first, then class 1 etc.  )
    Y_loc_sample = np.vstack(Y_loc_sample)                                                  # also normal numpy array
    
    data_dict = {'X'       : X_sample,                                                      # package the data and labels together into a dict
                 'Y_class' : Y_class_sample,
                 'Y_loc'   : Y_loc_sample}
    
    data_dict_shuffled = shuffle_arrays(data_dict)                                          # shuffle (so that these aren't in the order of the class labels)
    
    X_sample = data_dict_shuffled['X']                                                      # and unpack as this function doesn't use dictionaries
    Y_class_sample = data_dict_shuffled['Y_class']
    Y_loc_sample = data_dict_shuffled['Y_loc']
    
    return X_sample, Y_class_sample, Y_loc_sample



#%%

def shuffle_arrays(data_dict):
    """A function to shuffle a selection of arrays along their first axis.  
    The arrays are all shuffled in the same way (so good for data and labels)
    Inputs:
        data_dict | dictionary | containing e.g. X, X_m, Y_class, Y_loc
    Returns:
        data_dict_shuffled | dictionary | containing e.g. X, X_m, Y_class, Y_loc, shuffled along first axis
    History:
        2019/??/?? | MEG | Written
        2020/10/28 | MEG | Comment.  
            
    """
    import numpy as np
    import numpy.ma as ma
    
    data_dict_shuffled = {}                                                         # initiate
    args = np.arange(0, data_dict['X'].shape[0])                                    # get the numbers along the first dim (which is the number of data)
    np.random.shuffle(args)                                                         # shuffle this
    
    for data_label_name in data_dict:                                               # loop through each array in dictionary 
        data_dict_shuffled[data_label_name] = data_dict[data_label_name][args,:]    # and then copy to the new dict in the shuffled order (set by args)
        
    return data_dict_shuffled
