#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:07 2020

@author: matthew
"""

#%%

def train_double_network(model, files, n_epochs, loss_names,
                                    X_validate, Y_class_validate, Y_loc_validate, n_classes):
    """Train a double headed model using training data stored in separate files.  
    Inputs:
        model | keras model | the model to be trained
        files | list | list of paths and filenames for the files used during training
        n_epochs | int | number of epochs to train for
        loss names | list | names of outputs of losses (e.g. "class_dense3_loss)
    Returns
        model | keras model | updated by the fit process
        metrics_loss | r2 array | columns are: total loss/class loss/loc loss /validate total loss/validate class loss/ validate loc loss
        metrics_class | r2 array | columns are class accuracy, validation class accuracy
        
    2019/03/25 | Written.  
    """
    import numpy as np
    import keras
    
    n_files_train = len(files)                                                              # get the number of training files
    
    metrics_class = np.zeros((n_files_train*n_epochs, 2))                                    # train class acuracy, valiate class accuracy
    metrics_loss = np.zeros((n_files_train*n_epochs, 6))                                     # total loss/class loss/loc loss /validate total loss/validate class loss/ validate loc loss
    
    for e in range(n_epochs):                                                                        # loop through the number of epochs
        for file_num, file in enumerate(files):                                   # for each epoch, loop through all files once
        
            data = np.load(file)
            X_batch = data['X']
            Y_batch_class = data['Y_class']
            Y_batch_loc = data['Y_loc']
            
            if n_classes !=  Y_batch_class.shape[1]:
                Y_batch_class = keras.utils.to_categorical(Y_batch_class, n_classes, dtype='float32')       # convert to one hot encoding (from class labels)
                
    
            history_train_temp = model.fit(X_batch, [Y_batch_class, Y_batch_loc], batch_size=32, epochs=1, verbose = 0)
             
            metrics_loss[(e*n_files_train)+file_num, 0] = history_train_temp.history['loss'][0]                        # main los    
            metrics_loss[(e*n_files_train)+file_num, 1] = history_train_temp.history[loss_names[0]][0]                 # class loss
            metrics_loss[(e*n_files_train)+file_num, 2] = history_train_temp.history[loss_names[1]][0]                 # localization loss
            metrics_class[(e*n_files_train)+file_num, 0] = history_train_temp.history['class_dense3_accuracy'][0]           # classification accuracy        
            print(f'Epoch {e}, file {file_num}: Loss = {round(metrics_loss[(e*n_files_train)+file_num, 0],0)}, '
                                       f'Class. loss = {round(metrics_loss[(e*n_files_train)+file_num, 1],2)}, '
                                       f'Class. acc. = {round(metrics_class[(e*n_files_train)+file_num, 0],2)}, '
                                         f'Loc. loss = {round(metrics_loss[(e*n_files_train)+file_num, 2],0)}')
            
            
            
            
        history_validate_temp = model.evaluate(X_validate, [Y_class_validate, Y_loc_validate], batch_size = 32, verbose = 0)
        metrics_loss[(e*n_files_train)+file_num, 3] = history_validate_temp[0]                                          # main loss
        metrics_loss[(e*n_files_train)+file_num, 4] = history_validate_temp[1]                                          # class loss
        metrics_loss[(e*n_files_train)+file_num, 5] = history_validate_temp[2]                                          # localisation loss
        metrics_class[(e*n_files_train)+file_num, 1] = history_validate_temp[3]                                         # classification  accuracy
        print(f'Epoch {e}, valid.: Loss = {round(metrics_loss[(e*n_files_train)+file_num, 3],0)}, '
                          f'Class. loss = {round(metrics_loss[(e*n_files_train)+file_num, 4],2)}, '
                          f'Class. acc. = {round(metrics_class[(e*n_files_train)+file_num, 1],2)}, '
                            f'Loc. loss = {round(metrics_loss[(e*n_files_train)+file_num, 5],0)}')
    

    metrics_class = np.hstack((metrics_loss[:,1:2], metrics_loss[:,4:5], metrics_class ))                       # class loss, validate class loss, class accuracy, validate class accuracy
    metrics_localisation = np.hstack((metrics_loss[:,2:3], metrics_loss[:,5:]))                                 # localisation loss, validate localisation loss, localisation accuracy, validate localisation accuracy
    metrics_combined_loss = np.hstack((metrics_loss[:,1:2], metrics_loss[:,3:4]))                               # class accuracy, validate class accuracy
    
    return model, metrics_class, metrics_localisation, metrics_combined_loss
    


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

def merge_and_rescale_data(synthetic_data_files, real_data_files, output_range = {'min':0, 'max':225}):
    """ Given a list of synthetic data files and real data files (usually the augmented real data),
    
    Inputs:
        synthetic_data_files | list of Paths or string | locations of the .pkl files containing the masked arrays
        reak_data_files      | list of Paths or string | locations of the .pkl files containing the masked arrays
        output_range         | dict                    | min and maximum of each channel in each image.  Should be set to suit the CNN being used.  
    Returns:
        .npz files in step_04_merged_rescaled_data
    History:
        2020_10_29 | MEG | Written

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
        np.savez(f'step_04_merged_rescaled_data/data_file_{out_file}.npz', X = X_rescale[:data_mid,:,:,:], Y_class= Y_class[:data_mid,:], Y_loc = Y_loc[:data_mid,:])           # save the first half of the data
        out_file += 1                                                                                                                                                   # after saving once, update
        np.savez(f'step_04_merged_rescaled_data/data_file_{out_file}.npz', X = X_rescale[data_mid:,:,:,:], Y_class= Y_class[data_mid:,:], Y_loc = Y_loc[data_mid:,:])           # save the second half of the data
        out_file += 1                                                                                                                                                   # and after saving again, update again.  
        print('Done.  ')
        
        # data_channel_checker(X, window_title = 'Merged')
        # data_channel_checker(X_real, window_title = 'X_real')
        # data_channel_checker(X_synth, window_title = 'X_synth')
        # data_channel_checker(X_rescale, window_title = 'X_rescale')
        

#%%

def custom_range_for_CNN(r4_array, min_max, mean_centre = False):
    """ Rescale a rank 4 array so that each channel's image lies in custom range
    e.g. input with range of (-5, 15) is rescaled to (-125 125) or (-1 1) for use with VGG16 
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
    """
    import numpy as np
    import numpy.ma as ma
    
    
    # do all possible fips - now have 4x as much data
    flips = ['up_down', 'left_right', 'both']                                       # the three possible types of flip
    X_aug = [X]                                                                     # make the original data an entry in a list
    Y_loc_aug = [Y_loc]                                                             # ditto
    for flip in flips:
        X_temp, Y_loc_temp = augment_flip(X, Y_loc, flip)                           # do the augmentaiton via one of the flips.  
        X_aug.append(X_temp)                                                        # append to the lis of data
        Y_loc_aug.append(Y_loc_temp)                                                # also append the new locations (as they change with flips)
    X_aug = ma.vstack(X_aug)                                                        # convert from a list back to one big array (ie stack along the first axis).  If we had n data, we now have 4n
    Y_loc_aug = np.vstack(Y_loc_aug)                                                # and for locations
    Y_class_aug = np.tile(Y_class, (4,1))                                           # classes don't change with flips etc. so can just be repeated.  
        
    # rotate
    X_aug = [X_aug]                                                                # convert back to an entry in a list
    Y_loc_aug = [Y_loc_aug]                                                        # same for locations
    for random_rotate in range(3):                                                 # 4n data will again become 16n data, as we will do 3 sets of rotations and keep the originals.  
        X_temp, Y_loc_temp = augment_rotate(X_aug[0], Y_loc_aug[0])                # rotate 4n data to make another 4n of data
        X_aug.append(X_temp)
        Y_loc_aug.append(Y_loc_temp)
    X_aug = ma.vstack(X_aug)                                                       # 4 entries of 4n data are converted to 16n data
    Y_loc_aug = np.vstack(Y_loc_aug)                                               # same for locations
    Y_class_aug = np.tile(Y_class_aug, (4,1))                                      # classes don't change so can just be copied

    # translate
    X_aug = [X_aug]                                                                # now have 16n data, which becomes first item in list
    Y_loc_aug = [Y_loc_aug]
    for random_translate in range(3):                                              # 16n data will again become 64n data, as we will do 3 sets of translations and keep the originals.  
        X_temp, Y_loc_temp = augment_translate(X_aug[0],  Y_loc_aug[0], max_translate = (20,20)) 
        X_aug.append(X_temp)
        Y_loc_aug.append(Y_loc_temp)
    X_aug = ma.vstack(X_aug)                                                        # 4 entries of 16n data are converted to 64n data
    Y_loc_aug = np.vstack(Y_loc_aug)
    Y_class_aug = np.tile(Y_class_aug, (4,1))    
    
    
    # shuffle along the first axis (otherwise with small values of n_data, we just get the first (and original) data back.  )
    data_dict = {'X'       : X_aug,                                                      # package the data and labels together into a dict
                 'Y_class' : Y_class_aug,
                 'Y_loc'   : Y_loc_aug}
    
    data_dict_shuffled = shuffle_arrays(data_dict)                                          # shuffle (so that these aren't in the order of the class labels)
    
    X_aug = data_dict_shuffled['X']                                                      # and unpack as this function doesn't use dictionaries
    Y_class_aug = data_dict_shuffled['Y_class']
    Y_loc_aug = data_dict_shuffled['Y_loc']
    
    
    return X_aug[:n_data], Y_class_aug[:n_data], Y_loc_aug[:n_data]                     # return, and select only the desired data.  

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
