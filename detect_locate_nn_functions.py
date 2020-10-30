#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:07 2020

@author: matthew
"""

def custom_range_for_CNN(r4_array, custom_range, mean_centre = False):
    """ Rescale a rank 4 array so that each channels image lies in custom range
    e.g. input with range of [-5 15] is rescaled to [-125 125] or [-1 1] for use with VGG16 
    Inputs:
        r4_array | r4 masked array | works with masked arrays?  
    
    2019/03/20 | now includes mean centering so doesn't stretch data to custom range.  
                Instead only stretches until either min or max touches, whilst mean is kept at 0
    """
    import numpy as np
    
    
    if mean_centre:
        im_channel_means = np.mean(r4_array, axis = (1,2))                                                  # get the average for each image (in all thre channels)
        im_channel_means = expand_to_r4(im_channel_means, r4_array[0,:,:,0].shape)                                                   # expand to r4 so we can do elementwise manipulation
        r4_array -= im_channel_means                                                                        # do mean centering    

    im_channel_abs_max = np.max(np.abs(r4_array), axis = (1,2))                                         # 
    im_channel_abs_max = expand_to_r4(im_channel_abs_max, r4_array[0,:,:,0].shape)    
    r4_array = (custom_range/2)* (r4_array/im_channel_abs_max)

    return r4_array    

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
    flips = ['up_down', 'left_right', 'both']
    X_aug = [X]                                                                     # make this an entry in a list
    Y_loc_aug = [Y_loc]                                                             # ditto
    for flip in flips:
        X_temp, Y_loc_temp = augment_flip(X, Y_loc, flip)                           # do the augmentaiton via one of the flips.  
        X_aug.append(X_temp)                                                        # append to the lis of data
        Y_loc_aug.append(Y_loc_temp)                                                # also append the new locations (as they change with flips)
    X_aug = ma.vstack(X_aug)                                                        # convert from a list back to one big array (ie stack along the first axis)
    Y_loc_aug = np.vstack(Y_loc_aug)                                                # and for locations
    Y_class_aug = np.tile(Y_class, (4,1))                                           # classes don't change with flips etc. so can just be repeated.  
        
    # rotate
    X_aug = [X_aug]                                                                # convert back to an entry in a list
    Y_loc_aug = [Y_loc_aug]
    for random_rotate in range(3):
        X_temp, Y_loc_temp = augment_rotate(X_aug[0], Y_loc_aug[0])
        X_aug.append(X_temp)
        Y_loc_aug.append(Y_loc_temp)
    X_aug = ma.vstack(X_aug)
    Y_loc_aug = np.vstack(Y_loc_aug)
    Y_class_aug = np.tile(Y_class_aug, (4,1))    

    # translate
    X_aug = [X_aug]                                                                # first one is just the original
    Y_loc_aug = [Y_loc_aug]
    for random_translate in range(3):
        X_temp, Y_loc_temp = augment_translate(X_aug[0],  Y_loc_aug[0], max_translate = (20,20))
        X_aug.append(X_temp)
        Y_loc_aug.append(Y_loc_temp)
    X_aug = ma.vstack(X_aug)
    Y_loc_aug = np.vstack(Y_loc_aug)
    Y_class_aug = np.tile(Y_class_aug, (4,1))    
    
    return X_aug[:n_data], Y_class_aug[:n_data], Y_loc_aug[:n_data]

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
    
    data_dict_shuffled = shuffle_arrays(data_dict)                                          # shuffle (so that these aren't in the order of the class labe)
    
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
