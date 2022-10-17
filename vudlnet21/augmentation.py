#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:30:31 2022

@author: matthew
"""

import pdb

#%%



def augment_data(X, Y_class, Y_loc, n_data = 500, rotate = True, translate = True):
    """ A function to augment data and presserve the location label for any deformation.  
    Note that n_data is not particularly intelligent as many more data may be generated,
    and only n_data returned, so even if n_data is low, the function can still be slow.  
    Inputs:
        X           | rank 4 array | data.  
        Y_class     | rank 2 array | One hot encoding of class labels
        Y_loc       | rank 2 array | locations of deformation
        n_data      | int | number of data to be returned.  Usually higher than the number of data passed to the fucntion.  
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
    
    from vudlnet21.aux import shuffle_arrays
    
    flips = ['none', 'up_down', 'left_right', 'both']                                       # the three possible types of flip

    # 0: get the correct nunber of data    
    n_ifgs = X.shape[0]
    data_dict = {'X'       : X,                                                      # package the data and labels together into a dict
                 'Y_class' : Y_class,
                 'Y_loc'   : Y_loc}
    
    # 0: increase the number of data to the required output size.  
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
    if rotate:
        X_aug, Y_loc_aug = augment_rotate(X_aug, Y_loc_aug)                # rotate 
    
    # 3: Do the translations
    if translate:
        X_aug, Y_loc_aug = augment_translate(X_aug,  Y_loc_aug, max_translate = (20,20)) 
        
    return X_aug, Y_class_aug, Y_loc_aug                    # return, and select only the desired data.  
 


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
    
    if flip == 'up_down':                                       # conver the string input to a value
        X_flip = X[:,::-1,:,:]                              # reverse in dim 2 which is y
        Y_loc_flip[:,1] = X.shape[1] - Y_loc_flip[:,1]
    elif flip == 'left_right':
        X_flip = X[:,:,::-1,:]                              # reverse in dim 3 which is x
        Y_loc_flip[:,0] = X.shape[2] - Y_loc_flip[:,0]      # flipping horizontally
    elif flip == 'both':
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

# def choose_for_augmentation(X, Y_class, Y_loc, n_per_class):
#     """A function to randomly select only some of the data, but in  fashion that ensures that the classes are balanced 
#     (i.e. there are equal numbers of each class).  Particularly useful if working with real data, and the classes
#     are usually very unbalanced (lots of no_def lables, normally).  
    
#     Only works if the data are all in one file (i.e. can be read to RAM in one go)
    
#     Inputs:
#         X           | rank 4 array | data.  
#         Y_class     | rank 2 array | One hot encoding of class labels
#         Y_loc       | rank 2 array | locations of deformation
#         n_per_class | int | number of data per class. e.g. 3
#     Returns:
#         X_sample           | rank 4 array | data.  
#         Y_class_sample     | rank 2 array | One hot encoding of class labels
#         Y_loc_sample       | rank 2 array | locations of deformation
#     History:
#         2019/??/?? | MEG | Written
#         2019/10/28 | MEG | Update to handle dicts
#         2020/10/29 | MEG | Write the docs.  
#         2020/10/30 | MEG | Fix bug that was causing Y_class and Y_loc to become masked arrays.  
        
#     """
#     import numpy as np
#     import numpy.ma as ma
    
#     n_classes = Y_class.shape[1]                                                         # only works if one hot encoding is used
#     X_sample = []
#     Y_class_sample = []
#     Y_loc_sample = []
    
#     for i in range(n_classes):                                                              # loop through each class
#         args_class = np.ravel(np.argwhere(Y_class[:,i] != 0))                               # get the args of the data of this label
#         args_sample = args_class[np.random.randint(0, len(args_class), n_per_class)]        # choose n_per_class of these (ie so we always choose the same number from each label)
#         X_sample.append(X[args_sample, :,:,:])                                              # choose the data, and keep adding to a list (each item in the list is n_per_class_label x ny x nx n chanels)
#         Y_class_sample.append(Y_class[args_sample, :])                                      # and class labels
#         Y_loc_sample.append(Y_loc[args_sample, :])                                          # and location labels
    
#     X_sample = ma.vstack(X_sample)                                                          # maskd array, merge along the first axis, so now have n_class x n_per_class of data
#     Y_class_sample = np.vstack(Y_class_sample)                                              # normal numpy array, note that these would be in order of the class (ie calss 0 first, then class 1 etc.  )
#     Y_loc_sample = np.vstack(Y_loc_sample)                                                  # also normal numpy array
    
#     data_dict = {'X'       : X_sample,                                                      # package the data and labels together into a dict
#                  'Y_class' : Y_class_sample,
#                  'Y_loc'   : Y_loc_sample}
    
#     data_dict_shuffled = shuffle_arrays(data_dict)                                          # shuffle (so that these aren't in the order of the class labels)
    
#     X_sample = data_dict_shuffled['X']                                                      # and unpack as this function doesn't use dictionaries
#     Y_class_sample = data_dict_shuffled['Y_class']
#     Y_loc_sample = data_dict_shuffled['Y_loc']
    
#     return X_sample, Y_class_sample, Y_loc_sample

