#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:30:31 2022

@author: matthew
"""

#%%



def rescale_timeseries(displacement_r3, rescale_factor = 1.0):
    """Given a time series as a dict in the usual form, rescale it by a given factor.  Not tested for downsampling.  
    Inputs:
        displacment_r3 | dict | cumulative, lons, lats, mask, dem.  
        Rescale factor | float | factor to rescale by  
    Returns:
        displacment_r3_rescaled | dict | as above, but all rescaled.  
    History:
        2022_05_23 | MEG | Written.  
    """
    
    import numpy as np
    import numpy.ma as ma
    from skimage.transform import rescale

    #print(f"Rescaling the time series.  ")

    n_acq = displacement_r3['cumulative'].shape[0]
    displacement_r3_rescaled = {}
    
    mask = displacement_r3['cumulative'].mask[0,]                                                              # get a slice of the mask (as it's rank3, but we're using a time series with consistent pixels so it might as well be rank 2)
    displacement_r3_rescaled['mask'] = rescale(mask, rescale_factor, order = 0)                                                   # order 0 as boolean so don't want any interpolate
    try:
        displacement_r3_rescaled['dem'] = rescale(displacement_r3['dem'], rescale_factor)                          # 
    except:
        pass
    
    for ifg_n, ifg in enumerate(displacement_r3['cumulative']):
        ifg_rescaled = rescale(ifg, rescale_factor)
        if ifg_n == 0:
            displacement_r3_rescaled['cumulative'] = ma.zeros((n_acq, ifg_rescaled.shape[0],  ifg_rescaled.shape[1]))
        displacement_r3_rescaled['cumulative'] [ifg_n,] = ma.array(ifg_rescaled, mask = displacement_r3_rescaled['mask'])
        
    _, ny_rescaled, nx_rescaled = displacement_r3_rescaled['cumulative'].shape
    
    lons_mg, lats_mg = np.meshgrid(np.linspace(displacement_r3['lons'][0,0], displacement_r3['lons'][0,-1], nx_rescaled),           # x 
                                   np.linspace(displacement_r3['lats'][0,0], displacement_r3['lats'][-1,0], ny_rescaled))           # then y
    
    displacement_r3_rescaled['lons'] = lons_mg
    displacement_r3_rescaled['lats'] = lats_mg
    
    return displacement_r3_rescaled
        

#%%

def random_cropping(ifg, out_resolution = 224, def_loc = None):
    """ Given an input ifg that is larger than the crop required, make 9 crops of the ifg to the new scale.  
    If no labels (def_loc) are provided, then these are quasi random to sample top left/ middle left, bottom left etc (i.e.9)
    If labels are provided, it's random so that all of the label is retained.  
    
    Inputs:
        ifg | rank 2 masked array | interferogram.  
        out_resolution | int | output side length (output is square)
        def_loc | rank 2 array | closed polygon of deformation, x then y, in pixels.  e.g.:
                                                                                            array([[168, 183],
                                                                                                   [243, 183],
                                                                                                   [243, 233],
                                                                                                   [168, 233],
                                                                                                   [168, 183]])
                                                                                            
    Returns:
        ifg_cropped | rank 3 array | out_resolution x out_resolution x 9
        Y_loc | rank 2 array | 9 x 4, x centre y centre x half width, y half width.  
        
    History:
        2022_05_16 | MEG | Written
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    
    bands = [0, 1/3, 2/3, 3/3]
    
    # # inputs sanity check
    # f, ax = plt.subplots()
    # ax.imshow(ifg)
    # ax.plot(def_loc[:,0], def_loc[:,1])
    
    ifg_cropped = ma.zeros((out_resolution, out_resolution, 9))             # initialise
    
    #pdb.set_trace()                
    
    ny, nx = ifg.shape
    y_padding = ny - out_resolution
    x_padding = nx - out_resolution
    
    if def_loc is None:                                                                                                 # if no deformaiton label, cropping is easy as can take any part of the data.  
        data_n = 0
        for yband in np.arange(3):
            for xband in np.arange(3):
                y_start = np.random.randint(low = bands[yband] * y_padding , high = bands[yband + 1] * y_padding )
                x_start = np.random.randint(low = bands[xband] * x_padding , high = bands[xband + 1] * x_padding )
                ifg_cropped[:,:,data_n] = ifg[y_start: y_start + out_resolution, x_start: x_start + out_resolution]
                data_n += 1
        return ifg_cropped
                
    else:                                                                                                               # but, if we do have location information, cropping is harder, and completely random.  
        Y_loc = np.zeros((9, 4))                                                                                        # initialise to store locations for each crop.  
        def_x_start = np.min(def_loc[:,0])
        def_x_stop = np.max(def_loc[:,0])
        def_x_centre = int(np.mean([def_x_start, def_x_stop]))                                                          # 
        def_x_half_width = (def_x_stop - def_x_start)/2                                                                 # this won't be changed by cropping
        def_y_start = np.min(def_loc[:,1])
        def_y_stop = np.max(def_loc[:,1])
        def_y_centre = int(np.mean([def_y_start, def_y_stop]))                                              
        def_y_half_width = (def_y_stop - def_y_start)/2                                                                 # also won't be changed by cropping
        
        x_low = def_x_stop - out_resolution                                         # the lowest that x can start at and the max x deformation still be in the image
        if x_low < 0:
            x_low = 0 
        x_high = def_x_start
        if x_high > (nx - out_resolution):
            x_high = nx - out_resolution    
        
        
        y_low = def_y_stop - out_resolution                                         # the lowest that x can start at and the max x deformation still be in the image
        if y_low < 0:
            y_low = 0 
        y_high = def_y_start
        if y_high > (ny - out_resolution):
            y_high = ny - out_resolution
        
            
        data_n = 0 
        for i in range(9):
            #pdb.set_trace()
            x_start = np.random.randint(x_low, x_high)            # x_high is just the lowest value for the deformaiton.  
            y_start = np.random.randint(y_low, y_high)            #
            
            ifg_cropped[:,:, data_n] = ifg[y_start : y_start + out_resolution, x_start : x_start + out_resolution]
            Y_loc[data_n, :] = np.array([def_x_centre- x_start, def_y_centre - y_start, def_x_half_width, def_y_half_width])
            data_n +=1 
        return ifg_cropped, Y_loc
        



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

