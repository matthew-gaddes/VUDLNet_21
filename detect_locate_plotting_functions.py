#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:47:45 2020

@author: matthew
"""

#%%

def open_pkl_and_plot(pkl_path, n_data = 15, rad_to_m_convert = False,
                      window_title = None):
    """ A function to open a .pkl file of ifgs and quickly plot the first n_data ifgs.  
    Inputs:
        pkl_path | path or string | path to pkl file to be opened
        n_data | int | the first n_data data in the pkl will be plotted.  
        rad_to_m_convert | boolean | If True, data are in sentinel-1 rads and are convereted to .  
        window_title        | None or string | Sets the title of the window, if not None
    Returns:
        figure
    History:
        2020/10/28 | MEG | Written
    """
    import pickle
    import numpy as np
    from detect_locate_plotting_functions import plot_data_class_loc_caller
    
    s1_wav = 0.055465763                                                                            # in metres
    
    with open(pkl_path, 'rb') as f:                                              # the subset hosted in the github repo.  
        X = pickle.load(f)
        Y_class = pickle.load(f)
        Y_loc = pickle.load(f)
    f.close()
    
    if type(X) is dict:
        print('The variable X contained in this dictionary is dictionary, which usually means it contains the same data stored in various formats.  '
              'Taking the uuu phase and converting it to metres.  ')
        X = X['uuu']
        rad_to_m_convert = True
    
    if rad_to_m_convert:
        X = X *(s1_wav / (4 * np.pi))                                                                                              # convert from unwrapped phase to metres (for Sentinel-1)
        
    plot_data_class_loc_caller(X[:n_data,], Y_class[:n_data,], Y_loc[:n_data,], source_names = ['dyke', 'sill', 'no def'],
                               window_title = window_title)    

#%%

def plot_data_class_loc_caller(X_m, classes=None, locs=None, classes_predicted=None, locs_predicted=None, 
                               source_names = None, point_size=5, figsize = (6,4), window_title = None):
    """ A function to call plot_data_class_loc to plot more than 15 ifgs.  
    
    Inputs as per 'plot_data_class_loc'   
    """
    import numpy as np
    
    n_data = X_m.shape[0]
    n_plots = int(np.ceil(n_data/15))
    all_args = np.arange(0,n_data)
    
    if n_plots == 1:
        plot_data_class_loc(X_m, all_args, classes, locs, classes_predicted, locs_predicted, source_names, point_size, figsize)
    
    else:
        for n_plot in np.arange(n_plots-1):
            plot_args = np.arange(n_plot*15, (n_plot*15)+15)
            plot_data_class_loc(X_m, plot_args, classes, locs, classes_predicted, locs_predicted, source_names, point_size, figsize, window_title)
        
        plot_args = np.arange((n_plot+1)*15, n_data)                                      # plot the last one that might have some blank spaces
        plot_data_class_loc(X_m, plot_args, classes, locs, classes_predicted, locs_predicted, source_names, point_size, figsize, window_title)





def plot_data_class_loc(data, plot_args, classes=None,           locs=None, 
                                         classes_predicted=None, locs_predicted=None, 
                        source_names = None, point_size=5, figsize = (6,4), window_title = None):
    """A figure to plot some data, add the class and predicted class, and add the location and predicted location
    Inputs: 
        X_m | rank 4 masked array |ifgs in metres, with water and incoherence masked 
              rank 4 array |        can be CNN input with no masking and any range too.                        
        plot_args | rank 1 array | Which data to plot (e.g.: array([ 0,  1,  2,  3,  4]))
        classes             | array | one hot encoding
        locs                | rank 2 array | nx4,  locations of deforamtion, columns are x, y, x half width, y half width
        classes_predicted   | array | as per classes
        locs_predicted      | rank 2 array | as per locs
        source_names        |
        point_size          | int | size of the dot at the centre of the deformation
        figsize             | tuple | Size of figure in inches.  
        window_title        | None or string | Sets the title of the window, if not None
    Returns:
        Figure
    History:
        2019/??/?? | MEG | Written
        2020/10/21 | MEG | Moved to the detect_locate github repo.  
        2020/10/21 | MEG | Update so that no colorbars don't change the units (previously *100 to convert m to cm)
        2020/10/28 | MEG | Update docs
    """    
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import numpy as np
    import numpy.ma as ma
    
    from aux_functions import add_square_plot, remappedColorMap, centre_to_box
    
    cmap_noscaling = plt.get_cmap('coolwarm')
    label_fs = 8
    n_rows = 3
    n_cols = 5
    n_plots = int(len(plot_args))
    f1, axes = plt.subplots(n_rows, n_cols, figsize = figsize)
    if window_title is not None:
        f1.canvas.set_window_title(window_title)
    for n_plot in range(n_plots):                                           # loop through each plot arg
        axe = np.ravel(axes)[n_plot]                                        # convert axes to a rank 1 so that it's easy to index them as we loop through the plots
        
        #1: Draw the ifg (each ifg has its own colourscale)
        ifg_min = ma.min(data[plot_args[n_plot], :,:,0])                                                                   # min of ifg being plotted
        ifg_max = ma.max(data[plot_args[n_plot], :,:,0])                                                                   # max
        ifg_mid = 1 - ifg_max/(ifg_max + abs(ifg_min))                                                                     # mid point
        cmap_scaled = remappedColorMap(cmap_noscaling, start=0, midpoint=ifg_mid, stop=1.0, name='cmap_noscaling')         # rescale cmap to be the correct size
        axe.imshow(data[plot_args[n_plot], :,:,0], cmap = cmap_scaled)                                                     # plot ifg with camp
        
        #2: Draw a colorbar
        axe_cbar = inset_axes(axe, width="40%", height="3%", loc=8, borderpad=0.3)                                      # colourbar is plotted within the subplot axes
        norm2 = mpl.colors.Normalize(vmin=ifg_min, vmax=ifg_max)                                                        # No change to whatever the units in X are.
        cb2 = mpl.colorbar.ColorbarBase(axe_cbar, cmap=cmap_scaled, norm = norm2, orientation = 'horizontal')
        cb2.ax.xaxis.set_ticks_position('top')
        if np.abs(ifg_max) > np.abs(ifg_min):
            cb2.ax.xaxis.set_ticks([np.round(0,0), np.round(ifg_max, 3)])
        else:
            cb2.ax.xaxis.set_ticks([np.round(ifg_min,3), np.round(0,0)])       
        
        #3: Add labels/locations
        if locs is not None:
            start_stop_locs = centre_to_box(locs[plot_args[n_plot]])                                         # covert from centre width notation to start stop notation, # [x_start, x_stop, Y_start, Y_stop]
            add_square_plot(start_stop_locs[0], start_stop_locs[1], 
                            start_stop_locs[2], start_stop_locs[3], axe, colour='k')     # box around deformation
            axe.scatter(locs[plot_args[n_plot], 0], locs[plot_args[n_plot], 1], s = point_size, c = 'k')
            
        if locs_predicted is not None:
            start_stop_locs_pred = centre_to_box(locs_predicted[plot_args[n_plot]])                                         # covert from centre width notation to start stop notation, # [x_start, x_stop, Y_start, Y_stop]
            add_square_plot(start_stop_locs_pred[0], start_stop_locs_pred[1], 
                            start_stop_locs_pred[2], start_stop_locs_pred[3], axe, colour='r')     # box around deformation
            axe.scatter(locs_predicted[plot_args[n_plot], 0], locs_predicted[plot_args[n_plot], 1], s = point_size, c = 'r')
        
        if classes is not None and classes_predicted is None:                           # if only have labels and not predicted lables
            label = np.argmax(classes[plot_args[n_plot], :])                               # labels from the cnn
            axe.set_title(f'Ifg: {plot_args[n_plot]}, Label: {source_names[label]}' , fontsize=label_fs)
            
        elif classes_predicted is not None and classes is None:                           # if only have predicted labels and not labels
            label_model = np.argmax(classes_predicted[plot_args[n_plot]])                               # original label
            value_model = str(np.round(np.max(classes_predicted[plot_args[n_plot]]), 2))                                  # max value from that row
            axe.set_title(f'Ifg: {plot_args[n_plot]}, CNN label: {source_names[label_model]} ({value_model})', fontsize=label_fs)
        else:                                                                               # if we have both predicted labels and lables
            label = np.argmax(classes[plot_args[n_plot], :])                               # labels from the cnn
            label_model = np.argmax(classes_predicted[plot_args[n_plot]])                                         # original label
            value_model = str(np.round(np.max(classes_predicted[plot_args[n_plot]]), 2))                                  # max value from that row
            axe.set_title(f'Ifg: {plot_args[n_plot]}, Label: {source_names[label]}\nCNN label: {source_names[label_model]} ({value_model})', fontsize=label_fs)
        
        
        axe.set_ylim(top = 0, bottom = data.shape[1])
        axe.set_xlim(left = 0, right= data.shape[2])
        axe.set_yticks([])
        axe.set_xticks([])
        
        if n_plots < 15:                                                        # remove any left over/unused subplots
            axes_to_del = np.ravel(axes)[(n_plots):]
            for axe_to_del in axes_to_del:
                axe_to_del.set_visible(False)

            
#%%
