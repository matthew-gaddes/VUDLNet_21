#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:47:45 2020

@author: matthew
"""

import pdb

#%%

def plot_all_metrics(batch_metrics, epoch_metrics, metrics = None, title = 'Training metrics', 
                     two_column = False, out_path = None, y_epoch_start = 0):
    """ Given a dict of metrics for every batch and for every epoch, plot the combination of the two.  
    
    Inputs:
        batch_metrics | dist of lists | names of metrics are keys and list of metrics values are items.  There are as many entries as n_epochs * n _batches (i.e. for every batch that was used in training)
        epoch_metrics | dict of lists | standard metrics saved each epoch by Keras.                       There are as many entries as n_epochs  
        metrics | None or list | names of metrics to plot (i.e. so we don't have to plot all fo them).  If None, just plot all.
        title | string | figure and window title.  
        two_column | boolean | If True, plots are in two columns.  
        y_epoch_start | float | epoch number to start y values of plot on (i.e. to crop out very different values in first epoch).  Can be fractions of an epoch.  
    Returns:
        Figure
    History:
        2021_11_11 | MEG | Written
        2022_11_01 | MEG | Update so that y limits are adjusted to avoid first epoch values that are very different to rest
        
    """
    
    import matplotlib.pyplot as plt    
    import numpy as np
        
    if metrics is None:                                                                                   # if no specific metrics are requested
        metrics = list(batch_metrics.keys())                                                              # just get them all from the batch metrics
    
    n_losses_total = len(batch_metrics[list(batch_metrics.keys())[0]])
    n_epochs = len(epoch_metrics['loss'])
    n_batches = int(n_losses_total / n_epochs)                                                            # get the number of entries in the first item of the batch_metrics (which is epochs x n_batches), then divided by epochs to get n_batches
    n_metrics = len(metrics)
    
    if two_column == False:
        fig1, axes = plt.subplots(1, n_metrics, figsize = (28,7))                                           # many rows, one column
    else:
        fig1, axes = plt.subplots(int(np.ceil(n_metrics/2)), 2, figsize = (14,7))                           # seems to wor, first bit works out how many rows we need.  
    fig1.canvas.manager.set_window_title(title)
    
    xvals_batch = np.arange(0, n_losses_total)                                                           # for every batch in every epoch, the xvalue to plot it at
    xvals_epoch = xvals_batch[::-1][::n_batches][::-1]                                                      # for every epoch, the x value to plot it at (i.e. every n_epochs)
    
    
    for plot_n, metric in enumerate(metrics):
        ax = np.ravel(axes)[plot_n]                                                                     # get the ax to plot on
        ax.scatter(xvals_batch, batch_metrics[metric], c = 'k', marker = '.', alpha = 0.5)              # plot for each batch
        ax.scatter(xvals_epoch, epoch_metrics[metric], c = 'r', marker = 'o')                           # plot for each epoch
        ax.set_ylabel(metric)
        ax.grid(True)

        if 'accuracy' in metric:                                                                                        # accuracy should increase and can't get higher than 1.  Adjust lower limit
            ax.set_ylim(bottom = batch_metrics[metric][xvals_batch[int(y_epoch_start * n_batches)]], top = 1)      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
        else:                                                                                                           # loss should decrease and can't get lower than 0.  Adjust upper limi.t  
            ax.set_ylim(bottom = 0, top = (batch_metrics[metric][xvals_batch[int(y_epoch_start * n_batches)]]))      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
        ax.set_xlim(left = 0)
        
        if 'accuracy' in metric:                                                              # if accuracy is used in the metric, assume it's an accuracy and therefore
            ax.set_ylim(top = 1)                                                              # maxes out at 1
            
        ax.set_xticks(xvals_epoch)                                                            # change so a tick only after each epoch (and not each file)
        ax.set_xticklabels(np.arange(1,n_epochs+1, 1))                                          # number ticks
        ax.set_xlabel('Epoch number')
    
    if two_column and (not (n_metrics/2).is_integer()):                                        # if its two column and we didn't have an even number of metrics, delete the bottom right (left over one)
        np.ravel(axes)[-1].set_visible(False)
        
    if out_path is not None:
        fig1.savefig(out_path)                                                                   # 


#%%

# def custom_training_history(metrics, n_epochs, title = None):
#     """Plot training line graphs for loss and accuracy.  Loss on the left, accuracy on the right.  
#     Inputs
#         metrics | r2 array | (n_files * n_epochs) x 2 or 4 matrix,  train loss|validate loss | train accuracy|validate accuracy.  If no accuracy, only 2 columns
#         n_epochs | int | number of epochs model was trained for
#         title | string | title
#     Returns:
#         Figure
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
    
#     if metrics.shape[1] == 4:           # detemrine if we have accuracy as well as loss
#         accuracy_flag = True
#     else:
#         accuracy_flag = False
        
    
#     n_files = metrics.shape[0] / n_epochs
#     # Figure output
#     fig1, axes = plt.subplots(1,2)
#     fig1.canvas.set_window_title(title)
#     fig1.suptitle(title)
#     xvals = np.arange(0,metrics.shape[0])
#     validation_plot = np.ravel(np.argwhere(metrics[:,1] > 1e-10))                   # fewer validation data; find which ones to plot
    
#     axes[0].plot(xvals, metrics[:,0], c = 'k')                                       # training loss
#     axes[0].plot(xvals[validation_plot], metrics[validation_plot,1], c = 'r')        # validation loss
#     axes[0].set_ylabel('Loss')
#     axes[0].legend(['train', 'validate'], loc='upper left')
#     axes[0].axhline(y=0, color='k', alpha=0.5)
    
#     if accuracy_flag:
#         axes[1].plot(xvals, metrics[:,2], c = 'k')                                       # training accuracy
#         axes[1].plot(xvals[validation_plot], metrics[validation_plot,3], c = 'r')        # validation accuracy
#         axes[1].set_ylim([0,1])
#         axes[1].set_ylabel('Accuracy')
#         axes[1].yaxis.tick_right()
#         axes[1].legend(['train', 'validate'], loc='upper right')
        
#     #
#     titles = ['Training loss', 'Training accuracy']
#     for i in range(2):
#         axes[i].set_title(titles[i])
#         axes[i].set_xticks(np.arange(0,metrics.shape[0],2*n_files))                 # change so a tick only after each epoch (and not each file)
#         axes[i].set_xticklabels(np.arange(0,n_epochs, 2))                                  # number ticks
#         axes[i].set_xlabel('Epoch number')

#     if not accuracy_flag:
#         axes[1].set_visible(False)



#%%

def open_datafile_and_plot(file_path, n_data = 15, rad_to_m_convert = False,
                           window_title = None):
    """ A function to open a .pkl file of ifgs and quickly plot the first n_data ifgs.  
    Inputs:
        pkl_path | path | path to pkl file to be opened
        n_data | int | the first n_data data in the pkl will be plotted.  
        rad_to_m_convert | boolean | If True, data are in sentinel-1 rads and are convereted to .  
        window_title        | None or string | Sets the title of the window, if not None
    Returns:
        figure
    History:
        2020/10/28 | MEG | Written
        2020/11/11 | MEG | Update to handle either .pkl or .npz
        2021/09_23 | MEG | Update to only use Pathlib paths for the file_path (using the .suffix attribute)
    """
    import pickle
    import numpy as np
    from vudlnet21.plotting import plot_data_class_loc_caller
    
    s1_wav = 0.055465763                                                                    # in metres
    
    if file_path.suffix == '.pkl':                                                          # if it's a .pkl
        with open(file_path, 'rb') as f:                                                    # open the file
            X = pickle.load(f)                                                              # and extract data (X) and labels (Y)
            Y_class = pickle.load(f)
            Y_loc = pickle.load(f)
        f.close()
    elif file_path.suffix == '.npz':                                                        # if it's a npz
        data = np.load(file_path)                                                           # load it
        X = data['X']                                                                       # and extract data (X) and labels (Y)
        Y_class = data['Y_class']
        Y_loc = data['Y_loc']
    else:                                                                                   # no other file types are currently supported
        raise Exception(f"Error!  File was not understood as either a .pkl or a .npz so exiting.  ")
    
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
    History:
        2019/??/?? | MEG | Written
        2020/11/11 | MEG | Fix bug in how window_title is handled when there is only one plot
    """
    import numpy as np
    
    n_data = X_m.shape[0]
    n_plots = int(np.ceil(n_data/15))
    all_args = np.arange(0,n_data)
    
    if n_plots == 1:
        plot_data_class_loc(X_m, all_args, classes, locs, classes_predicted, locs_predicted, source_names, point_size, figsize, window_title)
    
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
    
    from vudlnet21.aux import add_square_plot, remappedColorMap, centre_to_box
    
    cmap_noscaling = plt.get_cmap('coolwarm')
    label_fs = 8
    n_rows = 3
    n_cols = 5
    n_plots = int(len(plot_args))
    f1, axes = plt.subplots(n_rows, n_cols, figsize = figsize)
    if window_title is not None:
        f1.canvas.set_window_title(window_title)
        
    
    
    for n_plot in range(n_plots):                                           # loop through each plot arg (that is the number of ifgs to plot)
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
            
            
#            pdb.set_trace()    
            
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
