#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:34:22 2021

@author: matthew
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
import sys
import pickle
import pdb
import datetime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input, Model
from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model

from vudlnet21.neural_net import define_two_head_model, numpy_files_sequence, save_model_each_epoch, save_batch_loss, numpy_files_sequence_old, build_2head_from_epochs
#from vudlnet21.neural_net import custom_range_for_CNN
from vudlnet21.plotting import plot_all_metrics, plot_data_class_loc_caller
from vudlnet21.file_handling import file_merger




#%% Things to set

dependency_paths = {'deep_learning_tools'   : '/home/matthew/university_work/20_deep_learning_tools'}                                      # Available from Github: https://github.com/matthew-gaddes/Deep-Learning-Tools

# settings that need to be the same as part 1
project_outdir = Path('./')
synthetic_ifgs_settings = {'defo_sources' : ['dyke', 'sill', 'no_def']}               # deformation patterns that will be included in the dataset.  
cnn_settings = {'input_range'       : {'min':-1, 'max':1}}

   
# step 06 (train the fully connected part of the network)
batch_size                     = 25                                              # as data files have 500 in them, this divides in cleanly (i.e. no half batches).  Not tested in case that it doesn't divide in cleanly, perhaps last batch is just smaller and faster?  
fc_loss_weights                   = [1., 1.]                                                      # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
n_epochs_fc                       = 10                                                                    # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
cnn_settings = {}
cnn_settings['n_files_train']     = 70                                              # the number of files that will be used to train the network
cnn_settings['n_files_validate']  = 5                                              # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
cnn_settings['n_files_test']      = 5                                               # the number of files held back for testing.  
n_epoch_class = 4
n_epoch_loc = 9



# step 07 (fine-tune the 5th block and the fully connected part of the network):
n_epochs_b5             = 5                                                           # the number of epochs to fine-tune for (ie. the number of times all the training data are passed through the model)
block5_loss_weights     = [0.05, 0.95]                                                # classification loss (e.g. 0.126) then localisation loss (e.g. 300) weighting
block5_loss_weights     = [2000, 1]                                                # classification loss then localisation loss weighting
block5_lr               = 1.e-5                                                                  # a pretty important parameter.  We have to set a learning rate manually as an adaptive approach (e.g. NADAM) will be high initially, and therefore make large updates that will wreck the model (as we're just fine-tuning a model so have something good to start with)
step_07_dir             = project_outdir / "step_07_fine_tune_training"

# step 08: Predict on the testing data.  
n_plot = 30                                                                         # number of test data to plot after and step 07


# step 09: Predict on the real testing data
#volcnet_dir             = Path('/home/matthew/university_work/02_neural_network_python/08_VolcNet_copy/')
#n_plot = 30                                                                         # number of test data to plot after and step 07

#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
############################################################                              Shouldn't need to change below here                    ############################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


sys.path.append(dependency_paths['deep_learning_tools'])
import deep_learning_tools
from deep_learning_tools.file_handling import file_list_divider

#%% Prepare the data that is used for training both steps.  


data_files = sorted(glob.glob(str(project_outdir / 'step_05_merged_rescaled_data' / '*npz')), key = os.path.getmtime)                             # make list of data files
data_files_train, data_files_validate, data_files_test = file_list_divider(data_files, cnn_settings['n_files_train'],                                                                            cnn_settings['n_files_validate'], cnn_settings['n_files_test'])        # divide the files into train, validate and test    


if len(data_files) < (cnn_settings['n_files_train'] + cnn_settings['n_files_validate'] + cnn_settings['n_files_test']):
    raise Exception(f"There are {len(data_files)} data files, but {cnn_settings['n_files_train']} have been selected for training, "
                    f"{cnn_settings['n_files_validate']} for validation, and {cnn_settings['n_files_test']} for testing, "
                    f"which sums to greater than the number of data files.  Perhaps adjust the number of files used for the training stages? "
                    f"For now, exiting.")

print(f"    There are {len(data_files)} data files.  {len(data_files_train)} will be used for training, "         # print to terminal status on how many files will be used etc.  
      f"{len(data_files_validate)} for validation, and {len(data_files_test)} for testing.  ")

train_generator = numpy_files_sequence(data_files_train, batch_size = batch_size)                                          # sequence using the training data
validation_generator = numpy_files_sequence(data_files_validate, batch_size = batch_size)                                  # and validation data

X_test, Y_class_test, Y_loc_test = file_merger(data_files_test)                                                            # load to RAM.  

#%% step 06: Train the fully connected part of the CNN

print('\nStep 05: Training the fully connected part of the CNN.')


vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
output_class, output_loc = define_two_head_model(vgg16_block_1to5.output, len(synthetic_ifgs_settings['defo_sources']))          # build the fully connected part of the model, and get the two model outputs
vgg16_2head = Model(inputs=vgg16_block_1to5.input, outputs=[output_class, output_loc])                                           # define the full model

for layer in vgg16_2head.layers[:20]:                                                                                             # freeze blocks 1-5 (ie, train only fully connected part of the network)
    layer.trainable = False    

opt_used = optimizers.Nadam(clipnorm = 1., clipvalue = 0.5)                                                                       # adam with Nesterov accelerated gradient

loss_class = losses.categorical_crossentropy                                                                                                      # good loss to use for classification problems, may need to switch to binary if only two classes though?
loss_loc = losses.mean_squared_error                                                                                                              # loss for localisation

vgg16_2head.compile(optimizer = opt_used, loss=[loss_class, loss_loc], loss_weights = fc_loss_weights,
                    metrics = ['accuracy'])                                  

try:
    plot_model(vgg16_2head, to_file=project_outdir / 'step_06_train_fully_connected_model' / 'vgg16_2head.png', show_shapes = True, show_layer_names = True)      # try to make a graphviz style image showing the complete model 
except:
    print(f"Failed to create a .png of the model, but continuing anyway.  ")                               # this can easily fail, however, so simply alert the user and continue.  



batch_metric_names = ['loss', 'class_dense3_loss', 'loc_dense6_loss', 'class_dense3_accuracy', 'loc_dense6_accuracy']                             # all the metrics used in our model
batch_metrics_fc = {}                                                                                                                  # dist to store metrics and their values for every batch
for batch_metric_name in batch_metric_names:
    batch_metrics_fc[batch_metric_name] = []
    
batch_metrics_saver = save_batch_loss(batch_metric_names, batch_metrics_fc)                                                            # initiaite the callback to record metrics every batch
epoch_saver = save_model_each_epoch(project_outdir / "step_06_train_fully_connected_model" / f"vgg16_2head_fc")       # also initiate the callback to save the model every epoch
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str( project_outdir / "step_06_train_fully_connected_model" / 'logs' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
                                                      , histogram_freq=1)


# vgg16_2head_history = vgg16_2head.fit(train_generator, epochs = n_epochs_fc, validation_data = validation_generator,
#                                       shuffle = False, callbacks = [batch_metrics_saver, epoch_saver, tensorboard_callback] )      # shuffle false required for fast opening of batches from numpy files (it ensures idx increases, rather than bein random)
#                                       #max_queue_size = 4, workers = 4, use_multiprocessing = True)                                 # bit of a nightmare.     

# plot_all_metrics(batch_metrics_fc, vgg16_2head_history.history, batch_metric_names[:4],                                             # plot metrics for all batches and epochs.  Note, miss last metric as it's localisation accuracy which is meaningless.  
#                   'Fully connected network training', two_column = False, y_epoch_start = 0.8,
#                   out_path = project_outdir / "step_06_train_fully_connected_model" / "fully_connected_training.png")        


vgg16_2head_step_06 = build_2head_from_epochs(vgg16_2head,  models_dir = project_outdir / "step_06_train_fully_connected_model",
                                              n_epoch_class = n_epoch_class, n_epoch_loc = n_epoch_loc, 
                                              n_plot = 15, source_names = synthetic_ifgs_settings['defo_sources'],
                                              X_test = X_test, Y_class_test = Y_class_test, Y_loc_test = Y_loc_test)

#vgg16_2head_step_06.evaluate(x = X_test, y = [Y_class_test, Y_loc_test], verbose = 1)                  # check that model is being loaded correctly.  


#%% Step 07: Fine-tune the 5th convolutional block and the fully connected network.  

print(f"Fine-tuning the 5th block and fully connected network.  ")

vgg16_2head_step_07 = keras.models.clone_model(vgg16_2head_step_06)                                     # deep copy of the model.  Note this doesn't include weights.  
vgg16_2head_step_07.set_weights(vgg16_2head_step_06.get_weights())                                      # add weights to the new model

#layers_name = [layer.name for layer in vgg16_2head_step_07.layers]                                     # useful to check layer names.  

for layer in vgg16_2head_step_07.layers[15:]:                                                           # unfreeze block 5 (and also all the fully connected bits, but they're already trainable)
    layer.trainable = True    

#block5_optimiser = optimizers.RMSprop(lr=block5_lr)                                                    # not tested
# block5_optimiser = optimizers.SGD(lr=block5_lr, momentum=0.9)                                         # not tested
block5_optimiser = keras.optimizers.Adam(block5_lr)                                                     # seems to work?  

vgg16_2head_step_07.compile(optimizer = block5_optimiser, metrics=['accuracy'],                         # recompile as we've changed which layers can be trained/ optimizer etc.  
                            loss=[loss_class, loss_loc], loss_weights = block5_loss_weights)                                  

# print(f"Checking that the step 07 model retains the performance of step 06 before further training:")
# vgg16_2head_step_07.evaluate(x = X_test, y = [Y_class_test, Y_loc_test], verbose = 1)                       # check that model is being loaded correctly (i.e. that metrics are OK)

batch_metrics_b5 = {}                                                                                         # dict to store metrics and their values for every batch
for batch_metric_name in batch_metric_names:
    batch_metrics_b5[batch_metric_name] = []
    
batch_metrics_saver = save_batch_loss(batch_metric_names, batch_metrics_b5)                            # initiaite the callback to record metrics every batch
epoch_saver = save_model_each_epoch(step_07_dir / f"vgg16_2head_fc")                                   # also initiate the callback to save the model every epoch


vgg16_2head_history_step_07 = vgg16_2head_step_07.fit(train_generator, epochs = n_epochs_b5, validation_data = validation_generator,
                                                      shuffle = False, callbacks = [batch_metrics_saver, epoch_saver, tensorboard_callback])                 # train


plot_all_metrics(batch_metrics_b5, vgg16_2head_history_step_07.history, batch_metric_names[:4],                                                              # plot metrics for all batches and epochs
                  'Block 5 and fully connected network training', two_column = False,
                  out_path = step_07_dir / "block5_training.png")

Y_class_test_cnn_8, Y_loc_test_cnn_8 = vgg16_2head_step_07.predict(X_test, verbose = 1)                                                                     # predict class labels

plot_data_class_loc_caller(X_test[:n_plot,], classes = Y_class_test[:n_plot,], classes_predicted = Y_class_test_cnn_8[:n_plot,],                            # plot all the testing data
                                              locs = Y_loc_test[:n_plot,],      locs_predicted = Y_loc_test_cnn_8[:n_plot,], 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = ' Real and synthetic testing data (after step 08)')


#%% Step 08: Test with synthetic and real data

sys.exit()

print('\n\nStep 08: Forward pass of the testing real and synthetic data mix through the network:')

Y_class_test_cnn, Y_loc_test_cnn = vgg16_2head.predict(X_test, verbose = 1)                                    # predict class labels

plot_data_class_loc_caller(X_test[:n_plot,], classes = Y_class_test[:n_plot,], classes_predicted = Y_class_test_cnn[:n_plot,],                    # plot all the testing data
                                           locs = Y_loc_test[:n_plot,],      locs_predicted = Y_loc_test_cnn[:n_plot,], 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = ' Real and synthetic testing data (after step 07)')


#%% plot all test data

for ifg_n, ifg in enumerate(X_test):
    
    
    ax.imshow
    



def plot_1_ifg(X, Y_loc = None, Y_class = None): 
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import numpy as np
    import numpy.ma as ma
    
    f, axe = plt.subplots(1)
    
    #1: Draw the ifg (each ifg has its own colourscale)
    ifg_min = ma.min(X)                                                                   # min of ifg being plotted
    ifg_max = ma.max(X)                                                                   # max
    ifg_mid = 1 - ifg_max/(ifg_max + abs(ifg_min))                                                                     # mid point
    cmap_scaled = remappedColorMap(cmap_noscaling, start=0, midpoint=ifg_mid, stop=1.0, name='cmap_noscaling')         # rescale cmap to be the correct size
    axe.imshow(X, cmap = cmap_scaled)                                                     # plot ifg with camp
    
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
    if Y_loc is not None:
        start_stop_locs = centre_to_box(Y_loc)                                         # covert from centre width notation to start stop notation, # [x_start, x_stop, Y_start, Y_stop]

        add_square_plot(start_stop_locs[0], start_stop_locs[1], 
                        start_stop_locs[2], start_stop_locs[3], axe, colour='k')     # box around deformation
        
        axe.scatter(locs[plot_args[n_plot], 0], locs[plot_args[n_plot], 1], s = point_size, c = 'k')
        
    
    
    
    if classes is not None:
        label = np.argmax(classes[plot_args[n_plot], :])                               # labels from the cnn
        axe.set_title(f'Ifg: {plot_args[n_plot]}, Label: {source_names[label]}' , fontsize=label_fs)
            
    axe.set_ylim(top = 0, bottom = data.shape[1])
    axe.set_xlim(left = 0, right= data.shape[2])
    axe.set_yticks([])
    axe.set_xticks([])

        



#%% Step 09: Test with just real data

print('\n\nStep 08: Forward pass of the real (VOLCNET) testing data through the network:')

volcnet_test_data = {}
with open(volcnet_dir / "v1_manually_split" / "training_data.pkl", 'rb') as f:                     # open only the train data
    data_dict = pickle.load(f)                                                                     # 
    volcnet_test_data['X_m'] = data_dict['X_m']                                                                                # this is a masked array of ifgs, units are metres
    volcnet_test_data['Y_class'] = data_dict['Y_class']
    volcnet_test_data['Y_loc'] = data_dict['Y_loc']


volcnet_test_data['X_rescale'] = custom_range_for_CNN(volcnet_test_data['X_m'], cnn_settings['input_range'], mean_centre = False)                                                      # rescale to range for CNN

volcnet_test_data['Y_class_cnn'], volcnet_test_data['Y_loc_cnn'] = vgg16_2head.predict(volcnet_test_data['X_rescale'], verbose = 1)                                                    # predict class labels

test = vgg16_2head.evaluate(volcnet_test_data['X_rescale'], [volcnet_test_data['Y_class'], volcnet_test_data['Y_loc']])                                                                 # evaluate (ie. get metrics)

n_plot = volcnet_test_data['X_m'].shape[0]                                                                                                                                              # plot all the data

plot_data_class_loc_caller(volcnet_test_data['X_m'][:n_plot,], classes = volcnet_test_data['Y_class'][:n_plot,], classes_predicted = volcnet_test_data['Y_class_cnn'][:n_plot,],       # plot all/some of the testing data
                                           locs = volcnet_test_data['Y_loc'][:n_plot,],      locs_predicted = volcnet_test_data['Y_loc_cnn'][:n_plot,], 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = ' Real (VOLCNET) testing data (after step 07)')


#%% old step 06




# batch_metric_names = ['loss', 'class_dense3_loss', 'loc_dense6_loss', 'class_dense3_accuracy', 'loc_dense6_accuracy']                             # all the metrics used in our model
# batch_metrics_fc = {}                                                                                                                  # dist to store metrics and their values for every batch
# for batch_metric_name in batch_metric_names:
#     batch_metrics_fc[batch_metric_name] = []


# # vgg16_2head.compile(optimizer = opt_used, metrics=batch_metric_names,                                                         # 
# #                     loss=[loss_class, loss_loc], loss_weights = block5_loss_weights)                                  






    

    
# batch_metrics_saver = save_batch_loss(batch_metric_names, batch_metrics_fc)                                                            # initiaite the callback to record metrics every batch
# epoch_saver = save_model_each_epoch(project_outdir / "step_06_train_fully_connected_model" / f"vgg16_2head_fc")       # also initiate the callback to save the model every epoch


# train_generator = numpy_files_sequence(data_files_train, batch_size = batch_size)                                          # sequence using the training data
# validation_generator = numpy_files_sequence(data_files_validate, batch_size = batch_size)                                  # and validation data




# sys.exit()

# vgg16_2head_history = vgg16_2head.fit(train_generator, epochs = n_epochs_fc, validation_data = validation_generator,
#                                       callbacks = [batch_metrics_saver, epoch_saver])                                         # train

# plot_all_metrics(batch_metrics_fc, vgg16_2head_history.history, batch_metric_names[:4],                                             # plot metrics for all batches and epochs
#                   'Fully connected network training', two_column = True,
#                   out_path = project_outdir / "step_05_train_fully_connected" / "fully_connected_training.png")

# sys.exit()

# #%% Fine tune the model (including block 5)


# #block5_optimiser = optimizers.RMSprop(lr=block5_lr)                                      
# block5_optimiser = optimizers.SGD(lr=block5_lr, momentum=0.9)                                            # set the optimizer used in this training part.  Note have to set a learning rate manualy as an adaptive one (eg Nadam) would wreck model weights in the first few passes before it reduced.  

# #%%



# X_test, Y_class_test, Y_loc_test  = file_merger(data_files_test)                                 # Open the test data to RAM




# if do_step_06:
    
    
#     # 6.1 deal with files

    
#     # 6.2 define, compile, and train the model
    
#     fc_model_input = Input(shape = vgg16_block_1to5.output_shape[1:])                                                 # the input to the fully connected model must be the same shape as the output of the 5th block of vgg16
#     output_class, output_loc = define_two_head_model(fc_model_input, len(synthetic_ifgs_settings['defo_sources']))    # build the full connected part of the model, and get the two model outputs
#     vgg16_2head_fc = Model(inputs=fc_model_input, outputs=[output_class, output_loc])                                 # define the model.  Input is the shape of vgg16 block 1 to 5 output, and there are two outputs (hence list)                                
#     plot_model(vgg16_2head_fc, to_file=project_outdir / 'step_06_train_fully_connected_model' / 'vgg16_2head_fc.png',                     # also plot the model.  This funtcion is known to be fragile due to Graphviz dependencies.  
#                 show_shapes = True, show_layer_names = True)
    
#     opt_used = optimizers.Nadam(clipnorm = 1., clipvalue = 0.5)                                                       # adam with Nesterov accelerated gradient
#     vgg16_2head_fc.compile(optimizer = opt_used, loss=[loss_class, loss_loc],                                         # compile the model
#                             loss_weights = fc_loss_weights, metrics=['accuracy'])                                      # accuracy is useful to have on the terminal during training
    
   
#     batch_metrics_fc = {}                                                                                                                  # dist to store metrics and their values for every batch
#     for batch_metric_name in batch_metric_names:
#         batch_metrics_fc[batch_metric_name] = []
        
#     batch_metrics_saver = save_batch_loss(batch_metric_names, batch_metrics_fc)                                                            # initiaite the callback to record metrics every batch
#     epoch_saver = save_model_each_epoch(project_outdir / "step_06_train_fully_connected_model" / f"vgg16_2head_fc")       # also initiate the callback to save the model every epoch
    
    
#     train_generator = numpy_files_sequence(bottleneck_files_train, batch_size = batch_size)                                          # sequence using the training data
#     validation_generator = numpy_files_sequence(bottleneck_files_validate, batch_size = batch_size)                                  # and validation data
    
    
#     vgg16_2head_fc_history = vgg16_2head_fc.fit(train_generator, epochs = n_epochs_fc, validation_data = validation_generator,
#                                                 callbacks = [batch_metrics_saver, epoch_saver])                                         # train
    
        
#     plot_all_metrics(batch_metrics_fc, vgg16_2head_fc_history.history, batch_metric_names[:4],                                             # plot metrics for all batches and epochs
#                       'Fully Connected network training', two_column = True, 
#                       out_path = project_outdir / "step_06_train_fully_connected_model" / "fully_connected_training.png")
    
#     print('\n\: Forward pass of the testing (bottleneck) data through the network:')
#     X_test_btln, Y_class_test_btln, Y_loc_test_btln  = file_merger(bottleneck_files_test)                                 # Open the test data to RAM
#     Y_class_test_cnn_btln, Y_loc_test_cnn_btln = vgg16_2head_fc.predict(X_test_btln, verbose = 1)                                    # predict class labels
    
#     plot_data_class_loc_caller(X_test[:n_plot,], classes = Y_class_test_btln[:n_plot,], classes_predicted = Y_class_test_cnn_btln[:n_plot,],                    # plot all the testing data
#                                                locs = Y_loc_test_btln[:n_plot,],      locs_predicted = Y_loc_test_cnn_btln[:n_plot,], 
#                                source_names = synthetic_ifgs_settings['defo_sources'], window_title = 'Testing data (after step 07)')

