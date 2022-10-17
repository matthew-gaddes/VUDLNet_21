#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:34:22 2021

@author: matthew
"""


# Imports
import numpy as np
from pathlib import Path
import glob
import os
import sys
import pickle
import pdb

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input, Model
from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model

from vudlnet21.neural_net import define_two_head_model, numpy_files_sequence, save_model_each_epoch, save_batch_loss, numpy_files_sequence_old
#from vudlnet21.neural_net import custom_range_for_CNN
from vudlnet21.plotting import plot_all_metrics, plot_data_class_loc_caller




#%% Things to set

dependency_paths = {'deep_learning_tools'   : '/home/matthew/university_work/20_deep_learning_tools'}                                      # Available from Github: https://github.com/matthew-gaddes/Deep-Learning-Tools

# settings that need to be the same as part 1
project_outdir = Path('./')
synthetic_ifgs_settings = {'defo_sources' : ['dyke', 'sill', 'no_def']}               # deformation patterns that will be included in the dataset.  
cnn_settings = {'input_range'       : {'min':-1, 'max':1}}

   
# step 06 (train the fully connected part of the network)
do_step_06                        = True
batch_size_fc                     = 25                                              # as data files have 500 in them, this divides in cleanly (i.e. no half batches).  Not tested in case that it doesn't divide in cleanly, perhaps last batch is just smaller and faster?  
fc_loss_weights                   = [1., 1.]                                                      # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
n_epochs_fc                       = 10                                                                    # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
cnn_settings = {}
cnn_settings['n_files_train']     = 70                                              # the number of files that will be used to train the network
cnn_settings['n_files_validate']  = 5                                              # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
cnn_settings['n_files_test']      = 5                                               # the number of files held back for testing.  


# step 07 (fine-tune the 5th block and the fully connected part of the network):
batch_size_b5           = 50
n_epochs_b5             = 10                                                                 # the number of epochs to fine-tune for (ie. the number of times all the training data are passed through the model)
block5_loss_weights = [0.05, 0.95]                                                  # as per fc_loss_weights, but by changing these more emphasis can be placed on either the clasification or localisation loss.  
block5_lr = 1.5e-9                                                                  # a pretty important parameter.  We have to set a learning rate manually as an adaptive approach (e.g. NADAM) will be high initially, and therefore make large updates that will wreck the model (as we're just fine-tuning a model so have something good to start with)
#block5_lr = 1.0e-6                                                                  # a pretty important parameter.  We have to set a learning rate manually as an adaptive approach (e.g. NADAM) will be high initially, and therefore make large updates that will wreck the model (as we're just fine-tuning a model so have something good to start with)


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

#pdb.set_trace()

sys.path.append(dependency_paths['deep_learning_tools'])
import deep_learning_tools
from deep_learning_tools.file_handling import file_list_divider

#%% 5: Compute bottlenceck features (i.e. pass the data through the 5 computationally expensive convolutional blocks of VGG16 to create the feature vectors ( normally 224x224x3 -> 7x7x512))

# if do_step_05:
#     print("\nStep 05: Computing the bottleneck features.")
    
#     vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                          # load the first 5 (convolutional) blocks of VGG16 and their weights.  
#     data_out_files = sorted(glob.glob(str(project_outdir / 'step_04_merged_rescaled_data/*.npz')))                      # get a list of the files output by step 05 (augmented real data and synthetic data mixed and rescaed to correct range, with 0s for masked areas.  )
    
#     n_data_per_file = np.load(data_out_files[0])['X'].shape[0]                                                          # get the number of data in the first file, assume that the same for all files.  
    
#     for file_n, data_out_file in enumerate(data_out_files):                                             # loop through each of the step 05 files.  
#         print(f'Bottleneck file {file_n}:')    
#         data_out_file = Path(data_out_file)                                                                                     # convert to path 
#         bottleneck_file_name = data_out_file.parts[-1].split('.')[0]                                                            # and get last part which is filename    
#         data = np.load(data_out_file)                                                                                           # load the numpy file
#         X = data['X']                                                                                                           # extract the data for it
#         Y_class = data['Y_class']                                                                                               # and class labels.  
#         Y_loc = data['Y_loc']                                                                                                   # and location labels.  
#         X_btln = vgg16_block_1to5.predict(X, verbose = 1)                                                                       # predict up to bottleneck    
#         np.savez(project_outdir / 'step_05_bottleneck' / f"{bottleneck_file_name}_bottleneck.npz", X = X_btln, Y_class = Y_class, Y_loc = Y_loc)     # save the bottleneck file, and the two types of label.  



#%% step 05: Train the fully connected part of the CNN

print('\nStep 05: Training the fully connected part of the CNN.')



data_files = sorted(glob.glob(str(project_outdir / 'step_05_merged_rescaled_data' / '*npz')), key = os.path.getmtime)                             # make list of data files
data_files_train, data_files_validate, data_files_test = file_list_divider(data_files, cnn_settings['n_files_train'], 
                                                                           cnn_settings['n_files_validate'], cnn_settings['n_files_test'])        # divide the files into train, validate and test    


if len(data_files) < (cnn_settings['n_files_train'] + cnn_settings['n_files_validate'] + cnn_settings['n_files_test']):
    raise Exception(f"There are {len(data_files)} data files, but {cnn_settings['n_files_train']} have been selected for training, "
                    f"{cnn_settings['n_files_validate']} for validation, and {cnn_settings['n_files_test']} for testing, "
                    f"which sums to greater than the number of data files.  Perhaps adjust the number of files used for the training stages? "
                    f"For now, exiting.")

print(f"    There are {len(data_files)} data files.  {len(data_files_train)} will be used for training,"         # print to terminal status on how many files will be used etc.  
      f"{len(data_files_validate)} for validation, and {len(data_files_test)} for testing.  ")




vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
output_class, output_loc = define_two_head_model(vgg16_block_1to5.output, len(synthetic_ifgs_settings['defo_sources']))          # build the fully connected part of the model, and get the two model outputs
vgg16_2head = Model(inputs=vgg16_block_1to5.input, outputs=[output_class, output_loc])                                           # define the full model

for layer in vgg16_2head.layers[:20]:                                                                                             # freeze blocks 1-5 (ie, train only fully connected part of the network)
    layer.trainable = False    

opt_used = optimizers.Nadam(clipnorm = 1., clipvalue = 0.5)                                                                       # adam with Nesterov accelerated gradient

loss_class = losses.categorical_crossentropy                                                                                                      # good loss to use for classification problems, may need to switch to binary if only two classes though?
loss_loc = losses.mean_squared_error                                                                                                              # loss for localisation

vgg16_2head.compile(optimizer = opt_used, loss=[loss_class, loss_loc], loss_weights = fc_loss_weights)                                  

train_generator_fc = numpy_files_sequence(data_files_train, batch_size = batch_size_b5)                                          # sequence using the training data
validation_generator_fc = numpy_files_sequence(data_files_validate, batch_size = batch_size_b5)                                  # and validation data



vgg16_2head_history = vgg16_2head.fit(train_generator_fc, epochs = n_epochs_fc, validation_data = validation_generator_fc,
                                      shuffle = False)
                                      

#%%

sys.exit()

batch_metric_names = ['loss', 'class_dense3_loss', 'loc_dense6_loss', 'class_dense3_accuracy', 'loc_dense6_accuracy']                             # all the metrics used in our model
batch_metrics_fc = {}                                                                                                                  # dist to store metrics and their values for every batch
for batch_metric_name in batch_metric_names:
    batch_metrics_fc[batch_metric_name] = []


# vgg16_2head.compile(optimizer = opt_used, metrics=batch_metric_names,                                                         # 
#                     loss=[loss_class, loss_loc], loss_weights = block5_loss_weights)                                  


try:
    plot_model(vgg16_2head, to_file=project_outdir / 'step_06_train_fully_connected_model' / 'vgg16_2head.png', show_shapes = True, show_layer_names = True)      # try to make a graphviz style image showing the complete model 
except:
    print(f"Failed to create a .png of the model, but continuing anyway.  ")                               # this can easily fail, however, so simply alert the user and continue.  



    

    
batch_metrics_saver = save_batch_loss(batch_metric_names, batch_metrics_fc)                                                            # initiaite the callback to record metrics every batch
epoch_saver = save_model_each_epoch(project_outdir / "step_06_train_fully_connected_model" / f"vgg16_2head_fc")       # also initiate the callback to save the model every epoch


train_generator_fc = numpy_files_sequence(data_files_train, batch_size = batch_size_b5)                                          # sequence using the training data
validation_generator_fc = numpy_files_sequence(data_files_validate, batch_size = batch_size_b5)                                  # and validation data




sys.exit()

vgg16_2head_history = vgg16_2head.fit(train_generator_fc, epochs = n_epochs_fc, validation_data = validation_generator_fc,
                                      callbacks = [batch_metrics_saver, epoch_saver])                                         # train

plot_all_metrics(batch_metrics_fc, vgg16_2head_history.history, batch_metric_names[:4],                                             # plot metrics for all batches and epochs
                  'Fully connected network training', two_column = True,
                  out_path = project_outdir / "step_05_train_fully_connected" / "fully_connected_training.png")

sys.exit()

#%% Fine tune the model (including block 5)


#block5_optimiser = optimizers.RMSprop(lr=block5_lr)                                      
block5_optimiser = optimizers.SGD(lr=block5_lr, momentum=0.9)                                            # set the optimizer used in this training part.  Note have to set a learning rate manualy as an adaptive one (eg Nadam) would wreck model weights in the first few passes before it reduced.  

#%%



X_test, Y_class_test, Y_loc_test  = file_merger(data_files_test)                                 # Open the test data to RAM




if do_step_06:
    
    
    # 6.1 deal with files

    
    # 6.2 define, compile, and train the model
    
    fc_model_input = Input(shape = vgg16_block_1to5.output_shape[1:])                                                 # the input to the fully connected model must be the same shape as the output of the 5th block of vgg16
    output_class, output_loc = define_two_head_model(fc_model_input, len(synthetic_ifgs_settings['defo_sources']))    # build the full connected part of the model, and get the two model outputs
    vgg16_2head_fc = Model(inputs=fc_model_input, outputs=[output_class, output_loc])                                 # define the model.  Input is the shape of vgg16 block 1 to 5 output, and there are two outputs (hence list)                                
    plot_model(vgg16_2head_fc, to_file=project_outdir / 'step_06_train_fully_connected_model' / 'vgg16_2head_fc.png',                     # also plot the model.  This funtcion is known to be fragile due to Graphviz dependencies.  
                show_shapes = True, show_layer_names = True)
    
    opt_used = optimizers.Nadam(clipnorm = 1., clipvalue = 0.5)                                                       # adam with Nesterov accelerated gradient
    vgg16_2head_fc.compile(optimizer = opt_used, loss=[loss_class, loss_loc],                                         # compile the model
                            loss_weights = fc_loss_weights, metrics=['accuracy'])                                      # accuracy is useful to have on the terminal during training
    
   
    batch_metrics_fc = {}                                                                                                                  # dist to store metrics and their values for every batch
    for batch_metric_name in batch_metric_names:
        batch_metrics_fc[batch_metric_name] = []
        
    batch_metrics_saver = save_batch_loss(batch_metric_names, batch_metrics_fc)                                                            # initiaite the callback to record metrics every batch
    epoch_saver = save_model_each_epoch(project_outdir / "step_06_train_fully_connected_model" / f"vgg16_2head_fc")       # also initiate the callback to save the model every epoch
    
    
    train_generator = numpy_files_sequence(bottleneck_files_train, batch_size = batch_size_fc)                                          # sequence using the training data
    validation_generator = numpy_files_sequence(bottleneck_files_validate, batch_size = batch_size_fc)                                  # and validation data
    
    
    vgg16_2head_fc_history = vgg16_2head_fc.fit(train_generator, epochs = n_epochs_fc, validation_data = validation_generator,
                                                callbacks = [batch_metrics_saver, epoch_saver])                                         # train
    
        
    plot_all_metrics(batch_metrics_fc, vgg16_2head_fc_history.history, batch_metric_names[:4],                                             # plot metrics for all batches and epochs
                      'Fully Connected network training', two_column = True, 
                      out_path = project_outdir / "step_06_train_fully_connected_model" / "fully_connected_training.png")
    
    print('\n\: Forward pass of the testing (bottleneck) data through the network:')
    X_test_btln, Y_class_test_btln, Y_loc_test_btln  = file_merger(bottleneck_files_test)                                 # Open the test data to RAM
    Y_class_test_cnn_btln, Y_loc_test_cnn_btln = vgg16_2head_fc.predict(X_test_btln, verbose = 1)                                    # predict class labels
    
    plot_data_class_loc_caller(X_test[:n_plot,], classes = Y_class_test_btln[:n_plot,], classes_predicted = Y_class_test_cnn_btln[:n_plot,],                    # plot all the testing data
                                               locs = Y_loc_test_btln[:n_plot,],      locs_predicted = Y_loc_test_cnn_btln[:n_plot,], 
                               source_names = synthetic_ifgs_settings['defo_sources'], window_title = 'Testing data (after step 07)')

    

#%% Step 07: Fine-tune the 5th convolutional block and the fully connected network.  

print(f"Starting to train the 5th block and fully connected network in parallel.  This will be significantly slower than the previous step "
      f"as bottleneck files can no longer be used.  ")





vgg16_block_1to5 = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define out own )
output_class, output_loc = define_two_head_model(vgg16_block_1to5.output, len(synthetic_ifgs_settings['defo_sources']))          # build the fully connected part of the model, and get the two model outputs

vgg16_2head = Model(inputs=vgg16_block_1to5.input, outputs=[output_class, output_loc])                                           # define the full model
vgg16_2head.load_weights(project_outdir / 'step_06_train_fully_connected_model' / f"vgg16_2head_fc_epoch_{(n_epochs_fc-1):03d}.h5", by_name = True)           # load the weights for the fully connected part which were trained in step 06 (by_name flag so that it doesn't matter that the models are different sizes))


for layer in vgg16_2head.layers[:15]:                                                                                             # freeze blocks 1-4 (ie, we are only fine tuneing the 5th block and the fully connected part of the network)
    layer.trainable = False    

#block5_optimiser = optimizers.RMSprop(lr=block5_lr)                                      
block5_optimiser = optimizers.SGD(lr=block5_lr, momentum=0.9)                                            # set the optimizer used in this training part.  Note have to set a learning rate manualy as an adaptive one (eg Nadam) would wreck model weights in the first few passes before it reduced.  
vgg16_2head.compile(optimizer = block5_optimiser, metrics=['accuracy'],                                  # recompile as we've changed which layers can be trained/ optimizer etc.  
                    loss=[loss_class, loss_loc], loss_weights = block5_loss_weights)                                  

try:
    plot_model(vgg16_2head, to_file=project_outdir / 'step_06_train_fully_connected_model' / 'vgg16_2head.png', show_shapes = True, show_layer_names = True)      # try to make a graphviz style image showing the complete model 
except:
    print(f"Failed to create a .png of the model, but continuing anyway.  ")                               # this can easily fail, however, so simply alert the user and continue.  



batch_metrics_b5 = {}                                                                                                                  # dist to store metrics and their values for every batch
for batch_metric_name in batch_metric_names:
    batch_metrics_b5[batch_metric_name] = []
    
batch_metrics_saver = save_batch_loss(batch_metric_names, batch_metrics_b5)                                                            # initiaite the callback to record metrics every batch
epoch_saver = save_model_each_epoch(project_outdir / "step_07_train_full_model" / f"vgg16_2head_fc")       # also initiate the callback to save the model every epoch


train_generator_b5 = numpy_files_sequence(data_files_train, batch_size = batch_size_b5)                                          # sequence using the training data
validation_generator_b5 = numpy_files_sequence(data_files_validate, batch_size = batch_size_b5)                                  # and validation data


vgg16_2head_history = vgg16_2head.fit(train_generator_b5, epochs = n_epochs_b5, validation_data = validation_generator_b5,
                                            callbacks = [batch_metrics_saver, epoch_saver])                                         # train

plot_all_metrics(batch_metrics_b5, vgg16_2head_history.history, batch_metric_names[:4],                                             # plot metrics for all batches and epochs
                  'Block 5 and fully connected network training', two_column = True,
                  out_path = project_outdir / "step_07_train_full_model" / "block5_training.png")



#%% Step 08: Test with synthetic and real data

print('\n\nStep 08: Forward pass of the testing real and synthetic data mix through the network:')

Y_class_test_cnn, Y_loc_test_cnn = vgg16_2head.predict(X_test, verbose = 1)                                    # predict class labels

plot_data_class_loc_caller(X_test[:n_plot,], classes = Y_class_test[:n_plot,], classes_predicted = Y_class_test_cnn[:n_plot,],                    # plot all the testing data
                                           locs = Y_loc_test[:n_plot,],      locs_predicted = Y_loc_test_cnn[:n_plot,], 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = ' Real and synthetic testing data (after step 07)')


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


