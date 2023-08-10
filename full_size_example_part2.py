#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:34:22 2021

@author: matthew
"""


# Imports
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
import sys
import pickle
import pdb
import datetime
from copy import deepcopy

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import Input, Model
from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model

from vudlnet21.neural_net import define_two_head_model, numpy_files_sequence, build_2head_from_epochs
from vudlnet21.plotting import plot_data_class_loc_caller
from vudlnet21.file_handling import file_merger




#%% Things to set



dependency_paths = {#'deep_learning_tools'   : '/home/matthew/university_work/crucial_1000gb/20_deep_learning_tools'}                                      # Available from Github: https://github.com/matthew-gaddes/Deep-Learning-Tools
                    'deep_learning_tools'     : '/home/matthew/university_work/crucial_1000gb/15_my_software_releases/Deep-Learning-Tools-1.2.0'}

# settings that need to be the same as part 1
project_outdir          = Path('./')
synthetic_ifgs_settings = {'defo_sources' : ['dyke', 'sill', 'no_def']}               # deformation patterns that will be included in the dataset.  
cnn_settings            = {'input_range'       : {'min':-1, 'max':1}}
batch_metric_names      = ['loss', 'class_dense3_loss', 'loc_dense6_loss', 'class_dense3_accuracy', 'loc_dense6_accuracy']                  # all the metrics used in our model

# shared training settings:
batch_size       = 100                                              # works on Nvidia GTX 1070 
block5_lr        = 1.e-5                                                                  # We have to set a learning rate manually as an adaptive approach (e.g. NADAM) will be high initially, and therefore make large updates that will wreck the model (as we're just fine-tuning a model so have something good to start with)
block5_optimiser = keras.optimizers.Adam(block5_lr)                                                     # seems to work?  
n_plot           = 45

# # Option 1: VGG16, synthetic data only.  
# synth_only                        = True
# cnn_settings['n_files_train']     = 35                                              # the number of files that will be used to train the network
# cnn_settings['n_files_validate']  = 3                                              # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
# cnn_settings['n_files_test']      = 1                                               # the number of files held back for testing.  
# model_type                        = "vgg16"                                           # convolution model              
# step_06_dir                       = project_outdir / "step_06a_synth_vgg16"
# fc_train_step_06                  = False
# fc_n_epochs                       = 100                                                                    # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
# fc_loss_weights                   = [1000, 1]                                                      # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# fc_n_epoch_class                  = 7
# fc_n_epoch_loc                    = 38                                                                        # if 15 epochs and start at 0, highest epoch number is 14 as python starts at 0.  
# step_07_dir                       = project_outdir / "step_07a_synth_vgg16"
# ft_train_step_07                  = False
# ft_n_epochs                       = 50                                                        # the number of epochs to fine-tune for (ie. the number of times all the training data are passed through the model)
# ft_loss_weights                   = [10, 1]                                                # classification loss (e.g. 0.126) then localisation loss (e.g. 300) weighting
# ft_n_epoch                        = 26                                                                        # if 15 epochs and start at 0, highest epoch number is 14 as python starts at 0.  

# # Option 2: VGG16, synthetic and real data
# synth_only                        = False
# cnn_settings['n_files_train']     = 75                                              # the number of files that will be used to train the network
# cnn_settings['n_files_validate']  = 5                                              # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
# cnn_settings['n_files_test']      = 0                                               # the number of files held back for testing.      
# model_type                        = "vgg16"                                           # convolution model
# step_06_dir                       = project_outdir / "step_06b_synth_real_vgg16"
# fc_train_step_06                  = False
# fc_n_epochs                       = 25                                                                    # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
# fc_loss_weights                   = [1000, 1]                                                      # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# fc_n_epoch_class                  = 6
# fc_n_epoch_loc                    = 12                                                                        # if 15 epochs and start at 0, highest epoch number is 14 as python starts at 0.  
# step_07_dir                       = project_outdir / "step_07b_synth_real_vgg16"
# ft_train_step_07                  = False
# ft_n_epochs                       = 25                                                        # the number of epochs to fine-tune for (ie. the number of times all the training data are passed through the model)
# ft_loss_weights                   = [10, 1]                                                # classification loss (e.g. 0.126) then localisation loss (e.g. 300) weighting
# ft_n_epoch                        = 13                                                          # best loss for localisation, good loss for combined.  

# Option 3: EfficientNet, synthetic and real data
synth_only                        = False
cnn_settings['n_files_train']     = 75                                              # the number of files that will be used to train the network
cnn_settings['n_files_validate']  = 5                                              # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
cnn_settings['n_files_test']      = 0                                               # the number of files held back for testing.      
model_type                        = "efficientnet"                                           # convolution model
step_06_dir                       = project_outdir / "step_06b_synth_real_efficientnet"
fc_train_step_06                  = True
fc_n_epochs                       = 25                                                                    # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
fc_loss_weights                   = [1000, 1]                                                      # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
fc_n_epoch_class                  = 6
fc_n_epoch_loc                    = 12                                                                        # if 15 epochs and start at 0, highest epoch number is 14 as python starts at 0.  
step_07_dir                       = project_outdir / "step_07b_synth_real_efficientnet"
ft_train_step_07                  = True
ft_n_epochs                       = 25                                                        # the number of epochs to fine-tune for (ie. the number of times all the training data are passed through the model)
ft_loss_weights                   = [10, 1]                                                # classification loss (e.g. 0.126) then localisation loss (e.g. 300) weighting
ft_n_epoch                        = 13                                                          # best loss for localisation, good loss for combined.  


#End options
step_08_dir = project_outdir/ "step_08_final_models"

#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
############################################################                              Shouldn't need to change below here                    ############################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


sys.path.append(dependency_paths['deep_learning_tools'])
import deep_learning_tools
from deep_learning_tools.file_handling import file_list_divider
from deep_learning_tools.plotting import plot_all_metrics
from deep_learning_tools.data_handling import custom_range_for_CNN
from deep_learning_tools.custom_callbacks import save_model_each_epoch, training_figure_per_epoch, save_all_metrics


#%% Prepare the data that is used for training both steps.  


if synth_only:
    print(f"Training the model with only the synthetic data.  ")
    data_files = sorted(glob.glob(str(project_outdir / 'step_05a_merged_rescaled_data_synth_only' / '*npz')), key = os.path.getmtime)                             # make list of data files

else:
    print(f"Training the model with a mixture of real and synthetic data.  ")
    data_files = sorted(glob.glob(str(project_outdir / 'step_05_merged_rescaled_data' / '*npz')), key = os.path.getmtime)                             # make list of data files
    
data_files_train, data_files_validate, _ = file_list_divider(data_files, cnn_settings['n_files_train'],
                                                             cnn_settings['n_files_validate'], cnn_settings['n_files_test'])                        # divide the files into train, validate.  Test come from elswhere (line below)
data_files_test = sorted(glob.glob(str(project_outdir / 'step_03_labelled_volcnet_data_testing' / '*pkl')), key = os.path.getmtime)                 # test data

if len(data_files) < (cnn_settings['n_files_train'] + cnn_settings['n_files_validate'] + cnn_settings['n_files_test']):
    raise Exception(f"There are {len(data_files)} data files, but {cnn_settings['n_files_train']} have been selected for training, "
                    f"{cnn_settings['n_files_validate']} for validation, and {cnn_settings['n_files_test']} for testing, "
                    f"which sums to greater than the number of data files.  Perhaps adjust the number of files used for the training stages? "
                    f"For now, exiting.")

print(f"    There are {len(data_files)} data files.  {len(data_files_train)} will be used for training, "         # print to terminal status on how many files will be used etc.  
      f"{len(data_files_validate)} for validation, and {len(data_files_test)} for testing.  ")

train_generator = numpy_files_sequence(data_files_train, batch_size = batch_size)                                          # sequence using the training data
validation_generator = numpy_files_sequence(data_files_validate, batch_size = batch_size)                                  # and validation data

X_test_m, Y_class_test, Y_loc_test = file_merger(data_files_test)                                                          # load to RAM, X_test_m are in metres, and shape nx224x224x1 and are masked arrays 
X_test = ma.repeat(deepcopy(X_test_m), 3, axis = 3)                                                                        # copy and make 3 channel.  
X_test = custom_range_for_CNN(X_test, cnn_settings['input_range'])                                                         # rescale to range for CNN (and convert from ma to np array.  )



#%% step 06: Train the fully connected part of the CNN
print('\nStep 06: Training the fully connected part of the CNN.')

if model_type == "vgg16":
    from tensorflow.keras.applications.vgg16 import VGG16
    encoder = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
elif model_type == 'efficientnet':
    from tensorflow.keras.applications.efficientnet import *
    encoder = EfficientNetB0(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
    #model = EfficientNetB7(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
elif model_type == "inception":
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    encoder = InceptionV3(weights='imagenet', include_top=False, input_shape = (224,224,3))
elif model_type == "resnet":
    from tensorflow.keras.applications.resnet_v2 import ResNet152V2
    encoder = ResNet152V2(weights='imagenet', include_top=False, input_shape = (224,224,3))
else:
    raise Exception("model_type is not recognised.  ")

for layer in encoder.layers:                                                                                             # freeze the convolutional part of the model (the encoder)
    layer.trainable = False    

output_class, output_loc = define_two_head_model(encoder.output, len(synthetic_ifgs_settings['defo_sources']))          # build the fully connected part of the model, and get the two model outputs
encoder_2head = Model(inputs=encoder.input, outputs=[output_class, output_loc])                                           # define the full model   
print(encoder_2head.summary())


opt_used = optimizers.Nadam(learning_rate = 0.001, clipnorm = 1., clipvalue = 0.5)                                                                       # adam with Nesterov accelerated gradient

loss_class = losses.categorical_crossentropy                                                                                                      # good loss to use for classification problems, may need to switch to binary if only two classes though?
loss_loc = losses.mean_squared_error                                                                                                              # loss for localisation

encoder_2head.compile(optimizer = opt_used, loss=[loss_class, loss_loc], loss_weights = fc_loss_weights,
                    metrics = ['accuracy'])                                  

try:
    plot_model(encoder_2head, to_file= step_06_dir / 'encoder_2head.png', show_shapes = True, show_layer_names = True)      # try to make a graphviz style image showing the complete model 
except:
    print(f"Failed to create a .png of the model, but continuing anyway.  ")                               # this can easily fail, however, so simply alert the user and continue.  


# 2: Callbacks
fc_metrics = save_all_metrics(batch_metric_names)                                                            # first part is a list of the metrics, second is a dict that will store the metrics for each epoch.  
epoch_saver = save_model_each_epoch(step_06_dir / f"encoder_2head_fc")       # also initiate the callback to save the model every epoch
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(step_06_dir/ 'logs' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
                                                      , histogram_freq=1)

fc_training_fig = training_figure_per_epoch(plot_all_metrics, fc_metrics.batch_metrics, fc_metrics.val_metrics, batch_metric_names[:4],                                             # plot metrics for all batches and epochs.  Note, miss last metric as it's localisation accuracy which is meaningless.  
                                            'Fully connected network training', two_column = False, y_epoch_start = 0.8,
                                            out_path = step_06_dir )


# 3: train (fit)
if fc_train_step_06:
    encoder_2head_history = encoder_2head.fit(train_generator, epochs = fc_n_epochs, validation_data = validation_generator,
                                          shuffle = False, callbacks = [fc_metrics, epoch_saver, tensorboard_callback, fc_training_fig] )      # shuffle false required for fast opening of batches from numpy files (it ensures idx increases, rather than bein random)
                                          #max_queue_size = 4, workers = 4, use_multiprocessing = True)                                 # bit of a nightmare.     
    with open(step_06_dir / 'training_history.pkl', 'wb') as f:                                                    # open the file
        pickle.dump(fc_metrics.batch_metrics, f)
        pickle.dump(fc_metrics.val_metrics, f)
                                          
    plt.switch_backend('Qt5agg')
    
    plot_all_metrics(fc_metrics.batch_metrics, fc_metrics.val_metrics, batch_metric_names[:4],                                             # plot metrics for all batches and epochs.  Note, miss last metric as it's localisation accuracy which is meaningless.  
                      'Fully connected training', two_column = False, y_epoch_start = 0.)
else:
    with open(step_06_dir / 'training_history.pkl', 'rb') as f:                                                    # open the file
        fc_metrics.batch_metrics = pickle.load(f)
        fc_metrics.val_metrics = pickle.load(f)
    plot_all_metrics(fc_metrics.batch_metrics, fc_metrics.val_metrics, batch_metric_names[:4],'Fully connected training (loaded)', two_column = False)



encoder_2head_step_06 = build_2head_from_epochs(encoder_2head,  models_dir = step_06_dir, 
                                              n_epoch_class = fc_n_epoch_class, n_epoch_loc = fc_n_epoch_loc)

#encoder_2head_step_06.evaluate(x = X_test, y = [Y_class_test, Y_loc_test], verbose = 1)                  # check that model is being loaded correctly.  

Y_class_test_cnn_6, Y_loc_test_cnn_6 = encoder_2head_step_06.predict(X_test, verbose = 1)                                                                     # predict class labels


plot_data_class_loc_caller(X_test_m[:n_plot,], classes = Y_class_test[:n_plot,], classes_predicted = Y_class_test_cnn_6[:n_plot,],                            # plot all the testing data
                                              locs = Y_loc_test[:n_plot,],      locs_predicted = Y_loc_test_cnn_6[:n_plot,], 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = 'Test data (after step 06)')


print("exiting as code not edited below this point")
sys.exit()

#%% Step 07: Fine-tune the 5th convolutional block and the fully connected network.  

print(f"Fine-tuning the 5th block and fully connected network.  ")

encoder_2head_step_07 = keras.models.clone_model(encoder_2head_step_06)                                     # deep copy of the model.  Note this doesn't include weights.  
encoder_2head_step_07.set_weights(encoder_2head_step_06.get_weights())                                      # add weights to the new model

#layers_name = [layer.name for layer in encoder_2head_step_07.layers]                                     # useful to check layer names.  

for layer in encoder_2head_step_07.layers[15:]:                                                           # unfreeze block 5 (and also all the fully connected bits, but they're already trainable)
    layer.trainable = True    

encoder_2head_step_07.compile(optimizer = block5_optimiser, metrics=['accuracy'],                         # recompile as we've changed which layers can be trained/ optimizer etc.  
                            loss=[loss_class, loss_loc], loss_weights = ft_loss_weights)                                  

# print(f"Checking that the step 07 model retains the performance of step 06 before further training:")
# encoder_2head_step_07.evaluate(x = X_test, y = [Y_class_test, Y_loc_test], verbose = 1)                       # check that model is being loaded correctly (i.e. that metrics are OK)

    
# Define the 4 callbacks
ft_metrics = save_all_metrics(batch_metric_names)                            # initiaite the callback to record metrics every batch
ft_epoch_saver = save_model_each_epoch(step_07_dir / f"encoder_2head_fc")                                   # also initiate the callback to save the model every epoch
ft_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str( step_07_dir / 'logs' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
                                                      , histogram_freq=1)

ft_training_fig = training_figure_per_epoch(plot_all_metrics, ft_metrics.batch_metrics, ft_metrics.val_metrics, batch_metric_names[:4],                                             # plot metrics for all batches and epochs.  Note, miss last metric as it's localisation accuracy which is meaningless.  
                                                   'Fine-tune training', two_column = False, y_epoch_start = 0.8,
                                                   out_path = step_07_dir)

if ft_train_step_07:
    encoder_2head_history_step_07 = encoder_2head_step_07.fit(train_generator, epochs = ft_n_epochs, validation_data = validation_generator,
                                                          shuffle = False, callbacks = [ft_metrics, ft_epoch_saver, ft_tensorboard_callback, ft_training_fig])                 # train
    
    with open(step_07_dir / 'training_history.pkl', 'wb') as f:                                                    # open the file
        pickle.dump(ft_metrics.batch_metrics, f)
        pickle.dump(ft_metrics.val_metrics, f)
            
    plt.switch_backend('Qt5agg')
    plot_all_metrics(ft_metrics.batch_metrics, ft_metrics.val_metrics, batch_metric_names[:4],'Fine tune training', two_column = False)
    
else:
    with open(step_07_dir / 'training_history.pkl', 'rb') as f:                                                    # open the file
        ft_metrics.batch_metrics = pickle.load(f)
        ft_metrics.val_metrics = pickle.load(f)
    plot_all_metrics(ft_metrics.batch_metrics, ft_metrics.val_metrics, batch_metric_names[:4],'Fine tune training (loaded)', two_column = False)
    


encoder_2head_step_07 = build_2head_from_epochs(encoder_2head_step_07,  models_dir = step_07_dir,
                                              n_epoch_class = ft_n_epoch, n_epoch_loc = ft_n_epoch)                                                         # note that class and localisation epoch must be the same as the 5th block has been modified.  


Y_class_test_cnn_7, Y_loc_test_cnn_7 = encoder_2head_step_07.predict(X_test, verbose = 1)                                                                     # predict class labels


plot_data_class_loc_caller(X_test_m[:n_plot,], classes = Y_class_test[:n_plot,], classes_predicted = Y_class_test_cnn_7[:n_plot,],                            # plot all the testing data
                                              locs = Y_loc_test[:n_plot,],      locs_predicted = Y_loc_test_cnn_7[:n_plot,], 
                           source_names = synthetic_ifgs_settings['defo_sources'], window_title = 'Test data (after step 07)')


#%% Step 08: save final models and evaluate

print('\n\nStep 08: Save the best model and forward pass of the real (VOLCNET) testing data through the network:')
    

Y_class_test_cnn, Y_loc_test_cnn = encoder_2head_step_07.predict(X_test, verbose = 1)                                                    # predict class labels


#n_plot = X_test.shape[0]                                                                                                                                              # plot all the data
#n_plot = 300

plot_data_class_loc_caller(X_test_m[:n_plot,], classes = Y_class_test[:n_plot,], classes_predicted = Y_class_test_cnn[:n_plot,],       # plot all/some of the testing data
                                            locs = Y_loc_test[:n_plot,],      locs_predicted = Y_loc_test_cnn[:n_plot,], 
                            source_names = synthetic_ifgs_settings['defo_sources'], window_title = 'Test data (after step 08)')

if synth_only:
    encoder_2head_step_07.save(step_08_dir / "model_synthetic_data_only")                  # 
    predictions_filename = 'synth_only_test_data_predictions.pkl'
    evaluate_filename = 'synth_only_test_data_evaluations.pkl'
else:
    encoder_2head_step_07.save(step_08_dir / "model_synthetic_and_real_data")                  # 
    predictions_filename = 'synth_real_test_data_predictions.pkl'
    evaluate_filename = 'synth_real_test_data_evaluations.pkl'

with open(step_08_dir / predictions_filename, 'wb') as f:                                                    # open the file
    pickle.dump(X_test, f)
    pickle.dump(X_test_m, f)
    pickle.dump(Y_class_test, f)
    pickle.dump(Y_class_test_cnn, f)
    pickle.dump(Y_loc_test, f)
    pickle.dump(Y_loc_test_cnn, f)


# also evaluate all the data, and by each label
evaluate_results = {}
evaluate_results['all'] = encoder_2head_step_06.evaluate(X_test, y = [Y_class_test, Y_loc_test], verbose = 1)                                                                 # evaluate (ie. get metrics)

for source_n, source in enumerate(synthetic_ifgs_settings['defo_sources']):

    print(f"Evaluating for source {source}:")
    args = np.ravel(np.argwhere(Y_class_test[:,source_n] == 1))                                                                                             # get only the data with that label
    evaluate_results[source] = encoder_2head_step_06.evaluate(X_test[args,], y = [Y_class_test[args,], Y_loc_test[args,]], verbose = 1)                      # 
    
with open(step_08_dir / evaluate_filename, 'wb') as f:                                                    # 
    pickle.dump(evaluate_results, f)    