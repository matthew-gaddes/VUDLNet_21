#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:34:22 2021

@author: matthew
"""

print(f"Succesffuly started the script. \n\n\n")

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

from tensorflow.keras import layers
from tensorflow.keras import Input, Model
from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from vudlnet21.neural_net import define_two_head_model, numpy_files_sequence, build_2head_from_epochs


from vudlnet21.file_handling import file_merger
from vudlnet21.evaluation import evaluate_model



def print_model_param_info(model):
    """ """
    import tensorflow.keras.backend as K
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    


#%% Things to set
print(f"\n\n Using the development version of deep learning tools ")
dependency_paths = {'deep_learning_tools'   : '/home/matthew/university_work/crucial_1000gb/20_deep_learning_tools'}                                      # Available from Github: https://github.com/matthew-gaddes/Deep-Learning-Tools
                    #'deep_learning_tools'     : '/home/matthew/university_work/crucial_1000gb/15_my_software_releases/Deep-Learning-Tools-1.2.0'}

# settings that need to be the same as part 1
project_outdir                  = Path('./')
synthetic_ifgs_settings         = {'defo_sources' : ['dyke', 'sill', 'no_def']}                                                             # deformation patterns that will be included in the dataset.  
batch_metric_names              = ['loss', 'class_dense3_loss', 'loc_dense6_loss', 'class_dense3_accuracy', 'loc_dense6_accuracy']          # all the metrics used in our model

# Settings shared with all options.  
step_06_or_07                   = 'step_07'
fc_train_step_06                = False
ft_train_step_07                = False                                                                                 # either step 6 or 7 can be run, so this is the opposite of previous status
n_plot                          = 45
cnn_settings                    = {}


# # Option 1A: VGG16, synthetic data only.  
# synth_onl0y                        = True
# model_type                        = "vgg16"                                         # convolution model
# cnn_settings['input_range']       =  {'min':-1, 'max':1}                           # note that Efficientnet is unusual as not - 1 to 1 
# data_dir                          = project_outdir / "step_05a_merged_rescaled_data_synth_only_range_-1_1" 
# cnn_settings['n_files_train']     = 35                                              # the number of files that will be used to train the network
# cnn_settings['n_files_validate']  = 3                                               # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
# cnn_settings['n_files_test']      = 1                                               # the number of files held back for testing.  
# step_06_dir                       = project_outdir / "step_06a_synth_vgg16"
# fc_n_epochs                       = 100                                             # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
# fc_batch_size                     = 100                                             # works on Nvidia GTX 1070 
# fc_loss_weights                   = [1000, 1]                                       # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  4
# fc_optimizer                      = optimizers.Nadam(learning_rate = 1e-4) 
# step_07_dir                       = project_outdir / "step_07a_synth_vgg16"
# ft_n_epochs                       = 200                                              # the number of epochs to fine-tune for (ie. the number of times all the training data are passed through the model)
# ft_batch_size                     = 100                                             # works on Nvidia GTX 1070 
# ft_loss_weights                   = [10, 1]                                         # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  4
# ft_unfreeze_from                  = 15                                              # layers after this are trained, before are not.  
# ft_optimizer                      = optimizers.SGD(learning_rate = 1e-6)            # SGD, 

# # Option 2: synthetic and real data
synth_only                        = False
cnn_settings['n_files_train']     = 75                                              # the number of files that will be used to train the network
cnn_settings['n_files_validate']  = 5                                               # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
cnn_settings['n_files_test']      = 0                                               # the number of files held back for testing.      

# # Option 2B: VGG16, synthetic and real data
# model_type                        = "vgg16"                                         # convolution model
# cnn_settings['input_range']       =  {'min':-1, 'max':1}                           # note that Efficientnet is unusual as not - 1 to 1 
# data_dir                          = project_outdir / "step_05_merged_rescaled_data_range_-1_1" 
# step_06_dir                       = project_outdir / "step_06b_synth_real_vgg16"
# fc_n_epochs                       = 100                                              # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
# fc_batch_size                     = 100                                                                # works on Nvidia GTX 1070 
# fc_loss_weights                   = [1000, 1]                                       # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# # fc_optimizer                      = optimizers.Nadam(learning_rate = 0.001, 
# #                                                       clipnorm = 1., clipvalue = 0.5)                    # adam with Nesterov accelerated gradient
# fc_optimizer                      = optimizers.Adam(learning_rate = 0.001)        # 
# step_07_dir                       = project_outdir / "step_07b_synth_real_vgg16"
# ft_n_epochs                       = 100                                              # the number of epochs to fine-tune for 
# ft_batch_size                     = 100                                                                # works on Nvidia GTX 1070 
# ft_loss_weights                   = [10, 1]                                         # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# ft_unfreeze_from                  = 15                                            # layers after this are trained, before are not.  
# ft_optimizer                      = optimizers.SGD(learning_rate = 1e-7)            # SGD, 1e-5 too large, previously set to 1e-6.  [1e-10 preserves model but doesn't learn significnatly] [ 1e-9 and 1e-8 seesm to lead to gradual degradation of performace]

# """"
# ft_loss_weights = [1000, 1] doesn't work even with SBG at 1e-10
# trying 100, 1, didn't make much difference'
# trying [10, 1], at 1e-9 doesn't learn (but only tried ~4 epochs)
# """


# Option 2C: EfficientNet, synthetic and real data
model_type                        = "efficientnet"                                  # convolution model
cnn_settings['input_range']       =  {'min':0, 'max':255}                           # note that Efficientnet is unusual as not - 1 to 1 
data_dir                          = project_outdir / "step_05_merged_rescaled_data_range_0_255" 
step_06_dir                       = project_outdir / "step_06c_synth_real_efficientnet"
fc_n_epochs                       = 150                                              # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
fc_batch_size                     = 100                                                                # works on Nvidia GTX 1070 
fc_loss_weights                   = [1000, 1]                                       # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
fc_optimizer                      = optimizers.Adam(learning_rate = 1e-6)           # 

step_07_dir                       = project_outdir / "step_07c_synth_real_efficientnet"
ft_n_epochs                       = 50                                              # the number of epochs to fine-tune for 
ft_batch_size                     = 25                                                                # works on Nvidia GTX 1070 
ft_loss_weights                   = [1000, 1]                                         # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
ft_unfreeze_from                  = 232                                            # layer 232 is block_7a_project_now, layer 234 is top_conv, 237 is end of EfficientNet and start of two head classifier
ft_optimizer                      = optimizers.SGD(learning_rate = 2e-7)            # SGD, 1e-5 too large, 1e-6 also eventually destroys performance


# Option 2D: InceptionV3, synthetic and real data
# model_type                        = "inception"                                     # convolution model
# cnn_settings['input_range']       =  {'min':-1, 'max':1}                           # note that Efficientnet is unusual as not - 1 to 1 
# data_dir                          = project_outdir / "step_05_merged_rescaled_data_range_-1_1" 
# step_06_dir                       = project_outdir / "step_06d_synth_real_inceptionv3"
# fc_n_epochs                       = 300                                                   # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
# fc_batch_size                     = 100                                                                # works on Nvidia GTX 1070 
# fc_loss_weights                   = [1000, 1]                                       # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# fc_optimizer                      = keras.optimizers.Adam(1.e-6)                                     # 
# step_07_dir                       = project_outdir / "step_07d_synth_real_inceptionv3"
# ft_n_epochs                       = 200                                              # the number of epochs to fine-tune for 
# ft_batch_size                     = 100                                                                # works on Nvidia GTX 1070 
# ft_loss_weights                   = [1000, 1]                                         # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# ft_unfreeze_from                  = 299                                              # Final convolutional (conv2d_93) and following activations.  
# ft_optimizer                      = keras.optimizers.Adam(1.e-5)                     # 

# Option 2E: ResNet152V2, synthetic and real data
# model_type                        = "resnet"                                       # convolution model
# cnn_settings['input_range']       =  {'min':-1, 'max':1}                           # note that Efficientnet is unusual as not - 1 to 1 
# data_dir                          = project_outdir / "step_05_merged_rescaled_data_range_-1_1" 
# step_06_dir                       = project_outdir / "step_06e_synth_real_resnet"
# fc_n_epochs                       = 100                                              # the number of epochs to train the fully connected network for (ie. the number of times all the training data are passed through the model)
# fc_batch_size                     = 100                                                                # works on Nvidia GTX 1070 
# fc_loss_weights                   = [1000, 1]                                       # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# fc_optimizer                      = keras.optimizers.Adam(1.e-6)                                     # 
# step_07_dir                       = project_outdir / "step_07e_synth_real_resnet"
# ft_n_epochs                       = 25                                              # the number of epochs to fine-tune for 
# ft_batch_size                     = 100                                             # works on Nvidia GTX 1070 
# ft_loss_weights                   = [10, 1]                                         # the relative weighting of the two losses (classificaiton and localisation) to contribute to the global loss.  Classification first, localisation second.  
# ft_unfreeze_from                  = 528                                             # All of the 5th convolutional block and onwards (conv5_block1_preact_bn including and onwards)
# ft_optimizer                      = keras.optimizers.Adam(1.e-5)                    # 

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
from deep_learning_tools.custom_callbacks import save_model_each_epoch, training_figure_per_epoch
from deep_learning_tools.custom_callbacks import save_all_metrics, duplicate_model_remover


#%% Prepare the data that is used for training both steps.  

data_files = sorted(glob.glob(str(data_dir / '*npz')), key = os.path.getmtime)                             # make list of data files
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

fc_train_generator = numpy_files_sequence(data_files_train, batch_size = fc_batch_size)                                          # sequence using the training data
fc_validation_generator = numpy_files_sequence(data_files_validate, batch_size = fc_batch_size)                                  # and validation data.  Note that batch size is set for fully connected training.  

ft_train_generator = numpy_files_sequence(data_files_train, batch_size = ft_batch_size)                                          # sequence using the training data
ft_validation_generator = numpy_files_sequence(data_files_validate, batch_size = ft_batch_size)                                  # and validation data.  Note that batch size is set for fine tune training.  

X_test_m, Y_class_test, Y_loc_test = file_merger(data_files_test)                                                          # load to RAM, X_test_m are in metres, and shape nx224x224x1 and are masked arrays 
X_test = ma.repeat(deepcopy(X_test_m), 3, axis = 3)                                                                        # copy and make 3 channel.  
X_test = custom_range_for_CNN(X_test, cnn_settings['input_range'])                                                         # rescale to range for CNN (and convert from ma to np array.  )


#%% Prepare the model without weights.  


if model_type == "vgg16":
    from tensorflow.keras.applications.vgg16 import VGG16
    encoder = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
elif model_type == 'efficientnet':
    from tensorflow.keras.applications.efficientnet import *
    encoder = EfficientNetB0(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
    #encoder = EfficientNetB7(weights='imagenet', include_top=False, input_shape = (224,224,3))                                       # VGG16 is used for its convolutional layers and weights (but no fully connected part as we define our own )
elif model_type == "inception":
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    encoder = InceptionV3(weights='imagenet', include_top=False, input_shape = (224,224,3))
elif model_type == "resnet":
    from tensorflow.keras.applications.resnet_v2 import ResNet152V2
    encoder = ResNet152V2(weights='imagenet', include_top=False, input_shape = (224,224,3))
else:
    raise Exception("model_type is not recognised.  ")

for layer in encoder.layers:                                                                                                    # freeze the convolutional part of the model (the encoder)
    layer.trainable = False    

output_class, output_loc = define_two_head_model(encoder.output, len(synthetic_ifgs_settings['defo_sources']))                   # build the fully connected part of the model, and get the two model outputs
encoder_2head = Model(inputs=encoder.input, outputs=[output_class, output_loc])                                                  # define the full model   


loss_class = losses.categorical_crossentropy                                                                                     # good loss to use for classification problems, may need to switch to binary if only two classes though?
loss_loc = losses.mean_squared_error                                                                                             # loss for localisation

encoder_2head.compile(optimizer = fc_optimizer, loss=[loss_class, loss_loc], loss_weights = fc_loss_weights,
                    metrics = ['accuracy'])                                  

#%% step 06: Train the fully connected part of the CNN



if step_06_or_07 == 'step_06':
    fc_metrics = save_all_metrics(batch_metric_names)                                                            # needs to be initialised whether training (and writing info to this object), or loading results from previous run to it.  

    if fc_train_step_06:
        print('\nStep 06: Training the fully connected part of the CNN.')
        
        with open(step_06_dir / 'model_summary.txt', 'w') as f:                                                                 # send a summary of the model to a text file
            encoder_2head.summary(print_fn = lambda x: f.write(x + '\n'))
        try:
            plot_model(encoder_2head, to_file= step_06_dir / 'encoder_2head.png', show_shapes = True, show_layer_names = True)      # try to make a graphviz style image showing the complete model 
        except:
            print(f"Failed to create a .png of the model, but continuing anyway.  ")                               # this can easily fail, however, so simply alert the user and continue.  
        
        # 2: Callbacks
        step_06_model_save_names = ['best_class', 'best_loc']                                                               # names of files with the best models in saved each epoch (if metric is improving)
        epoch_save_class = ModelCheckpoint(step_06_dir / (f"{step_06_model_save_names[0]}" + "_epoch_{epoch:03d}.h5") ,
                                           monitor='val_class_dense3_accuracy', verbose = 1, 
                                           save_best_only=True, save_weights_only=False)
        
        epoch_save_loss = ModelCheckpoint(step_06_dir / (f"{step_06_model_save_names[1]}" + "_epoch_{epoch:03d}.h5" ) ,
                                           monitor='val_loc_dense6_loss', verbose = 1, 
                                           save_best_only=True, save_weights_only=False)
        
        fc_training_fig = training_figure_per_epoch(plot_all_metrics, fc_metrics.batch_metrics, fc_metrics.val_metrics, batch_metric_names[:4],                                             # plot metrics for all batches and epochs.  Note, miss last metric as it's localisation accuracy which is meaningless.  
                                                    'Fully connected network training', two_column = False, y_epoch_start = 0.8,                             # y_epoch_start original 0.8
                                                    out_path = step_06_dir )
        
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_lr=1e-8, verbose = 1)
        step_06_duplicate_remover = duplicate_model_remover(step_06_dir, step_06_model_save_names)

        callbacks = [fc_metrics, epoch_save_class, epoch_save_loss, fc_training_fig, reduce_lr, step_06_duplicate_remover]
        
        encoder_2head_history = encoder_2head.fit(fc_train_generator, epochs = fc_n_epochs, validation_data = fc_validation_generator,
                                                   shuffle = False, callbacks = callbacks )                                                 # shuffle false required for fast opening of batches from numpy files (it ensures idx increases, rather than bein random)
                                              
        with open(step_06_dir / 'training_history.pkl', 'wb') as f:                                                                         # save the metrics created when training the model
            pickle.dump(fc_metrics.batch_metrics, f)
            pickle.dump(fc_metrics.val_metrics, f)
                                              
        plt.switch_backend('Qt5agg')
        plot_all_metrics(fc_metrics.batch_metrics, fc_metrics.val_metrics, 
                         batch_metric_names[:4],  'Fully connected training', 
                         two_column = False, y_epoch_start = 0.)
    else:
        print('\nStep 06: Loading a model that was trained previously.')
        try:    
            with open(step_06_dir / 'training_history.pkl', 'rb') as f:                                                    
                fc_metrics.batch_metrics = pickle.load(f)
                fc_metrics.val_metrics = pickle.load(f)
                plot_all_metrics(fc_metrics.batch_metrics, fc_metrics.val_metrics, 
                                 batch_metric_names[:4],'Fully connected training (loaded)', 
                                 two_column = False)
        except:
            f"Failed to open and plot the results of a previous training run.  "
    
    
    if model_type == "resnet":    
        # As so large, load only the best localiation model to avoid GPU memoery issues. 
        encoder_2head_step_06 = build_2head_from_epochs(encoder_2head,  
                                                        models_dir = step_06_dir,
                                                        n_models=1)                   
    else:
        encoder_2head_step_06 = build_2head_from_epochs(encoder_2head,
                                                        models_dir = step_06_dir,
                                                        n_models=2) 
        
    evaluate_model(encoder_2head_step_06, X_test, X_test_m, Y_class_test, Y_loc_test, 
                   step_06_dir, n_plot = 15,
                   source_names = synthetic_ifgs_settings['defo_sources'])


#%% Step 07: Fine-tune the 5th convolutional block and the fully connected network.  


if step_06_or_07 == 'step_07':
    
    print("Fine-tuning the 5th block and fully connected network.  ")
    
    if model_type == "resnet":                                                                                              # resent is so large, load only the best localiation model to avoid GPU memoery issues. 
        encoder_2head_step_07 = build_2head_from_epochs(encoder_2head,  models_dir = step_06_dir, n_models=1) 
    else:
        encoder_2head_step_07 = build_2head_from_epochs(encoder_2head,  models_dir = step_06_dir, n_models=2) 
    
    
#    layers_name = [layer.name for layer in encoder_2head_step_07.layers]                                     # useful to check layer names.  

    for layer_n, layer in enumerate(encoder_2head_step_07.layers):
        if not isinstance(layer, layers.BatchNormalization):                        # if not batch normalistaion layetr
            if layer_n < ft_unfreeze_from:
                layer.trainable = False                                             # keep early parts frozen
            else:
                layer.trainable = True                                              # unfreeze some of the encoder
        else:
            layer.trainable = False                                             # but if batchnormalisation, keep frozen
            
    
    print(f"Step 07 trainable layers:")
    for layer in encoder_2head_step_07.layers:
        if layer.trainable:
            print(f"    {layer.name} (output shape: {layer.output_shape}")
    
    print_model_param_info(encoder_2head_step_07)
    
    encoder_2head_step_07.compile(optimizer = ft_optimizer, metrics=['accuracy'],                         # recompile as we've changed which layers can be trained/ optimizer etc.  
                                loss=[loss_class, loss_loc], loss_weights = ft_loss_weights)                                  
    
    # print(f"Checking that the step 07 model retains the performance of step 06 before further training:")
    # encoder_2head_step_07.evaluate(x = X_test, y = [Y_class_test, Y_loc_test], verbose = 1)                       # check that model is being loaded correctly (i.e. that metrics are OK)
        
    # Define the 5 callbacks
    ft_metrics = save_all_metrics(batch_metric_names)                                                            # initiaite the callback to record metrics every batch
    step_07_model_save_names = ['best_model']                                                               # names of files with the best models in saved each epoch (if metric is improving)
    epoch_save_both = ModelCheckpoint(step_07_dir / (f"{step_07_model_save_names[0]}" + "_epoch_{epoch:03d}.h5") ,
                                       monitor='val_loss', verbose = 1, 
                                       save_best_only=True, save_weights_only=False)
    
    ft_training_fig = training_figure_per_epoch(plot_all_metrics, ft_metrics.batch_metrics, ft_metrics.val_metrics, batch_metric_names[:4],                                             # plot metrics for all batches and epochs.  Note, miss last metric as it's localisation accuracy which is meaningless.  
                                                       'Fine-tune training', two_column = False, y_epoch_start = 0.0,
                                                       out_path = step_07_dir)
    step_07_duplicate_remover = duplicate_model_remover(step_07_dir, step_07_model_save_names)
    callbacks = [ft_metrics, epoch_save_both, ft_training_fig, step_07_duplicate_remover]
    
    
    # Either do the training or load history from previous run.  
    if ft_train_step_07:
        encoder_2head_history_step_07 = encoder_2head_step_07.fit(ft_train_generator, epochs = ft_n_epochs, validation_data = ft_validation_generator,
                                                              shuffle = False, callbacks = callbacks)                 # train
        
        with open(step_07_dir / 'training_history.pkl', 'wb') as f:                                                    # open the file
            pickle.dump(ft_metrics.batch_metrics, f)
            pickle.dump(ft_metrics.val_metrics, f)
                
        plt.switch_backend('Qt5agg')
        plot_all_metrics(ft_metrics.batch_metrics, ft_metrics.val_metrics, 
                         batch_metric_names[:4],'Fine tune training', two_column = False)
        
    else:
        try:
            with open(step_07_dir / 'training_history.pkl', 'rb') as f:                                                    
                ft_metrics.batch_metrics = pickle.load(f)
                ft_metrics.val_metrics = pickle.load(f)
            plot_all_metrics(ft_metrics.batch_metrics, ft_metrics.val_metrics, 
                             batch_metric_names[:4],'Fine tune training (loaded)', 
                             two_column = False)
        except:
            print(f"Failed to open the training history and plot it, but "
                  f"continuing anyway.  ")
            
    evaluate_model(encoder_2head_step_07, X_test, X_test_m, Y_class_test, Y_loc_test, 
                   step_07_dir, n_plot = 15,
                   source_names = synthetic_ifgs_settings['defo_sources'])
        
 




