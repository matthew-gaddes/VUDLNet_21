#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:14:07 2020

@author: matthew
"""

import tensorflow as tf

#%%


def build_2head_from_epochs(model, models_dir, n_epoch_class, n_epoch_loc, source_names = None, X_test = None, Y_class_test = None, Y_loc_test = None, n_plot = 15):
    """
    The performance of each head of a two headed model may be best at different epochs for different heads.  
    Merge the two heads to create the optimal model.  
    
    Inputs:
        model | keras model | model to be updated.  
        models_dir | pathlib Path | directory to where the model was stored after each epoch
        n_epoch_class | int | epoch number of the classification head to use
        n_epoch_loc | int | epoch number of the localisation head to use
        X_test | numpy rank 4 | test data
        Y_class_test | numpy rank 2 | classification labels.  
        Y_loc_test | numpy rank 2 | localisation labels
        sources_names | list of strings | sources names as used by the classification head
        n_plot | int | number of testing data to plot.  
        
    Returns:
        model | keras model | updated model
        figure
        
    History:
        2022_11_01 | MEG | Written
    
    """
    import tensorflow.keras as keras
    from vudlnet21.plotting import plot_data_class_loc_caller
    
    print(f"Loading the two models at the required epochs...", end = '')
    class_model = keras.models.load_model(models_dir / f"vgg16_2head_fc_epoch_{n_epoch_class:03d}.h5")          # load the best classification model
    loc_model = keras.models.load_model(models_dir / f"vgg16_2head_fc_epoch_{n_epoch_loc:03d}.h5")              # load the best localisation model
    print(f" Done.  ")
    
    #model = keras.models.clone_model(model)                                                                    # if you don't want to change the original model.  Would need to update remaining references to it though.  
    
    for layer in model.layers:                                                                                  # loop through all layers
        if layer.name[:3] == 'loc':                                                                                     # if it's a localisation layer that we created..
            model.get_layer(layer.name).set_weights(loc_model.get_layer(layer.name).get_weights())              # update the weights
        if layer.name[:5] == 'class':                                                                                   # same for if a classification layer...
            model.get_layer(layer.name).set_weights(class_model.get_layer(layer.name).get_weights())
            
    if X_test != None:
        print(f"Starting the forward pass of the testing data:")
        Y_class_test_cnn, Y_loc_test_cnn = model.predict(X_test, verbose = 1, batch_size = 15)                                                               # predict to make new Ys
        
        print(f"Staring to plot the testing data...", end = '')
        plot_data_class_loc_caller(X_test[:n_plot,], classes = Y_class_test[:n_plot,], classes_predicted = Y_class_test_cnn[:n_plot,],                    # plot some of the testing data.  
                                                   locs = Y_loc_test[:n_plot,],      locs_predicted = Y_loc_test_cnn[:n_plot,], 
                                   source_names = source_names, 
                                   window_title = f"Test data - class_epoch:{n_epoch_class} loc_epoch:{n_epoch_loc}")
        print(f" Done.  ")
    else:
        print(f"No test data provided.  ")
    
    return model
            
    
   

#%%


class numpy_files_sequence(tf.keras.utils.Sequence):                                                                                  # inheritance not tested like ths.  
    """A data generator for use with .npz files that contains X, Y_class, and Y_loc.  Can be used with either training, validation, or testing data.  
    Key methods:
            __len__                 to get the number of batches to pass all the data through (i.e. for one epoch)                        
            __getitem__             to get a batch of data.  
    If built correctly, it should guarantee that each sample is only used once per epoch.  
    """
        
    def __init__(self, file_list, batch_size):                                          # constructor
        """
        Inputs:
            file_list | list of strings or paths | locations of numpy files of data.  
            batch_size | int | number of data for each batch.  Note tested if larger than the number of data in a single file.  
        """
        self.file_list = file_list
        self.batch_size = batch_size
        self.n_file_loaded = -1

    def __len__(self):                                                      # number of batches in an epoch
        """As one large file (e.g. 1000 data) can't be used as a batch on a GPU (but maybe on a CPU?), we will break each 
        file into n_batches_per_file.  Therefore, the total number of batches (per epoch) will be n_files x n_batches"""
        
        import numpy as np
        n_files = len(self.file_list)                                                           # get the number of data files.  
        n_data_per_file = np.load(self.file_list[0])['X'].shape[0]                              # get the number of data in a file (assumed to be the same for all files)
        n_batches_per_file = int(np.ceil(n_data_per_file / self.batch_size))                    # the number of batches required to cover every data in the file.  
        n_batches = n_files * n_batches_per_file
        self.n_batches_per_file = n_batches_per_file
        return n_batches

    def __getitem__(self, idx):                                             # iterates over the data and returns a complete batch, index is a number upto the number of batches set by __len__, with each number being used once but in a random order.  
        
        import numpy as np
        n_file, n_batch = divmod(idx, self.n_batches_per_file)                              # idx tells us which batch, but that needs mapping to a file and to which batch in that file.  

        if self.n_file_loaded != n_file:                       
            data = np.load(self.file_list[n_file])                                              # load the correct numpy file.  
            self.X = data['X']
            self.Y_class = data['Y_class']
            self.Y_loc = data['Y_loc']
            self.n_file_loaded = n_file
        
        X_batch = self.X[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]
        Y_class = self.Y_class[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]
        Y_loc = self.Y_loc[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]
        
        return X_batch, [Y_class, Y_loc]





#%%


def define_two_head_model(model_input, n_class_outputs = 3):
    """ Define the two headed model that we have designed to performed classification and localisation.  
    Inputs:
        model_input | tensorflow.python.framework.ops.Tensor | The shape of the tensor that will be input to our model.  Usually the output of VGG16 (?x7x7x512)  Nb ? = batch size.  
        n_class_output | int | For a one hot encoding style output, there must be as many neurons as classes
    Returns:
        output_class |tensorflow.python.framework.ops.Tensor | The shape of the tensor output by the classifiction head.  Usually ?x3
        output_loc | tensorflow.python.framework.ops.Tensor | The shape of the tensor output by the localisation head.  Usually ?x4
    History:
        2020_11_11 | MEG | Written
        2021_08_24 | MEG | Remove a layer of 1024 neurons.  
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    
    vgg16_block_1to5_flat = Flatten(name = 'vgg16_block_1to5_flat')(model_input)                              # flatten the model input (ie deep representation turned into a column vector)

    # 1: the clasification head
    x = Dropout(0.2, name='class_dropout1')(vgg16_block_1to5_flat)
    x = Dense(256, activation='relu', name='class_dense1')(x)                                                 # add a fully connected layer
    x = Dropout(0.2, name='class_dropout2')(x)
    x = Dense(128, activation='relu', name='class_dense2')(x)                                                 # add a fully connected layer
    output_class = Dense(n_class_outputs, activation='softmax',  name = 'class_dense3')(x)                  # and an ouput layer with 7 outputs (ie one per label)
    
    # 2: the localization head
    x = Dense(2048, activation='relu', name='loc_dense1')(vgg16_block_1to5_flat)                                                 # add a fully connected layer
    x = Dense(1024, activation='relu', name='loc_dense2')(x)                                                 # add a fully connected layer
    x = Dropout(0.2, name='loc_dropout1')(x)
    x = Dense(512, activation='relu', name='loc_dense4')(x)                                                 # add a fully connected layer
    x = Dense(128, activation='relu', name='loc_dense5')(x)                                                 # add a fully connected layer
    output_loc = Dense(4, name='loc_dense6')(x)        
    
    return output_class, output_loc
    





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


