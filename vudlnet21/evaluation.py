#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:43:27 2024

@author: matthew
"""

#%%

def evaluate_model(model, X_test, X_test_m, Y_class_test, Y_loc_test, 
                   outdir, source_names, n_plot = 15):
    """ Evaludate a model on the testing data.  
    """
    import pickle
    import numpy as np
    
    from vudlnet21.plotting import plot_data_class_loc_caller

    # check that model is being loaded correctly.          
    #encoder_2head_step_06.evaluate(x = X_test, y = [Y_class_test, Y_loc_test], verbose = 1)                 
    
    # predict the deformation class and location
    Y_class_test_cnn_6, Y_loc_test_cnn_6 = model.predict(X_test, verbose = 1)                                                                     
    
    # plot all the testing data
    plot_data_class_loc_caller(X_test_m[:n_plot,], classes = Y_class_test[:n_plot,], 
                               classes_predicted = Y_class_test_cnn_6[:n_plot,],                            
                               locs = Y_loc_test[:n_plot,], 
                               locs_predicted = Y_loc_test_cnn_6[:n_plot,], 
                               source_names = source_names, 
                               window_title = 'Test data (after step 06)')
        

    # save some useful outputs.          
    with open(outdir / 'test_data_predictions.pkl', 'wb') as f:                                                    
        pickle.dump(X_test, f)
        pickle.dump(X_test_m, f)
        pickle.dump(Y_class_test, f)
        pickle.dump(Y_class_test_cnn_6, f)
        pickle.dump(Y_loc_test, f)
        pickle.dump(Y_loc_test_cnn_6, f)
    
    # also evaluate all the data, and by each label
    evaluate_results = {}
    evaluate_results['all'] = model.evaluate(X_test, y = [Y_class_test, Y_loc_test], 
                                             verbose = 1)                                                                 
    # evaluate by label.  
    for source_n, source in enumerate(source_names):
        print(f"Evaluating for source {source}:")
        # get only the data with that label
        args = np.ravel(np.argwhere(Y_class_test[:,source_n] == 1))                                                                                             
        evaluate_results[source] = model.evaluate(X_test[args,], 
                                                  y = [Y_class_test[args,], Y_loc_test[args,]], 
                                                  verbose = 1)                      
    # save the evaluation results
    with open(outdir / 'test_data_evaluations.pkl', 'wb') as f:                                                    
        pickle.dump(evaluate_results, f)    

