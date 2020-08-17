#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:27:31 2020


# open matlab defo data

# make synthetic data?

# open real data
# augment real data?

# rescale and bottlenecks.  

# train model
    train in different ways


# save weights file to website?
 get weights file from website with wget? 

# predict model

@author: matthew
"""


#%% Things to set

# General settings
threshold_m = 0.015                  # deformation above this level will be included in the localization box
threshold_noise = 0.08               # threshold at which deformation is not condiered visible over topo and turb APS

defo_fraction = 0.8                     # this fraction of the deformation above the threshold must be on land (i.e. checked using the DEM)
turb_aps_mean = 2.5                        # max size of turbulent APS in cm
topo_aps_mean = 56.0                    # rad/km of delay for each sar acquisition 
topo_aps_var = 8.                       # rad/km variance for each sar acquisition

sar_speckle_strength = 0.05        # strength (variance) of gaussain speckled noise added to SAR real and imaginary
coh_scale = 1.4                 # sets spatial scale of incoherent areas
coh_threshold = 0.8                  # coherence is in range of 0-1, values above this are classed as incoherent

padding = 10
incidence = 30                  # ditto - 30 roughly right for S1
s1_wav = 0.056                 # Sentinel-1 wavelength in m
#pix_in_m = 92.6                    # pixels in m.  92.6m for a SRTM3 pixel on equator



synth_data_settings = {'n_files' : 2,
                       'n_per_file' : 5,
                       'n_classes' : 4,                                          # no def, inflation, dyke mogi would be 4
                       'min_deformation' : 0.02,                                 # deformation of at least X m is required
                       'max_deformation' : 0.25,                                 # but deformation must not be larger than X m
                       'n_pix' : 224}                                           # number of pixels of square outputs
    

dem_settings = {'volcano_csv_file' : '/home/matthew/university_work/02_neural_networks_python/Detect-Locate-CNN/smithsonian_name_lat_lon.csv',
                'srtm_tiles_folder' : '/home/matthew/university_work/data/SRTM/SRTM_3_tiles/',
                'void_fill' : True,
                'download' : True,
                'width_km' : 30,
                'water_mask_resolution' : 'f'}                      # c(rude), l(ow), i(termediate), h(igh), f(ull)

dependency_paths = {'syinterferopy_bin' : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'srtm_dem_tools_bin' : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/'}                                # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools

#%% Imports

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import pickle
import os
import sys


sys.path.append(dependency_paths['syinterferopy_bin'])
sys.path.append(dependency_paths['srtm_dem_tools_bin'])

from aux_functions import create_volcano_dems, create_random_defo_m, def_and_dem_translate, check_def_visible, normalise_m1_1
from syinterferopy_functions import coherence_mask, atmosphere_topo, atmosphere_turb

print('Temporary import of matrix_show for debugging.  ')
sys.path.append('/home/matthew/university_work/python_stuff/python_scripts/')
from small_plot_functions import matrix_show


#%% 01 Create of load dems for each volcano.  

print("Trying to open a .pkl of volcano dems... ", end = '')
try:
    with open('volcano_dems.pkl', 'rb') as f:
        volcanoes = pickle.load(f)
    f.close()
    print('Success!')
except:
    print('Failed.  Starting to make these (~2 minutes per volcano = 8 hours)...', end = '')
    volcanoes = create_volcano_dems(**dem_settings)                                                 # open all subaerial volcanoes and make DEMS for them.  
    with open(f'volcano_dems.pkl', 'wb') as f:
        pickle.dump(volcanoes, f)
    f.close()
    print('Done!')

dem_settings['ny'], dem_settings['nx'] = volcanoes[0]['dem'].shape

import sys; sys.exit()

#%% 03 Create synthetic interferograms using DEMS (units are metres at this point)

"""
what are they key args?
    a path to output to?
    kwargs set earlier about signal strength etc?
    outputs

"""
output_folder = './'
outputs = ['uuu', 'uud']

for output_extension in outputs:                                               # loop through them
    os.makedirs(output_folder + '/' + output_extension)                                  # making them


for out_file_num in range(synth_data_settings['n_files']):
    # begin to generate the data for this output file    
    succesful_generate = 0
    X_all = [ma.zeros((synth_data_settings['n_per_file'], synth_data_settings['n_pix'], synth_data_settings['n_pix'], 3)), 
             ma.zeros((synth_data_settings['n_per_file'], synth_data_settings['n_pix'], synth_data_settings['n_pix'], 3)),
             ma.zeros((synth_data_settings['n_per_file'], synth_data_settings['n_pix'], synth_data_settings['n_pix'], 3)),
             ma.zeros((synth_data_settings['n_per_file'], synth_data_settings['n_pix'], synth_data_settings['n_pix'], 3)),
             ma.zeros((synth_data_settings['n_per_file'], synth_data_settings['n_pix'], synth_data_settings['n_pix'], 3))]                                        # initate a list of empty arrays, each array will become one file  
    Y_class = np.zeros((synth_data_settings['n_per_file'],1))                                                                   # initate for labels showing type of deformation
    Y_loc = np.zeros((synth_data_settings['n_per_file'],4))                                                                     # initate for labels showing location of deformation
    
    while succesful_generate < synth_data_settings['n_per_file']:
        volcano_n = np.random.randint(0, len(volcanoes))     
        defo_source_n = np.random.randint(1, synth_data_settings['n_classes'])                                                  # random choice of which deformation source to use, exclude 0 as this is for no deformation class.  
        print(f"Volcano: {volcanoes[volcano_n]['name']} ", end = '')
        
        # 0: generate incoherence mask, choose dem, generate turbulent atmopshere, generate topo correalted atmosphere.  
        dem_large = volcanoes[volcano_n]['dem']                                                           # open a dem
        mask_coherence = coherence_mask(synth_data_settings['n_pix'], coh_scale, coh_threshold, verbose = True)           # if threshold is 0, all of the pixels are incoherent , and if 1, none are.  
        APS_turb_m, _ = atmosphere_turb(1, synth_data_settings['n_pix'], difference=False, interpolate_threshold = 70, mean_cm = turb_aps_mean)               
        APS_turb_m = APS_turb_m[0,]                                                                                             # remove the 1st dimension      
        print(f"| Turbulent APS and coherence mask generated ", end = '')
        if np.random.rand() < 0.5:
            asc_or_desc = 'asc'
            heading = 348
        else:
            asc_or_desc = 'desc'
            heading = 192
        
        # 1: generate deformation

        print(f"| Deformation source: {defo_source_n} ", end = '')
        
        # 1a: Move deformation and dem around and check that still visible
        viable = False; count = 0                                                                       # make dem and ph_def and check that def is visible
        while viable is False and count < 15:
            print(count)
            defo_m = create_random_defo_m(dem_large, volcanoes[volcano_n]['ll_extent'], volcanoes[volcano_n]['lonlat'], 
                              defo_source_n, synth_data_settings['min_deformation'], synth_data_settings['max_deformation'], asc_or_desc)                    # make a deformation signal.  By calling each time, a defo of the 
            
            defo_m, dem, viable_location, loc_list, mask_coh_water, mask_def, mask_water = def_and_dem_translate(dem_large, defo_m, mask_coherence, threshold = 0.3, 
                                                                                                                 n_pixs=synth_data_settings['n_pix'], defo_fraction = 0.8)
            f, axes = plt.subplots(1,2)
            matrix_show(defo_m, ax=axes[0], fig = f)
            matrix_show(dem, ax=axes[1], fig = f)
            for ax in axes:
                ax.set_aspect('equal')
            
            APS_topo_m = atmosphere_topo(dem, topo_aps_mean, topo_aps_var, difference=True)
            viable_snr = check_def_visible(defo_m, mask_def, APS_topo_m, APS_turb_m, threshold_noise)
            
            if (viable_location is False) or (viable_snr is False):
                viable = False
                count += 1
            else:
                viable = True
        
        dem = ma.array(dem, mask=mask_coh_water)                                                # mask dem - water and also incoherent bits.  Suppose we don't need to mask incoherent bits, but done to make consistent.  
        
    
        # 1b: Check that still visible over noise (topo APS and turb APS)
        

            
        # 2: combine signals
        if viable_location and viable_snr:
            print(f"Viable combination found.  ")
            if np.random.rand() < (1/ synth_data_settings['n_classes']):                                                        # a certain fraction of the signals shouldn't contain deformation.  
                ph_all = ((4*np.pi)/s1_wav)*(APS_topo_m + APS_turb_m)                                                           # comnbine the signals in m, then convert to rads
                Y_class[succesful_generate,0] = 0                                                                               # label as no deformation
                Y_loc[succesful_generate,:] = np.array([0,0,0,0])                                                               # 0 0 0 0 for no deformaiton
            else:
                ph_all = ((4*np.pi)/s1_wav) * (APS_topo_m + APS_turb_m + defo_m)                                                 # combine the signals in m, then convert to rads.  Here we include deformation.  
                Y_class[succesful_generate,0] = defo_source_n                                                                    # write the label as a number
                Y_loc[succesful_generate,:] = np.array([loc_list[0][0],loc_list[0][1],loc_list[1][0],loc_list[1][1]])            # location of deformation 
            

            ph_all_wrap = (ph_all + np.pi) % (2 * np.pi ) - np.pi                       # wrap
            #1 Genreate SAR amplitude 
            look_az = heading - 90     
            look_in = 90 - incidence                                                 # 
            ls = LightSource(azdeg=look_az, altdeg=look_in)
            sar_amplitude = normalise_m1_1(ls.hillshade(dem))                                                                   # in range [-1 1]
            
            # make real and imaginary
            ifg_real = sar_amplitude * np.cos(ph_all_wrap)
            ifg_imaginary = sar_amplitude * np.sin(ph_all_wrap)
            ifg_real = ifg_real + sar_speckle_strength * np.random.randn(ifg_real.shape[0], ifg_real.shape[1])                 # add noise to real
            ifg_imaginary = ifg_imaginary + sar_speckle_strength * np.random.randn(ifg_real.shape[0], ifg_real.shape[1])         # and imaginary

            # make things rank 4
            ph_all = ma.expand_dims(ma.expand_dims(ph_all, axis = 0), axis = 3)
            dem = ma.expand_dims(ma.expand_dims(dem, axis = 0), axis = 3)
            ifg_real = ma.expand_dims(ma.expand_dims(ifg_real, axis = 0), axis = 3)
            ifg_imaginary = ma.expand_dims(ma.expand_dims(ifg_imaginary, axis = 0), axis = 3)
            ph_all_wrap = ma.expand_dims(ma.expand_dims(ph_all_wrap, axis = 0), axis = 3)
            

            X_1s = []                                                                                   # this list will store one visualisation of 3 channel data in each form (uuu, uud, rid, www, wwwd)
            X_1s.append(ma.concatenate((ph_all, ph_all, ph_all), axis = 3))                             # uuu
            X_1s.append(ma.concatenate((ph_all, ph_all, dem), axis = 3))                                # uud
            X_1s.append(ma.concatenate((ifg_real, ifg_imaginary, dem), axis = 3))                       # rid                                                                                     X_1s.append(ma.concatenate((ph_all_wrap, ph_all_wrap, ph_all_wrap), axis = 3))              # www
            X_1s.append(ma.concatenate((ph_all_wrap, ph_all_wrap, dem), axis = 3))                      #wwd
            
            nans_present_flag = 0                                                           #check if any have nans in them.  
            for X_1 in X_1s:
                nans_present_current = np.max(np.ravel(np.isnan(X_1s)).astype(int))          # nans flag as an int
                if nans_present_current == 1:
                    nans_present_flag = 1                                                   # if there are nans, set the flag to 1
                else:
                    pass                                                                    # if no nans, don't update the flag (leave at 1 if already encountered nans)
            nans_present_flag = bool(nans_present_flag)                              # convert from int to boolean
                    
            if nans_present_flag:
                pass                                                                    # do nothing and start te loop over if we have nans
            else:
                for i in range(len(X_1s)):
                    X_all[i][succesful_generate,: :, :] = X_1s[i]
                succesful_generate += 1
                print(f"Generated {succesful_generate} of {synth_data_settings['n_per_file']} synthetic interferograms.  ")
        else:
            print(f"No viable combination found. Restarting.    ")
   
    # step back to loop through each file, not each ifg.   save what is required, not the most elegant way of doing this.  
    if 'uuu' in outputs:
        with open(f'{output_folder}uuu/data_file_{out_file_num}.pkl', 'wb') as f:
            pickle.dump(X_all[0], f)
            pickle.dump(Y_class, f)
            pickle.dump(Y_loc, f)

    if 'uud' in outputs:
        with open(f'{output_folder}uud/data_file_{out_file_num}.pkl', 'wb') as f:
            pickle.dump(X_all[1], f)
            pickle.dump(Y_class, f)
            pickle.dump(Y_loc, f)
            
    if 'rid' in outputs:
        with open(f'{output_folder}rid/data_file_{out_file_num}.pkl', 'wb') as f:
            pickle.dump(X_all[2], f)
            pickle.dump(Y_class, f)
            pickle.dump(Y_loc, f)

    if 'www' in outputs:
        with open(f'{output_folder}www/data_file_{out_file_num}.pkl', 'wb') as f:
            pickle.dump(X_all[3], f)
            pickle.dump(Y_class, f)
            pickle.dump(Y_loc, f)
            
    if 'wwd' in outputs:
        with open(f'{output_folder}wwd/data_file_{out_file_num}.pkl', 'wb') as f:
            pickle.dump(X_all[4], f)
            pickle.dump(Y_class, f)
            pickle.dump(Y_loc, f)
    

    


                    
                


            
            
            
            # pick whether to include deformaion
                # if includes deformation:
                        #, generate with random parameters (but ensure signal is correct size)
                        # check visible within scene
            # write label
            # conver to different forms.  
                
    
    # save rank 4 ma to file (or files, with different formats in different files.)
    
    
#%%
