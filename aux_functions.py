#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:43:13 2020

@author: matthew
"""

#%%





#%%



#%%

def normalise_m1_1(r2_array):
    """ Rescale a rank 2 array so that it lies within the range[-1, 1]
    """
    import numpy as np
    r2_array = r2_array - np.min(r2_array)
    r2_array = 2 * (r2_array/np.max(r2_array))
    r2_array -= 1
    return r2_array

