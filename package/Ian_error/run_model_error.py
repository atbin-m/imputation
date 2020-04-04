#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:15:03 2019

@author: ian
"""

import pandas as pd
import model_error as moderr

path = 'C:/Users/22372102/Desktop/Quick_Access/Python_exe/ustar_checking/Gingin_2018_L4_imputed.csv'

# Import the data to a pandas dataframe
df = pd.read_csv(path, index_col = ['DateTime'], parse_dates = True, 
                 infer_datetime_format = True)
df.sort_index(inplace = True)

# Tell the algorithm the names used for your variables
names_dict = {'Observations': 'Fc', 'Model': 'Fc_SOLO'} # 'Fc_predicted_eXtreme Gradient Boost'

# Scaling coefficient kwarg is used to ensure that you end up with gCm-2a-1
# (from umol to mol multiply by 10^-6, from mol to g multiply by 12) 
x = moderr.model_error(dataframe = df, scaling_coefficient = 10**-6 * 12, 
                       names_dict = names_dict)

# Do a single realisation of model error
single = x.estimate_model_error()

# Do a full propagation over the entire dataset
uncert = x.propagate_model_error()

# Plot the histogram and overlaid Gaussian PDF of the values
x.plot_pdf()