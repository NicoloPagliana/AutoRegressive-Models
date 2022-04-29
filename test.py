# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:39:22 2021

@author: nicopaglia
"""

from AutoRegModels import AutoReg, AutoRegMH

import pandas as pd
import matplotlib.pyplot as plt


# Load data
series = pd.read_csv('Data\MMM_2006-01-01_to_2018-01-01.csv', 
                     usecols=['Date','Close'],
                     index_col=0, 
                     date_parser=pd.to_datetime, 
                     squeeze=True)

# Plot the series
series.plot(ylabel=series.name)

# Train splitting
n_tr = len(series)-200

# Initialize AutoReg model
model = AutoReg(series=series, d=14, H=1)
# Fit the model on the first n_tr data
model.fit(idx_end=n_tr)
# Predict the data after n_tr
series_pred = model.predict_indices(idx_start=n_tr,idx_end=len(series)-1)

# Compute error
error = model.prediction_error(idx_start=n_tr,idx_end=len(series)-1)

# Plot preidction vs ground truth
fig = plt.figure()
plt.plot(series[n_tr:].index,series[n_tr:].values,label='True')
plt.plot(series_pred.index,series_pred.values,label='Prediction')
plt.ylabel(series.name)

