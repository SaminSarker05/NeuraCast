'''

'''

import os
import numpy as np          
import pandas as pd              
import matplotlib.pyplot as plt  
import random
import tensorflow.keras as keras

x_train = pd.read_hdf('path-to-x-train-data')
y_train = pd.read_hdf('path-to-y-train-data')
x_valid = pd.read_hdf('path-to-x-valid-data')
y_valid = pd.read_hdf('path-to-y-valid-data')

# normalize data by substracting mean and dividing by std

variables = pd.read_csv('meta-data-path', index_col=0)
norm = variables[variables['type'].isin(['Interventions', 'Labs', 'Vitals'])]

for index, data in norm.iterrows():
  x_train[index] -= data['mean']
  x_valid[index] -= data['mean']
  x_train[index] /= (data['std'] + 1e-12)
  x_valid[index] /= (data['std'] + 1e-12)


# fill in missing data points with forward fill and fillna

fill = variables[variables['type'].isin(['Vitals', 'Labs'])].index

x_train[fill] = x_train.groupby(level=0)[fillvars].ffill()
x_valid[fill] = x_valid.groupby(level=0)[fillvars].ffill()

x_train.fillna(value=0, inplace=True)
x_valid.fillna(value=0, inplace=True)

