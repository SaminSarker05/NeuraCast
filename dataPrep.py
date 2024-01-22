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
x_test = pd.read_hdf('path-to-x-test-data')
y_test = pd.read_hdf('path-to-y-test-data')

# normalize data by substracting mean and dividing by std

variables = pd.read_csv('meta-data-path', index_col=0)
norm = variables[variables['type'].isin(['Interventions', 'Labs', 'Vitals'])]

for index, data in norm.iterrows():
  x_train[index] -= data['mean']
  x_valid[index] -= data['mean']
  x_train[index] /= (data['std'] + 1e-12)
  x_valid[index] /= (data['std'] + 1e-12)

