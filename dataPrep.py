'''
1. read data from hdf5 file paths
2. normalize features to enchance model interpretability
3. fill missing variables in each patient observation
4. pad encounters to have same observation count
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


# padd all patient encounters since observation count not identical

from tensorflow.keras.preprocessing import sequence

maxlen = 500

trainid = x_train.index.levels[0]
validid = x_valid.index.levels[0]

x_train = [x_train.loc[patient].values for patient in trainid]
y_train = [x_train.loc[patient].values for patient in trainid]

x_train = sequence.pad_sequences(x_train, dtype='float32', maxlen=maxlen, padding='post', truncating='post')
y_train = sequence.pad_sequences(y_train, dtype='float32', maxlen=maxlen, padding='post', truncating='post')

x_valid = [x_valid.loc[patient].values for patient in validid]
y_valid = [y_valid.loc[patient].values for patient in validid]

x_valid = sequence.pad_sequences(x_valid, dtype='float32', maxlen=maxlen, padding='post', truncating='post')
y_valid = sequence.pad_sequences(y_valid, dtype='float32', maxlen=maxlen, padding='post', truncating='post')