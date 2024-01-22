'''
'''

import os
import numpy as np          
import pandas as pd              
import matplotlib.pyplot as plt  
import random
import tensorflow.keras as keras

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking
from tensorflow.keras.optimizers import RMSprop


# define input and mask out padded values

data = Input((None, x_train.shape[-1]), name='data')
mask = Masking(0, name='input_masked')(data)


# stack the LSTM layers each with 128 units

lstm_kwargs = {'dropout': 0.25, 'recurrent_dropout': 0.1, 'return_sequences': True, 'implementation': 2}
lstm = LSTM(128, name='lstm', **lstm_kwargs)(mask)


# define output layer with sigmoid activation function for classification

output = TimeDistributed(Dense(1, activation='sigmoid'), name='output')(lstm)


# use RMS optimizer and binary cross entropy as loss function

optimizer = RMSprop(lr=0.005)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

