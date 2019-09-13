import numpy as np
import datetime as dt
import pickle as pkl
from matplotlib import pyplot as plt
import seaborn as sbn
import pandas as pd
import sys

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.utils import resample

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils

from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

from keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import confusion_matrix

import random

# -----------------------------------------------------------------------------
BATCH_SIZE = 10000
EPOCHS = 1000
NUM_DOSES = 7
# -----------------------------------------------------------------------------
# BUILD MODEL ARCH

model_m = Sequential()

model_m.add(Reshape( (NUM_DOSES,1), input_shape=(NUM_DOSES,)))
#model_m.add(keras.layers.BatchNormalization())
model_m.add(Conv1D(100, 5, activation='relu', input_shape=(NUM_DOSES,) ))
model_m.add(Conv1D(20, 2, activation='relu'))
#model_m.add(MaxPooling1D(3))
#model_m.add(Conv1D(30, 3, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(10, activation='relu'))
model_m.add(Dense(5, activation='relu'))
model_m.add(Dense(1, activation='linear'))

print(model_m.summary())

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='./models/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=False),
    keras.callbacks.EarlyStopping(monitor='mean_square_error', patience=5,),
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
]

model_m.compile(loss="mean_absolute_percentage_error",
                optimizer=keras.optimizers.Adam(lr=1e-3, decay=1e-5), metrics=['mse'])

# -----------------------------------------------------------------------------
# DATA IN + PREPROCESSING
with open('./train_data.pkl', 'rb') as f:
    train = pkl.load(f)

X = train['x']
Y = train['y'].reshape(-1,1) #np_utils.to_categorical(train['y']).reshape((-1, NUM_DOSES*2,))

print('X shape: %s' %str(X.shape))
print('Y shape: %s' %str(Y.shape))

print('X[1]: %r' %X[1])
print('Y[1]: %r' %Y[1])
# -----------------------------------------------------------------------------
# TRAIN
history = model_m.fit(X,
                      Y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
