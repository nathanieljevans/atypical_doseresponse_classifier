import numpy as np
import datetime as dt
import pickle as pkl
from matplotlib import pyplot as plt
import seaborn as sbn
import pandas as pd
import sys

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import to_categorical

from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.metrics import auc, roc_curve, confusion_matrix

import random

# -----------------------------------------------------------------------------
BATCH_SIZE = 10000   # sub-sampling of training data
EPOCHS = 1000        # number of epochs to train the data on
NUM_DOSES = 7        # number of dose points used in this assay
N_REPL = 1           # number of replicates used in this model
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# BUILD MODEL ARCH
# -----------------------------------------------------------------------------
try:
    model_m = Sequential()

    #model_m.add(Reshape( (N_REPL + 1, NUM_DOSES, 1), input_shape=(N_REPL + 1, NUM_DOSES) ))

    model_m.add(Conv2D(100, kernel_size=3, padding='same', activation='relu', input_shape=(N_REPL + 1, NUM_DOSES, 1) ))

    model_m.add(Conv2D(20, kernel_size=2, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model_m.add(Dropout(0.5))
    model_m.add(Flatten())
    model_m.add(Dense(10, activation='relu'))
    model_m.add(Dense(10, activation='relu'))
    model_m.add(Dense(NUM_DOSES*2, activation='softmax'))
    model_m.add(Reshape( (7,2) ))

    print(model_m.summary())

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='./models/best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=False),
        keras.callbacks.EarlyStopping(monitor='mean_square_error', patience=5,),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    ]

    model_m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except:
    model_m.summary()
    raise
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# DATA IN + PREPROCESSING
# -----------------------------------------------------------------------------
with open('./classifier_data/train_data.pkl', 'rb') as f:
    train = pkl.load(f)

X = train['x'].reshape(-1, 2, 7, 1)
Y = to_categorical( train['y'].reshape(-1, 7, 1) )

print('X shape: %s' %str(X.shape))
print('Y shape: %s' %str(Y.shape))

print('X[0]: %r' %X[0])
print('Y[0]: %r' %Y[0])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
history = model_m.fit(X,
                      Y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
