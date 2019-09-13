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

model = load_model('./best_model.159-55.24.h5')

with open('../data/test_data.pkl', 'rb') as f:
    test = pkl.load(f)

DOSES = np.array([0.01, 0.04, 0.12, 0.37, 1.11, 3.33, 10.0])

X = test['x']
Y = test['y']

Yhat = model.predict(X)

#MSE = ((Y - Yhat)**2).mean(axis=0)
#print('TEST MSE: %.3f' %MSE)

ndisp = 25
[print('Yhat: %.2f [%r] || Y: %.3f [%r] || correctly classified doses: %d/7' %(yhat, 1*(DOSES > yhat), y, 1*(DOSES > y),  np.sum([1 if a == b else 0 for a,b in zip(DOSES > yhat, DOSES > y)]))) for yhat, y in zip(Yhat[0:ndisp], Y[0:ndisp])]

DOSE_CLASS_ACC = np.array([np.sum([1 if a == b else 0 for a,b in zip(DOSES > yhat, DOSES > y)])/7 for yhat, y in zip(Yhat, Y)])
print('avg dose classification accuracy: %.1f%%' %(100.0*DOSE_CLASS_ACC.mean(axis=0)))

TP = np.sum([np.sum([1 if a == b and a == 1 else 0 for a,b in zip(DOSES > yhat, DOSES > y)]) for yhat, y in zip(Yhat, Y)])
FP = np.sum([np.sum([1 if a != b and b == 0 else 0 for a,b in zip(DOSES > yhat, DOSES > y)]) for yhat, y in zip(Yhat, Y)])
TN = np.sum([np.sum([1 if a == b and a == 0 else 0 for a,b in zip(DOSES > yhat, DOSES > y)]) for yhat, y in zip(Yhat, Y)])
FN = np.sum([np.sum([1 if a != b and b == 1 else 0 for a,b in zip(DOSES > yhat, DOSES > y)]) for yhat, y in zip(Yhat, Y)])

sens = TP / (TP + FN)
spec = TN / (TN + FP)

print('dose class specificity: %.1f%%' %(100.*spec))
print('dose class sensitivity: %.1f%%' %(100.*sens))
