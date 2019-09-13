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
from keras.utils import to_categorical

import random

model = load_model('./best_model.20-0.10.h5')

with open('./classifier_data/test_data.pkl', 'rb') as f:
    test = pkl.load(f)

X = test['x'].reshape(-1, 2, 7, 1)
Y = to_categorical( test['y'].reshape(-1, 7, 1) )

print('predicting test data, this may take a moment...')
Yhat = model.predict(X)
print('predictions complete.')

out_data = None
for i,x,y,yhat in zip(range(X.shape[0]), X, Y, Yhat):
    df = pd.DataFrame({'conc':x.reshape(2,7)[0], 'cell_viab':x.reshape(2,7)[1], 'truth_norm':y[:,0], 'truth_atyp':y[:,1], 'pred_norm':yhat[:,0], 'pred_atyp':yhat[:,1], 'test_index':i})
    #df = df.sort_values(by='conc')
    #df = df.assign(log_conc = np.log10(df['conc']))

    if (i < 5):
        df = df.sort_values(by='conc')
        df = df.assign(log_conc = np.log10(df['conc']))

        print(df)

        f, ax = plt.subplots(3,1,figsize=(12,7), sharex=True)

        sbn.scatterplot(x='log_conc', y='cell_viab', color='red', data=df, ax=ax[0])

        sbn.lineplot(x='log_conc', y='truth_atyp', color='blue', label='truth=atypical', data=df, ax=ax[1])
        sbn.lineplot(x='log_conc', y='truth_norm', color='blue', label='truth=normal', data=df, ax=ax[2])
        #ax.lines[0].set_linestyle("--")

        sbn.lineplot(x='log_conc', y='pred_atyp', color='green', label='predicted=atypical', data=df, ax=ax[1])
        sbn.lineplot(x='log_conc', y='pred_norm', color='green', label='predicted=normal', data=df, ax=ax[2])
        #ax.lines[4].set_linestyle("--")

        ax[0].set_title('dose-response')
        ax[1].set_title('atypical truth/prediction')
        ax[2].set_title('normal truth/prediction')

        ax[1].set_yscale('log')
        ax[2].set_yscale('log')

        ax[1].set_ylabel('logliklihood')
        ax[2].set_ylabel('logliklihood')

        plt.legend()
        plt.show()

    elif (i%1000==0):
        print('packaging data... [%.2f%%]' %(100*i/X.shape[0]), end='\r')

    if i == 0:
        out_data = df
    else:
        out_data = out_data.append(df, ignore_index=True)

out_data.to_csv('./classifier_test_results.csv', index=False)

'''
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
'''
