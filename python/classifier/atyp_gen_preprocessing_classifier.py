
import sys
import pandas as pd
import numpy as np
import pickle as pkl

TEST_TRAIN_SPLIT = 0.7


if __name__ == '__main__':

    train_path = sys.argv[1]

    dat = pd.read_csv(train_path)

    print(dat.head())

    conc_trunc = np.array([float(x.replace('DOSE_','')) for x in dat.columns if 'DOSE_' in x ])
    dose_cols = [x for x in dat.columns if 'DOSE_' in x ]

    print('Truncated doses: %r' %conc_trunc)

    cv = dat[dose_cols].values


    X = np.array([np.array([conc_trunc,cv_obs]) for cv_obs in cv])
    Y = np.array([1*(conc_trunc > t) for t in dat['t']])

    print('reshaping data to functional form, this may take a moment...')
    print('X shape: %s' %str(X.shape))
    print('Y shape: %s' %str(Y.shape))
    print('complete.')

    print('first 3 obs (X,Y)')
    print(X[0:3])
    print(Y[0:3])

    split = int(X.shape[0]*TEST_TRAIN_SPLIT)

    xTRAIN = X[:split,:]
    xTEST = X[split:, :]

    yTRAIN = Y[:split]
    yTEST = Y[split:]

    with open('./classifier_data/train_data.pkl', 'wb') as f:
        pkl.dump({'x':xTRAIN, 'y':yTRAIN}, f)

    with open('./classifier_data/test_data.pkl', 'wb') as f:
        pkl.dump({'x':xTEST, 'y':yTEST}, f)
