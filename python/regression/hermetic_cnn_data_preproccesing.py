
import sys
import pandas as pd
import numpy as np
import pickle as pkl


if __name__ == '__main__':

    train_path = sys.argv[1]

    dat = pd.read_csv(train_path)

    dose_names = ['DOSE_0.01', 'DOSE_0.04','DOSE_0.12','DOSE_0.37','DOSE_1.11','DOSE_3.33','DOSE_10.00']
    doses = [0.01, 0.04, 0.12, 0.37, 1.11, 3.33, 10.0]

    print( dat[dose_names].head() )

    dose_vectors = dat[dose_names].values
    transitions = dat['t'].values

    split = int(dose_vectors.shape[0]*0.70) # 50/50 split -- we can produce as much data as we want...

    xTRAIN = dose_vectors[:split,:]
    xTEST = dose_vectors[split:, :]

    yTRAIN = transitions[:split] #[ [1 if d > t else 0 for d in doses] for t in transitions[:split]]
    yTEST = transitions[split:] #[ [1 if d > t else 0 for d in doses] for t in transitions[split:]]

    print(xTRAIN[1])

    print(yTRAIN[1])

    with open('./train_data.pkl', 'wb') as f:
        pkl.dump({'x':xTRAIN, 'y':yTRAIN}, f)

    with open('./test_data.pkl', 'wb') as f:
        pkl.dump({'x':xTEST, 'y':yTEST}, f)
