import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats

x = []
for i in range(1, 35):
    df = pd.read_csv(f'../study/LPD_COMPETITION/answer/answer{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)
# print(x.shape) # (5, 72000, 1)

df = pd.read_csv(f'../study/LPD_COMPETITION/answer/answer{i}.csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        a = []
        for k in range(34):
            a.append(x[k,i,j].astype('int'))
        a = np.array(a)
        # df.iloc[[i],[j]] = (pd.DataFrame(a).astype('int').quantile(0.5,axis = 0)[0]).astype('float32')
        df.iloc[[i][j]] = (stats.mode(a)[0])

y = pd.DataFrame(df, index = None, columns = None)
print(y)
y.to_csv('../study/LPD_COMPETITION/answer_final/answer_result33.csv')   

# from scipy import stats

# aa = stats.mode()

# ( 1 2 2 2 5) > 2
# (1 2 2 2 5) > 2

