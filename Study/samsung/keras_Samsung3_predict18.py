import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

x1 = np.load('./samsung/samsung2_data18.npy', allow_pickle=True)[0]
y1 = np.load('./samsung/samsung2_data18.npy', allow_pickle=True)[1]
x1_pred = np.load('./samsung/samsung2_data18.npy', allow_pickle=True)[2]
x2 = np.load('./samsung/KODEX_data18.npy', allow_pickle=True)[0]
x2_pred = np.load('./samsung/KODEX_data18.npy', allow_pickle=True)[1]
print(x1.shape, y1.shape, x1_pred.shape) # (1082, 6, 6) (1082, 1, 2) (1, 6, 6)
print(x2.shape, x2_pred.shape) # (1082, 6, 6) (1, 6, 6)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, train_size=0.8, shuffle=True, random_state=311
)

from sklearn.model_selection import train_test_split
x1_train, x1_val, x2_train, x2_val, y1_train, y1_val = train_test_split(
    x1_train, x2_train, y1_train, train_size=0.8, shuffle=True, random_state=311
)

print(x1_train.shape, x1_test.shape) # (692, 6, 6) (217, 6, 6)
print(x2_train.shape, x2_test.shape) # (692, 6, 6) (217, 6, 6)
print(x1_val.shape, x2_val.shape)    # (173, 6, 6) (173, 6, 6)
print(y1_train.shape, y1_test.shape, y1_val.shape) # (692, 1, 2) (217, 1, 2) (173, 1, 2)

y1_train = y1_train.reshape(y1_train.shape[0], 2)
y1_test = y1_test.reshape(y1_test.shape[0], 2)
y1_val = y1_val.reshape(y1_val.shape[0], 2)

print(y1_train.shape, y1_test.shape, y1_val.shape) # (692, 2) (217, 2) (173, 2)

# 전처리(3차원->2차원)
x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])
x1_val = x1_val.reshape(x1_val.shape[0], x1_val.shape[1]*x1_val.shape[2])
x1_pred = x1_pred.reshape(x1_pred.shape[0], x1_pred.shape[1]*x1_pred.shape[2])

x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2])
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2])
x2_val = x2_val.reshape(x2_val.shape[0], x2_val.shape[1]*x2_val.shape[2])
x2_pred = x2_pred.reshape(x2_pred.shape[0], x2_pred.shape[1]*x2_pred.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_val = scaler.transform(x1_val)
x1_pred = scaler.transform(x1_pred)

scaler.fit(x2_train)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
x2_val = scaler.transform(x2_val)
x2_pred = scaler.transform(x2_pred)

print(x1_train.shape, x1_test.shape) # (692, 36) (217, 36) 
print(x2_train.shape, x2_test.shape) # (692, 36) (217, 36)
print(x1_val.shape, x2_val.shape)    # (173, 36) (173, 36)
print(x1_pred.shape, x2_pred.shape)  # (1, 36) (1, 36)

# 전처리(2차원->3차원)
x1_train = x1_train.reshape(x1_train.shape[0], 6, 6)
x1_test = x1_test.reshape(x1_test.shape[0], 6, 6)
x1_val = x1_val.reshape(x1_val.shape[0], 6, 6)
x1_pred = x1_pred.reshape(x1_pred.shape[0], 6, 6)

x2_train = x2_train.reshape(x2_train.shape[0], 6, 6)
x2_test = x2_test.reshape(x2_test.shape[0], 6, 6)
x2_val = x2_val.reshape(x2_val.shape[0], 6, 6)
x2_pred = x2_pred.reshape(x2_pred.shape[0], 6, 6)

print(x1_train.shape, x1_test.shape) # (692, 6, 6) (217, 6, 6) 
print(x2_train.shape, x2_test.shape) # (692, 6, 6) (217, 6, 6)
print(x1_val.shape, x2_val.shape)    # (173, 6, 6) (173, 6, 6)
print(x1_pred.shape, x2_pred.shape)  # (1, 6, 6) (1, 6, 6)

# 2~3. 모델 구성, (컴파일, 훈련)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten

# 4. 평가, 예측
model = load_model('../data/modelcheckpoint/keras_Samsung3_01_min.hdf5')
# model = load_model('./samsung/keras_Samsung0118.h5')
result = model.evaluate([x1_test, x2_test], y1_test)
print("loss : ", result[0])
print("mae : ", result[1])

y_predict = model.predict([x1_test, x2_test])

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test, y_predict))
print("RMSE : ", RMSE(y1_test, y_predict))
print("mse : ", mean_squared_error(y1_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2 : ", r2)

predict = model.predict([x1_pred, x2_pred])
print("1월 18, 19일 삼성전자 시가 예측 : ", predict)
print("1월 19일 삼성전자 시가 예측 : ", predict[:,1:])

# loss :  199683.71875
# mae :  322.9437561035156
# RMSE :  446.8598
# mse :  199683.69
# R2 :  0.9971561012803087
# 1월 18, 19일 삼성시가 예측 : 
# [[87995.77 89930.96]]

# loss :  130280.6484375
# mae :  261.1093444824219
# RMSE :  360.9441
# mse :  130280.63
# R2 :  0.9981430045399327
# 1월 18, 19일 삼성시가 예측 : 
# [[89552.484 89258.17 ]]

# loss :  133355.671875
# mae :  265.740234375
# RMSE :  365.17896
# mse :  133355.67
# R2 :  0.9980992193486227
# 1월 18, 19일 삼성시가 예측 : 
# [[89262.06 89108.83]]

# loss :  277586.4375
# mae :  413.2773132324219
# RMSE :  526.86475
# mse :  277586.44
# R2 :  0.996045796784985
# 1월 18, 19일 삼성시가 예측 : 
# [[87451.49  89064.234]]

# loss :  1583039.875
# mae :  1172.85107421875
# RMSE :  1258.1891
# mse :  1583039.8
# R2 :  0.9765769782017499
# 1월 18, 19일 삼성전자 시가 예측 :  [[88055.36  87154.375]]
# 1월 19일 삼성전자 시가 예측 :  [[87154.375]]