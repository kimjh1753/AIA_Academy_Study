# 실습 : 모델을 구성하시오.

import numpy as np

from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)  # (506, 13)
print(y.shape)  # (506,)
print("==========================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0 0.0
print(dataset.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(dataset.DESCR)

# 데이터 전처리(MinMax)
# x = x /711.               
# x = (x - 최소) / (최대 - 최소)
#   = (x - np.mix(x) / (np.max(x) - np.min(x)))
print(np.max(x[0])) # 396.9

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = QuantileTransformer(output_distribution='normal')

# 과제 : 알아올것 ###################################################
scaler = MaxAbsScaler()
# scaler = PowerTransformer(method='yeo-johnson')
# scaler = PowerTransformer(method='box-cox')
####################################################################

scaler.fit(x)
x = scaler.transform(x)

# minmax
# print(np.max(x), np.min(x)) # 711.0 0.0 => 1.0 0.0
# print(np.max(x[0]))         # 0.9999999999999999

# standard
print(np.max(x), np.min(x)) # 711.0 0.0 => 24.678376790228196 -18.76100251828754
print(np.max(x[0]))         # 1.44

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)
print(x_train.shape)    # (404, 13)
print(y_train.shape)    # (404,)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(1000, activation='relu')(input1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=8,
          validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss, mae : ", loss, mae)


y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 전처리 전
# loss, mae :  15.841497421264648 3.3060100078582764
# RMSE :  3.980137940751986
# mse :  15.84149802741346
# R2 :  0.8104697138779872

# 전처리 후 x = x / 711.
# loss mae :  12.428569793701172 2.640331745147705
# RMSE :  3.5254177971244336
# mse :  12.428570644281695
# R2 :  0.851302538041412

# MinMaxscaler
# loss, mae :  14.775877952575684 2.445089817047119
# RMSE :  3.843940663250225
# mse :  14.775879822588578
# R2 :  0.823218945226423

# standardScaler 후
# loss, mae :  11.293810844421387 2.3707048892974854
# RMSE :  3.3606265998081124
# mse :  11.293811143337834
# R2 :  0.8648789872207374

# RobustScaler 후
# loss, mae :  7.473526954650879 2.046253204345703
# RMSE :  2.7337752031140585
# mse :  7.473526861161312
# R2 :  0.9105854962778589

# QuantileTransformer 후
# loss, mae :  15.258278846740723 2.4473443031311035
# RMSE :  3.906184821127002
# mse :  15.258279856802988
# R2 :  0.8174474319294035