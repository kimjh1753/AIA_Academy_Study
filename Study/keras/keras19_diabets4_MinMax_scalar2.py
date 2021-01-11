# 당뇨병 회귀모델
# 실습 : 18에서 했던 것과 동일하게 19-1,2,3,4,5, EarlyStopping 까지 총 6개의 파일을 완성하시오.

# MinMaxScalar

# [3] x_train 데이터만 전처리 한다.
# validation split

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

# 1. 데이터
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])

print(x.shape, y.shape)         # (442, 10) (442,) input = 10, output = 1

print(np.max(x), np.min(y))     # 0.198787989657293 25.0  ---> 전처리 해야 함
print(dataset.feature_names)    # 10 column
                                # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ********* 데이터 전처리 ( MinMax ) *********
#[1] x를 0 ~ 1로 만들기 위해서 모든 데이터를 최댓값 711 로 나눈다. 
# y는 바꿀 필요 없음
# x = x / 711.     # 소수점으로 만들어 주기 위해서 숫자 뒤에 '.' 을 붙인다.
                                
# print(dataset.DESCR)
'''
  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, T-Cells (a type of white blood cells)
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, thyroid stimulating hormone
      - s5      ltg, lamotrigine
      - s6      glu, blood sugar level
'''

print(np.max(x[0])) # max = 396.9 ??? 최댓값 711 인데 왜 396 이 나왔을까? ---> 컬럼마다 최솟값과 최댓값이 다르다. 

# [2] 최소가 0인지 몰랐을 때 -----> sklearn에서 수식 제공 중 -----------> MinMaxScaler
# x = (x-최소값) / (최댓값-최소값)
# x = (x - np.min(x)) / (np.max(x)-np.max(x))

# MinMaxScaler 사용
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)     # 질문 : transform 왜 해?
# print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0      ----> 최댓값 1.0 , 최솟값 0.0
# print(np.max(x[0]))         # max = 0.9999999999999999     -----> 컬럼마다 최솟값과 최댓값을 적용해서 구해준다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=66
)
# x_train, x_val, y_train, y_val = train_test_split(
#         x, y, train_size=0.8, shuffle=True, random_state=66
# )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
# scaler.transform(x_val)

print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0      ----> 최댓값 1.0 , 최솟값 0.0
print(np.max(x[0]))         # max = 0.9999999999999999     -----> 컬럼마다 최솟값과 최댓값을 적용해서 구해준다.

# print(x_train.shape)    #(353, 10)
# print(x_test.shape)     #(89, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(10,))
Dense1 = Dense(120, activation='relu')(input1)
Dense1 = Dense(120, activation='relu')(Dense1)
Dense1 = Dense(120, activation='relu')(Dense1)
Dense1 = Dense(120, activation='relu')(Dense1)
Dense1 = Dense(120, activation='relu')(Dense1)
Dense1 = Dense(120, activation='relu')(Dense1)
Dense1 = Dense(120, activation='relu')(Dense1)
Dense1 = Dense(120, activation='relu')(Dense1)
Dense1 = Dense(120, activation='relu')(Dense1)
outputs = Dense(1)(Dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=8)
print("loss, mse : ", loss, mse)

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))    
print("mse : ", mean_squared_error(y_predict, y_test))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 전처리 전
# loss :  5250.9423828125
# mae :  54.374446868896484
# RMSE :  72.46338646455477
# R2 :  0.19092300584609234

# MinMax Scaler2
# loss, mse :  4612.8642578125 52.08906555175781
# RMSE :  67.91807113484292
# mse :  4612.864386677584
# R2 :  0.28923949573082375
