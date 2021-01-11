# 당뇨병 회귀모델
# 실습 : 18에서 했던 것과 동일하게 19-1,2,3,4,5, EarlyStopping 까지 총 6개의 파일을 완성하시오.

# 다 : 1 mlp 모델
# 전처리 전

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

#1. DATA
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1

print(np.max(x), np.min(y))     #0.198787989657293 25.0  ---> 전처리 해야 함
print(dataset.feature_names)    # 10 column
                                # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape)    #(353, 10)
# print(x_test.shape)     #(89, 10)

#2. Modeling
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

#3. Compile, Train
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, verbose=1)

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score (y_test, y_predict)
print("R2 : ", r2)

# 전처리 전
# loss :  5250.9423828125
# mae :  54.374446868896484
# RMSE :  72.46338646455477
# # R2 :  0.19092300584609234