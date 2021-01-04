# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성하시오.
# 실습 19_3

#1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 0.198787989657293 -0.137767225690012
print(datasets.feature_names)
# print(datasets.DESCR)

# 데이터 전처리(MinMax)
# x = x /711.               
# x = (x - 최소) / (최대 - 최소)
#   = (x - np.mix(x) / (np.max(x) - np.min(x)))

print(np.max(x), np.min(x)) # 0.198787989657293 -0.137767225690012 => 1.0 0.0
print(np.max(x[0]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)
x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape)
print(x_test.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(10,))
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=8,
          validation_data=(x_val, y_val), verbose=1)

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

# 실습 1
# loss, mae :  3866.4638671875 52.442867279052734
# RMSE :  62.18089290432054
# mse :  3866.46344237858
# R2 :  0.40424662949550205

# 실습 2
# loss, mae :  3457.067138671875 49.00520706176758
# RMSE :  58.79682791534668
# mse :  3457.066972906891
# R2 :  0.4673273569342071

# 실습 3
# loss, mae :  3354.364013671875 47.564918518066406
# RMSE :  57.91686968097033
# mse :  3354.3637936425002
# R2 :  0.4831520934460434

# 실습 4
# loss, mae :  3347.784423828125 47.468528747558594
# RMSE :  57.860042197813684
# mse :  3347.7844831327798
# R2 :  0.4841658483851953

# 실습 5
# loss, mae :  3274.51513671875 47.460147857666016
# RMSE :  57.223379018574725
# mse :  3274.5151063034577
# R2 :  0.4954553585154087