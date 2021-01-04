# 2개의 파일을 만드시오.
# 1/ Early Stopping을 적용하지 않은 최고의 모델 (파라미터 튜닝 완벽하게)

# 1. 데이터
import numpy as np
from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape)     # (404, 13) (404,)

print(np.max(x_train), np.min(x_train)) # 최댓값 711.0, 최솟값 0.0
print(np.max(x_train[0]))               # 396.9

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.8, shuffle=True, random_state=66
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape, y_train.shape)     # (323, 13) (323,)

print(np.max(x_train), np.min(x_train)) # 최댓값 711.0, 최솟값 0.0      ----> 최댓값 1.0 , 최솟값 0.0
print(np.max(x_train[0]))               # max = 0.9908463918048585

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
aaa = Dense(200, activation='relu')(input1)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
aaa = Dense(200, activation='relu')(aaa)
outputs = Dense(1)(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=1)

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

# 실습 1
# loss, mse :  17.16822052001953 2.737574577331543
# RMSE :  4.143455294099354
# mse :  17.168221774199967
# R2 :  0.7937598743994111