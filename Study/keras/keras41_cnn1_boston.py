# CNN 으로 구성
# 2차원을 4차원으로 늘여서 하시오.

import numpy as np
from sklearn.datasets import load_boston

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (506, 13) (506,)
print("========================")
print(x[:5])
print(y[:10])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)

# 데이터 전처리(MinMaxScaler)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

print(x_train.shape, x_test.shape)  # (404, 13, 1, 1) (102, 13, 1, 1)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
          strides=1, input_shape=(13, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='same'))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=35, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_split=0.2, callbacks=[es], verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)

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

# sklearn Dense boston
# loss, mae :  15.000182151794434 2.9936561584472656
# RMSE :  3.873007354200969
# mse :  15.00018596569479
# R2 :  0.8205353096631526

# sklearn LSTM boston
# loss:  13.589542388916016
# RMSE :  3.6863995903399993
# mse :  13.589541939658915
# R2 :  0.8374124866452932

# skleran cnn boston
# loss :  12.100605964660645
# RMSE :  3.4785928650806563
# mse :  12.10060832099005
# R2 :  0.8552263331814386