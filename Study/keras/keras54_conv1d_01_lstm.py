# keras23_3을 카피해서 Conv1D로 완성할것 
# LSTM과 비교

import numpy as np
# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print("x.shape : ", x.shape)    # (13, 3)
print("y.shape : ", y.shape)    # (13,)
print("x_pred : ", x_pred.shape) # (3,)

# 코딩하시오!!! LSTM
# 나는 80을 원하고 있다.

x_pred = x_pred.reshape(1, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(13, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2, padding='same', input_shape=(3,1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(1))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=2000, batch_size=16, callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=8)
print("loss : ", loss)

result = model.predict(x_pred)
print("result : ", result)

# LSTM EarlyStopping, MInMaxScaler
# loss :  [0.09484697878360748, 0.24658511579036713]
# result :  [[80.741135]] 

# conv1d_01_lstm
# loss :  [4.564180374145508, 1.48143470287323]
# result :  [[84.7642]]

# StandardScaler
# loss :  [11.66964340209961, 1.9976978302001953]
# result :  [[77.443634]]