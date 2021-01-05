# input_shape / input_length / input_dim

# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
# x_pred = np.array([5,6,7])

# x_pred = x_pred.reshape(1, 3)

print("x.shape : ", x.shape)    # (4, 3)
print("y.shape : ", y.shape)    # (4,)
# print("x_pred : ", x_pred.shape) # (3,)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_pred = scaler.transform(x_pred)

x = x.reshape(4, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)  

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(LSTM(10, activation='relu', input_length=3, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x, y, epochs=2000, batch_size=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

x_pred = np.array([5,6,7])  #(3,)   -> (1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print("result : ", result)

# LSTM
# 0.06006427854299545
# [[8.500742]]

# LSTM EarlyStopping
# 1.4192874004947953e-05
# [[8.02655]]

# LSTM EarlyStopping, MInMaxScaler
# loss :  4.747360435430892e-06
# result :  [[8.043124]]