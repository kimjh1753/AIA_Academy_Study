# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
# x_pred = np.array([5,6,7])  #(3,)   -> (1, 3, 1)

print("x.shape : ", x.shape)    # (4, 3)
print("y.shape : ", y.shape)    # (4,)

# x_pred = x_pred.reshape(1, 3)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_pred = scaler.transform(x_pred)

# x = x.reshape(4, 3, 1)  
# x_pred = x_pred.reshape(1, 3, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stoppping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x, y, epochs=2000, batch_size=1, callbacks=[early_stoppping])

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

x_pred = np.array([5,6,7])  #(3,)   -> (1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print("result : ", result)

# LSTM
# loss :  0.04061836376786232
# result :  [[8.214923]]

# LSTM EarlyStopping
# loss :  2.931799940597557e-07
# result :  [[8.037477]]

# LSTM EarlyStopping, MInMaxScaler
# loss :  1.115599616241525e-06
# result :  [[8.052482]]