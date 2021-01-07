# keras23_3을 카피해서
# LSTM층을 두개를 만들것!!!

# 예)
# model.add(LSTM(10, input_shape=(3,1)))
# model.add(LSTM(10)

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

print(x.shape[0])   # 13
print(x.shape[1])   # 3

# 코딩하시오!!! LSTM
# 나는 80을 원하고 있다.

x_pred = x_pred.reshape(1, 3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
x_pred = x_pred.reshape(1, 3, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True)) 
# return_sequences=True -> (None, 3, 1)을 그대로 던져 주고 model.summary는 (None, 3, 10) 나온다. 여기서 10은 나가는 노드의 개수이다.
model.add(LSTM(20))  # 두번째 LSTM 부터는 c계열 값 적용 X
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480                # return_sequences=True -> (None, 3, 1)을 그대로 던져 주고 
#                                                                             model.summary는 (None, 3, 10) 나온다. 여기서 10은 나가는 노드의 개수이다.
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 20)                2480
# _________________________________________________________________
# dense (Dense)                (None, 20)                420
# _________________________________________________________________
# dense_1 (Dense)              (None, 20)                420
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                210
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 4,021
# Trainable params: 4,021
# Non-trainable params: 0
# _________________________________________________________________
# PS C:\Study> 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=40, mode='auto')
model.fit(x, y, epochs=3000, batch_size=8, callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=8)
print("loss : ", loss)

result = model.predict(x_pred)
print("result : ", result)

# LSTM 두개 사용
# loss :  [0.00010644139547366649, 0.008578007109463215]
# result :  [[80.968254]]
