# keras23_LSTM3_scale 을 함수형으로 코딩

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

x_pred = x_pred.reshape(1, 3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(3,))
aaa = Dense(10, activation='relu')(input1)
aaa = Dense(20)(aaa)
aaa = Dense(20)(aaa)
outputs = Dense(1)(aaa)
model = Model(inputs = input1, outputs = outputs)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=45, mode='auto')
model.fit(x, y, batch_size=1, epochs=4000, callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)

result = model.predict(x_pred)
print(result)

# keras23_LSTM3_scale
# [0.006796032190322876, 0.07546142488718033]
# [[80.307556]]

# keras26_LSTM_hamsu
# loss :  [0.00023922852415125817, 0.012460635043680668]
# [[80.054405]]