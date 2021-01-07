# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
x_pred = np.array([5,6,7])

# x_pred = x_pred.reshape(1, 3)

print("x.shape : ", x.shape)        # (4, 3)
print("y.shape : ", y.shape)        # (4,)
# print("x_pred : ", x_pred.shape)    # (3,)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_pred = scaler.transform(x_pred)

x = x.reshape(4, 3, 1)  
# x_pred = x_pred.reshape(1, 3, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=2000, batch_size=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate(x, y)
print(loss)

x_pred = np.array([5,6,7])  #(3,)   -> (1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print(result)

# LSTM
# 0.02622945047914982
# [[8.293237]]

# SimpleRNN
# 5.233444971963763e-05
# [[8.011615]]

# GRU
# 0.02622945047914982
# [[8.293237]]

# GRU (MinMaxScaler, EarlyStopping)
# 1.0988354915753007e-05
# [[8.059416]]

# GRU 파라미터
# 3 X (input_dim + bias(1) + output) X output