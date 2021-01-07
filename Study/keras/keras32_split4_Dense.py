# 과제 및 실습        Dense
# 전처리, EarlyStopping 등등 다 넣을 것!!
# 데이터는 1~100 / 
#     x              y
# 1, 2, 3, 4, 5      6
# ...
# 95,96,97,98,99    100

# predict 만들 것
# 96,97,98,99,100 -> 101
# ...
# 100,101,102,103,104 -> 105 
# 예상 predict는 (101, 102, 103, 104, 105)

import numpy as np

# 1. 데이터
a = np.array(range(1, 101))
size = 6

# 모델을 구성하시오.

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)    

dataset = split_x(a, size)
# print("==========================")
# print(dataset)

x = dataset[:,0:5]
y = dataset[:,5:]
print(x)
print(y)
print(x.shape) # (95, 5)
print(y.shape) # (95, 1)

x1_pred = np.array([96,97,98,99,100])
x2_pred = np.array([97,98,99,100,101])
x3_pred = np.array([98,99,100,101,102])
x4_pred = np.array([99,100,101,102,103])
x5_pred = np.array([100,101,102,103,104])

x1_pred = x1_pred.reshape(1, 5)
x2_pred = x2_pred.reshape(1, 5)
x3_pred = x3_pred.reshape(1, 5)
x4_pred = x4_pred.reshape(1, 5)
x5_pred = x5_pred.reshape(1, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle=True
)
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size = 0.8, random_state = 66, shuffle=True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x1_pred = scaler.transform(x1_pred)
x2_pred = scaler.transform(x2_pred)
x3_pred = scaler.transform(x3_pred)
x4_pred = scaler.transform(x4_pred)
x5_pred = scaler.transform(x5_pred)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=5))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='mse', patience=10, mode='auto')
model.fit(x, y, epochs=2000, validation_data=(x_val, y_val), batch_size=13, verbose=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=13)
print(loss)

result1 = model.predict(x1_pred)
result2 = model.predict(x2_pred)
result3 = model.predict(x3_pred)
result4 = model.predict(x4_pred)
result5 = model.predict(x5_pred)

print("x1_pred : ", result1)
print("x2_pred : ", result2)
print("x3_pred : ", result3)
print("x4_pred : ", result4)
print("x5_pred : ", result5)

# keras32_split3_LSTM.py
# 0.029302777722477913
# x1_pred :  [[101.54239]]
# x2_pred :  [[102.587364]]
# x3_pred :  [[103.63438]]
# x4_pred :  [[104.68373]]
# x5_pred :  [[105.78269]]

# keras32_split4_Dense
# 0.020351111888885498
# x1_pred :  [[101.2414]]
# x2_pred :  [[102.24285]]
# x3_pred :  [[103.218605]]
# x4_pred :  [[104.19083]]
# x5_pred :  [[105.15988]]