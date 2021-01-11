# 주말과제
# LSTM 모델로 구성 input_shape=(28*28, 1)
# LSTM 모델로 구성 input_shape=(28*14, 2)
# LSTM 모델로 구성 input_shape=(28*7, 4)
# LSTM 모델로 구성 input_shape=(7*7, 16)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

# 1. 데이터
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

print(x_train[0])
print("y_train[0] : ", y_train[0])
print(x_train[0].shape)                 # (28, 28)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[1], 1)

#OnHotEncdoing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)        # (60000, 10)
print(y_test.shape)         # (10000, 10)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(28*28,1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=1, batch_size=200, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

y_test = np.array(model.predict(x_train[:1]))
print(y_test[:10])
print("======================\n")

y_pred = np.array(model.predict(x_test[:1]))
print(y_pred[:10])

# keras40_mnist2_cnn
# loss :  0.00260396976955235
# acc :  0.9854999780654907
# [[8.6690171e-08 2.8707976e-08 9.1137373e-09 9.6521189e-06 4.6547077e-09
#   9.9998856e-01 7.6187533e-08 5.5741470e-08 1.3864026e-06 2.0224462e-07]]
# ============
# [[7.0327958e-30 2.2413428e-23 6.9391834e-21 9.2217209e-22 5.1841172e-22
#   8.7506048e-26 2.4799229e-27 1.0000000e+00 8.0364114e-26 3.3208760e-17]]

# keras40_mnist3_dnn
# loss :  0.005172424484044313
# acc :  0.9724000096321106
# [[9.4863184e-15 2.2668929e-19 1.8625454e-22 5.9676188e-07 2.5733180e-25
#   9.9999940e-01 1.5588427e-20 7.8994310e-23 5.6835017e-22 2.6443269e-20]]
# ============
# [[3.0520350e-26 2.7246760e-23 4.5444517e-25 3.6449811e-28 1.3460386e-28
#   2.1042897e-27 6.9805158e-30 1.0000000e+00 1.8761058e-26 2.6409651e-25]]

# keras_40_mnist4_lstm
# loss :  0.08996500074863434
# acc :  0.11349999904632568
# [[0.1015713  0.11221761 0.09865601 0.10124439 0.09684425 0.0904313
#   0.09777842 0.10323054 0.09756004 0.10046616]]
# ======================

# [[0.1015713  0.11221761 0.09865601 0.10124439 0.09684425 0.0904313
#   0.09777842 0.10323054 0.09756004 0.10046616]]