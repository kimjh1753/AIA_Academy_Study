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
from tensorflow.keras.layers import Dense, Conv1D, Flatten
model = Sequential()
model.add(Conv1D(filters=200, kernel_size=2, padding='same', input_shape=(28*28,1)))
model.add(Flatten())
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

# keras_40_mnist4_lstm
# loss :  0.08996500074863434
# acc :  0.11349999904632568
# [[0.1015713  0.11221761 0.09865601 0.10124439 0.09684425 0.0904313
#   0.09777842 0.10323054 0.09756004 0.10046616]]
# ======================

# [[0.1015713  0.11221761 0.09865601 0.10124439 0.09684425 0.0904313
#   0.09777842 0.10323054 0.09756004 0.10046616]]

# conv1d_07_mnist
# loss :  0.1981179267168045
# acc :  0.9391999840736389
# [[1.2089557e-05 1.2149359e-03 1.7330735e-03 7.1673468e-02 1.8096423e-06
#   9.2438346e-01 2.5927767e-04 6.4673967e-04 2.9331226e-05 4.5813958e-05]]
# ======================

# [[6.3425246e-06 7.0553666e-05 8.1437378e-04 2.9978184e-05 1.3049509e-05
#   4.0742732e-05 7.9702040e-05 9.9887472e-01 2.1434824e-05 4.9038277e-05]]