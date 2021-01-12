import numpy as np
from tensorflow.keras.datasets import cifar100

# 1. 데이터
(c100_x_train, c100_y_train), (c100_x_test, c100_y_test) = cifar100.load_data()

x_data = np.load('../data/npy/cifar100_x.npy')
x_data = np.load('../data/npy/cifar100_x.npy')
y_data = np.load('../data/npy/cifar100_y.npy')
y_data = np.load('../data/npy/cifar100_y.npy')

x_train = c100_x_train.reshape(c100_x_train.shape[0], c100_x_train.shape[1], c100_x_train.shape[2], c100_x_train.shape[3])/255.
x_test = c100_x_test.reshape(c100_x_test.shape[0], c100_x_test.shape[1], c100_x_test.shape[2], c100_x_test.shape[3])/255.

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(c100_y_train)
y_test = to_categorical(c100_y_test)

print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', 
                 strides=1, input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(9, (2,2), padding='valid'))
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
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=2000, validation_split=0.2, verbose=1, callbacks=[es])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("accuracy : ", acc)

# keras cifar100 cnn
# loss :  5.992544174194336
# loss :  0.23280000686645508

# load_9_cifar100
# loss :  5.810074806213379
# accuracy :  0.23880000412464142