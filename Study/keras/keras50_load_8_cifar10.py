import numpy as np
from tensorflow.keras.datasets import cifar10

# 1. 데이터
(c10_x_train, c10_y_train), (c10_x_test, c10_y_test) = cifar10.load_data()

x_data = np.load('../data/npy/cifar10_x.npy')
x_data = np.load('../data/npy/cifar10_x.npy')
y_data = np.load('../data/npy/cifar10_y.npy')
y_data = np.load('../data/npy/cifar10_y.npy')

x_train = c10_x_train.reshape(c10_x_train.shape[0], c10_x_train.shape[1], c10_x_train.shape[2], c10_x_train.shape[3])/255.
x_test = c10_x_test.reshape(c10_x_test.shape[0], c10_x_test.shape[1], c10_x_test.shape[2], c10_x_test.shape[3])/255.

print(x_train.shape, x_test.shape)   # (50000, 32, 32, 3) (50000, 1)

# OnHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(c10_y_train)
y_test = to_categorical(c10_y_test)

print(y_train.shape)    # (50000, 10)
print(y_test.shape)     # (10000, 10)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

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
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=2000, validation_split=0.2, verbose=1, callbacks=[es])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("accuracy : ", acc)

# keras cifar10 cnn
# loss :  3.212538480758667
# accuracy :  0.5156999826431274

# load_8 cifar10
# loss :  2.7016751766204834
# accuracy :  0.5307999849319458
