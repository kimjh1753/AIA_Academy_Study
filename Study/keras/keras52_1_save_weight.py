# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

print(x_train[0])
print("y_train[0] : ", y_train[0])
print(x_train[0].shape)                 # (28, 28)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.
# (x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# OnHotEncoding
# 여러분이 하시오!!!!!
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(filters=16, kernel_size=(4,4), padding='same', strides=1))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(10, activation='softmax'))

model.save('../data/h5/k52_1_model1.h5')

# 3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                     save_best_only=True, mode='auto')
# filepath - 가중치 세이브, 최저점을 찍을 때마다 weight 가 들어간 파일을 만듬
# 세이브 된 최적 가중치를 이용해서 모델 평가, 예측을 좀 더 쉽고 빠르게 할 수 있다
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[es, cp], batch_size=32)

model.save('../data/h5/k52_1_model2.h5')
model.save_weights('../data/h5/k52_1_weight.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss : ", result[0]) # 리스트의 첫번째 값은 loss
print("accuracy : ", result[1])  # 리스트이 두번째 값은 acc

# keras52_1_save_weight
# loss :  0.05136147886514664
# accuracy :  0.9843000173568726
