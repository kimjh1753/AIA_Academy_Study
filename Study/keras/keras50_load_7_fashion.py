import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# 1. 데이터
(f_x_train, f_y_train), (f_x_test, f_y_test) = fashion_mnist.load_data()

x_data = np.load('../data/npy/fashion_x.npy')
x_data = np.load('../data/npy/fashion_x.npy')
y_data = np.load('../data/npy/fashion_y.npy')
y_data = np.load('../data/npy/fashion_y.npy')

x_train = f_x_train.reshape(f_x_train.shape[0], f_x_train.shape[1], f_x_train.shape[2], 1).astype('float32')/255.
x_test = f_x_test.reshape(f_x_test.shape[0], f_x_test.shape[1], f_x_test.shape[2], 1)/255.

# OnHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(f_y_train)
y_test = to_categorical(f_y_test)

print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', 
                 strides=1, input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2))
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
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

# keras fashion cnn
# loss :  0.5602368116378784
# acc :  0.9172999858856201

# load_7_fashion
# loss :  0.5568920969963074
# acc :  0.9115999937057495