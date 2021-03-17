import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load('../study/LPD_COMPETITION/npy/train_x.npy')
y_train = np.load('../study/LPD_COMPETITION/npy/train_y.npy')
x_test = np.load('../study/LPD_COMPETITION/npy/test_x.npy')
y_test = np.load('../study/LPD_COMPETITION/npy/test_y.npy')

print(x_train.shape, y_train.shape) # (48000, 28, 28, 3) (48000, 1000)
print(x_test.shape, y_test.shape)   # (9000, 28, 28, 3) (9000, 1000)

model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(28*28, 3)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=300, validation_split=0.2, callbacks=[es], batch_size=64)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)





