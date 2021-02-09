import numpy as np

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape) # (160, 150, 150, 3) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 3) (120,)

# 실습
# 모델을 만들어랏!!!

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_test)
print("y_pred : ", y_pred)


