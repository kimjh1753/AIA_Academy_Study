# 실습
# 남자 여자 구별
# ImageDataGenator을 이용하여 fit 사용해서 완성

import numpy as np

x_train = np.load('../data2/image/npy/keras67_train_x.npy')
y_train = np.load('../data2/image/npy/keras67_train_y.npy')
x_test = np.load('../data2/image/npy/keras67_test_x.npy')
y_test = np.load('../data2/image/npy/keras67_test_y.npy')

print(x_train.shape, y_train.shape) # (1389, 128, 128, 3) (1389,)
print(x_test.shape, y_test.shape)   # (347, 128, 128, 3) (347,)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization

model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, 
          validation_split=0.2, verbose=1, callbacks=[es, rl])

loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
# loss :  0.5072045922279358
print("acc : ", acc)
# acc :  0.4927953779697418

# y_pred = model.predict(x_test)
# print("y_pred : ", y_pred)