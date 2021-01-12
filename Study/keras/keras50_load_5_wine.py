import numpy as np

# 1. 데이터
x_data = np.load('../data/npy/wine_x.npy')
y_data = np.load('../data/npy/wine_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size = 0.8, random_state = 66, shuffle = True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_data.shape)          # (178, 13) 
print(x_train.shape)    # (142, 13)
print(x_test.shape)     # (36, 13) 

print(x_train.shape[0], x_train.shape[1])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

print(x_train.shape, x_test.shape)  # (142, 13, 1, 1) (36, 13, 1, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_data.shape)          # (178, 3)
print(y_train.shape)    # (142, 3)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
          strides=1, input_shape=(13, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='same'))
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
model.add(Dense(3, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=3000, validation_split=0.2, 
          verbose=1, batch_size=13, callbacks=[early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accruacy : ", accuracy)

# sklearn cnn wine
# loss :  0.09509149193763733
# accruacy :  0.9722222089767456

# load_5_wine
# loss :  0.015299942344427109
# accruacy :  1.0