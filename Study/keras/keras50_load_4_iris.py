import numpy as np

# 1. 데이터
x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

# 전처리 알아서 해 / MinMaxScaler, train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size = 0.8, random_state = 66, shuffle = True
)
x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, train_size = 0.8, random_state = 66, shuffle = True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# OneHotEncoding(tensorflow)
from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

print(x_train.shape)    # (150, 4) -> (120, 4)
print(y_train.shape)    # (150,) -> (120, 3)     

# OneHotEncoding(sklearn)
# from sklearn.preprocessing import OneHotEncoder
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# y_val = y_val.reshape(-1, 1)

# one = OneHotEncoder()
# one.fit(y_train)
# y_train = one.transform(y_train).toarray()
# y_test = one.transform(y_test).toarray()
# y_val = one.transform(y_val).toarray()

# print(x_train.shape)    # (150, 4) -> (120, 4)
# print(y_train.shape)    # (150,) -> (120, 3)       

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1, 1)

print(x_train.shape, x_test.shape, x_val.shape) # (120, 4, 1, 1) (30, 4, 1, 1) (30, 4, 1, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
          strides=1, input_shape=(4, 1, 1)))
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
                   # mean_squared_error
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')              

model.fit(x_train, y_train, validation_data=(x_val, y_val), 
          epochs=2000, callbacks=[early_stopping], verbose=1, batch_size=15)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss, acc : ", loss, acc)

# y[-5:-1] = ? 0 아니면 1
# y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
# print(y_test[-5:-1])

y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred)
print(y_pred.argmax(axis=1))

# sklearn cnn iris
# loss, acc :  0.09878843277692795 1.0
# [[4.7329595e-11 3.6303129e-03 9.9636972e-01]
#  [1.2238149e-08 9.9941492e-01 5.8506744e-04]
#  [9.4182706e-03 4.2638236e-01 5.6419939e-01]
#  [1.8536648e-12 9.9998701e-01 1.2974484e-05]]
# [2 1 2 1]


# load_4_iris
# loss, acc :  0.0767645612359047 1.0
# [[1.8267774e-16 3.8487917e-05 9.9996150e-01]
#  [7.0052709e-12 9.9937683e-01 6.2313821e-04]
#  [3.2612756e-03 3.1215015e-01 6.8458861e-01]
#  [1.4410473e-15 9.9995649e-01 4.3463366e-05]]
# [2 1 2 1]