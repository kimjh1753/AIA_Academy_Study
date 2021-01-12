import numpy as np

# 1. 데이터
x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)   # (150, 4) (150,)

# 모델을 완성하시오!!!

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

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
aaa = Dense(1000, activation='relu')(input1)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
outputs = (Dense(3, activation='softmax'))(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
                   # mean_squared_error
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')              

model.fit(x_train, y_train, validation_data=(x_val, y_val), 
          epochs=1000, callbacks=[early_stopping], verbose=1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss, acc : ", loss, acc)

# y[-5:-1] = ? 0 아니면 1
# y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
# print(y_test[-5:-1])

# loss, acc :  0.07098645716905594 1.0
# [[1.6942617e-22 1.5770547e-06 9.9999845e-01]
#  [1.0000000e+00 0.0000000e+00 0.0000000e+00]
#  [1.0000000e+00 0.0000000e+00 0.0000000e+00]
#  [1.0837853e-03 4.1211313e-01 5.8680314e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]

y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred)
print(y_pred.argmax(axis=1))

# loss, acc :  0.041334398090839386 1.0
# [[8.5435504e-14 3.4071996e-05 9.9996591e-01]
#  [2.0802061e-31 9.9999607e-01 3.9425763e-06]
#  [1.5944058e-04 1.3975589e-01 8.6008465e-01]
#  [1.9400661e-26 9.9997354e-01 2.6421489e-05]]
# [2 1 2 1]

# load_npy_iris
# loss, acc :  0.08711619675159454 0.9666666388511658
# [[5.82321069e-16 1.13473325e-05 9.99988675e-01]
#  [1.81890636e-09 9.99969602e-01 3.04327878e-05]
#  [5.89580259e-06 3.39810699e-02 9.66013014e-01]
#  [5.52343393e-10 9.99982357e-01 1.76427238e-05]]
# [2 1 2 1]
