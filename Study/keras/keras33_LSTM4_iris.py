# 사이킷런
# LSTM 으로 모델링
# Dense 와 성능비교
# 다중 분류

import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape)      # (150, 4)
print(y.shape)      # (150, )
print(x[:5])
print(y)

# 전처리 알아서 해 / MinMaxScaler, train_test_split

# ## OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# # from keras.utils.np_utils import to_categorical

# y = to_categorical(y)
# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)
 
print(y)
print(x.shape)  # (150, 4)
print(y.shape)  # (150, 3)

# print(np.max(x[0]))


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)
x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# # OneHotEncoding(tensorflow)
# from tensorflow.keras.utils import to_categorical
# # from keras.utils.np_utils import to_categorical

# y = to_categorical(y)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

# print(x_train.shape)    
# print(y_train.shape)    

# OneHotEncoding(sklearn)
from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
y_val = one.transform(y_val).toarray()

print(x.shape, x_train.shape, x_test.shape, x_val.shape) # (150, 4) (120, 4) (30, 4) (30, 4)
print(x_train.shape)    # (150, 4) -> (120, 4)
print(y_train.shape)    # (150,) -> (120, 3)       

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(4, 1))
aaa = LSTM(1000, activation='relu')(input1)
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
          epochs=2000, callbacks=[early_stopping], verbose=1, batch_size=13)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
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

# sklearn Dense iris
# loss, acc :  0.041334398090839386 1.0
# [[8.5435504e-14 3.4071996e-05 9.9996591e-01]
#  [2.0802061e-31 9.9999607e-01 3.9425763e-06]
#  [1.5944058e-04 1.3975589e-01 8.6008465e-01]
#  [1.9400661e-26 9.9997354e-01 2.6421489e-05]]
# [2 1 2 1]

# sklearn LSTM iris
# loss, acc :  0.08124087750911713 1.0
# [[2.7368754e-10 2.9734897e-03 9.9702650e-01]
#  [6.6601649e-05 9.9899536e-01 9.3812286e-04]
#  [5.4915622e-04 3.3081597e-01 6.6863483e-01]
#  [8.5338586e-05 9.9947685e-01 4.3781847e-04]]
# [2 1 2 1]
