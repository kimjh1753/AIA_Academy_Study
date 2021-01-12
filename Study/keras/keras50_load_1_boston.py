import numpy as np

# 1. 데이터
x_data = np.load('../data/npy/boston_x.npy')
y_data = np.load('../data/npy/boston_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=0.8, random_state=66, shuffle=True
)

# 데이터 전처리(MinMaxScaler)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

print(x_train.shape, x_test.shape)  # (404, 13, 1, 1) (102, 13, 1, 1)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

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
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=32, validation_split=0.2, callbacks=[es], verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# skleran cnn boston
# loss :  12.100605964660645
# RMSE :  3.4785928650806563
# mse :  12.10060832099005
# R2 :  0.8552263331814386

# load_1_boston
# loss :  9.944470405578613
# RMSE :  3.1534856464483014
# mse :  9.94447172235546
# R2 :  0.8810227058319411