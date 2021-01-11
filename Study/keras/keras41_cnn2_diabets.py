# CNN 으로 구성
# 2차원을 4차원으로 늘여서 하시오.

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.python.ops.gen_math_ops import Min

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)
print(x[:5])
print(y[:10])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (353, 10) (89, 10)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

print(x_train.shape, x_test.shape)  # (353, 10, 1, 1) (89, 10, 1, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', 
          strides=1, input_shape=(10, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(9, (2,2), padding='same'))
model.add(Flatten())
model.add(Dense(3000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=16, verbose=1, callbacks=[es])

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# sklearn Dense diabets 
# loss, mae :  3393.61083984375 47.12013244628906
# RMSE :  58.2547106883262
# mse :  3393.611317380587
# R2 :  0.47710474684639814

# skleran LSTM diabets
# loss, mae :  4602.60986328125 53.904117584228516
# RMSE :  67.84254315463556
# mse :  4602.610661688588
# R2 :  0.2908194127049417

# sklearn cnn diabets
# loss, mae :  4153.85986328125 49.24408721923828
# RMSE :  64.45044963301498
# mse :  4153.860457897802
# R2 :  0.35996385190814173