import numpy as np

# 1. 데이터
x_data = np.load('../data/npy/diabetes_x.npy')
y_data = np.load('../data/npy/diabetes_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=0.8, random_state=66, shuffle=True
)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(10,))
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=8, epochs=100, 
          validation_split=0.2, verbose=1)

# 4. 평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
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

# 실습 1
# loss, mae :  3393.61083984375 47.12013244628906
# RMSE :  58.2547106883262
# mse :  3393.611317380587
# R2 :  0.47710474684639814

# load_2_diabetes 
# loss, mae :  3342.246826171875 47.57976150512695
# RMSE :  57.81216679455603
# mse :  3342.2466294815663
# R2 :  0.48501913331268487