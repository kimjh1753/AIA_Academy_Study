# 실습
# R2를 음수가 아닌 0.5 이하로 줄이기
# 1. 레이어는 인풋과 아웃풋을 포함 5개 이상
# 2. batch_size = 1
# 3. epochs = 100 이상
# 4. 히든레이어의 노드의 갯수는 10 이상
# 5. 데이터 조작 금지.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array
# np.array
# array()

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])
x_pred = array([16,17,18])

#2. 모델링
model = Sequential()
model.add(Dense(200, input_dim=1, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(15, activation='relu'))
model.add(Dense(17, activation='linear'))
model.add(Dense(300))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(200))
model.add(Dense(500))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=147, batch_size=1, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", results)

y_predict = model.predict(x_test)
# print("y_predict : ", y_predict)
# np.sqrt(results[0])

# 사이킷런 sklearn
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)



