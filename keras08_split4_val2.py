# 실습 validation_data 를 만들기!!!!
# train_test_split를 사용할 것

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# x_train = x[:60]    # 순서 0번째부터 59번째까지 :::: 값 1 ~ 60 
# x_val = x[60:80]    # 61 ~ 80
# x_test = x[80:]     # 81 ~ 100
# 리스트의 슬라이싱 

# y_train = y[:60]    # 순서 0번째부터 59번째까지 :::: 값 1 ~ 60 
# y_val = y[60:80]    # 61 ~ 80
# y_test = y[80:]     # 81 ~ 100
# 리스트의 슬라이싱 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  
                                                  train_size=0.8, shuffle=True)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# Shuffle이 False일때
# loss :  8.501083357259631e-07
# mae :  0.0009212493896484375

# Shuffle이 True일때
# loss :  8.836965048608647e-10
# mae :  2.63899564743042e-05

# validation = 0.2
# loss :  0.0009252004092559218
# mae :  0.025919830426573753

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
