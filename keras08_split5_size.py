# 실습 train_size / test_size에 대해 완벽 이해할 것

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
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    # train_size=0.9, test_size=0.2, shuffle=True)
                                                    train_size=0.2, test_size=0.7, shuffle=False)
                                                    # 위 두가지의 경우에 대해 확인 후 정리할 것!!!

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# train_size=0.9, test_size=0.2, shuffle=True
# ValueError: The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.

# train_size=0.2, test_size=0.9, shuffle=True)
# ValueError: The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.

# train_size=0.7, test_size=0.2, shuffle=True
# [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
#  25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
#  49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70]
# (70,)
# (20,)
# (70,)
# (20,)

'''
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
'''