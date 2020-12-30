# 실습
# 다:1 mlp 함수형
# keras10_mlp2.py를 함수형으로 바꾸시오

import numpy as np
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101), range(501, 601)])
y = np.array([range(711, 811), range(211, 311)])

print(x.shape)      # (4, 100)
print(y.shape)      # (2, 100)

# x = np.arange(20).reshape(10,2)
x = np.transpose(x)
y = np.transpose(y)
print(x)
print(x.shape)      # (100, 4)
print(y.shape)      # (100, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)
print(x_train.shape)        # (80, 4)
print(y_train.shape)        # (80,)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# from keras.layers import Dense

input1 = Input(shape=(4,))
aaa = Dense(10, activation='relu')(input1)
aaa = Dense(100)(aaa)
aaa = Dense(50)(aaa)
aaa = Dense(50)(aaa)
aaa = Dense(50)(aaa)
outputs = Dense(2)(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary

# model = Sequential()
# model.add(Dense(10, input_dim=4))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(2))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=300, batch_size=1, 
          validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE
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









