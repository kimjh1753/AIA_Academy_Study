# 실습 
# 드랍아웃 적용
import numpy as np

from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)  # (506, 13)
print(y.shape)  # (506,)
print("==========================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0 0.0
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리(MinMax)
# x = x /711.               
# x = (x - 최소) / (최대 - 최소)
#   = (x - np.mix(x) / (np.max(x) - np.min(x)))
print(np.max(x[0]))

from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# print(np.max(x), np.min(x)) # 711.0 0.0 => 1.0 0.0
# print(np.max(x[0]))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)
print(x_train.shape)
print(y_train.shape)

x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, train_size = 0.8, shuffle=True)

print(x_train.shape)
print(x_val.shape)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
# model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=40, mode='auto')

model.fit(x_train, y_train, validation_data=(x_val, y_val), 
        epochs=2000, batch_size=8, callbacks=[early_stopping], verbose=1)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss, mae : ", loss, mae)

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

# 전처리 전
# loss, mae :  15.841497421264648 3.3060100078582764
# RMSE :  3.980137940751986
# mse :  15.84149802741346
# R2 :  0.8104697138779872

# 전처리 후 x = x / 711.
# loss mae :  12.428569793701172 2.640331745147705
# RMSE :  3.5254177971244336
# mse :  12.428570644281695
# R2 :  0.851302538041412

# x 통째로 전처리한놈
# loss, mae :  9.726927757263184 2.064275026321411
# RMSE :  3.118802284915046
# mse :  9.72692769239131
# R2 :  0.8836254383621532

# 제대로 전처리
# loss, mae :  521.8694458007812 10.636632919311523
# RMSE :  22.844437006386865
# mse :  521.8683021387776
# R2 :  -5.243718141504652

# validation 값 분리
# loss, mae :  6.475992679595947 1.9397125244140625
# RMSE :  2.5447971155415856
# mse :  6.475992359268774
# R2 :  0.9225201630141171

# EarlyStopping(patience=30) 사용
# loss, mae :  5.496500492095947 1.8316468000411987
# RMSE :  2.344461644241536
# mse :  5.496500401319727
# R2 :  0.9342389657891477

# Dropout 이후
# loss, mae :  6.073357105255127 1.9191125631332397
# RMSE :  2.46441815380415
# mse :  6.073356836799455
# R2 :  0.9273373605824557