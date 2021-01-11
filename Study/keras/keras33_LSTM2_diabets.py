# 사이킷런
# LSTM 으로 모델링
# Dense 와 성능비교

# 1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)
print(x[:5])
print(y[:10])
 
print(np.max(x), np.min(x)) # 0.198787989657293 -0.137767225690012
print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x.shape, x_train.shape, x_test.shape) # (442, 10) (353, 10) (89, 10)

x = x.reshape(442, 10, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(10, 1)))
model.add(Dense(26, activation='relu'))      
model.add(Dense(65, activation='relu'))      
model.add(Dense(13, activation='relu'))      
model.add(Dense(13, activation='relu')) 
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, batch_size=13, epochs=300, 
          validation_split=0.2, verbose=1)

# 4. 평가 예측
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

