# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성하시오.
# 실습 19_2

#1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))

print(datasets.feature_names)
# print(datasets.DESCR)

# 데이터 전처리(MinMax)
# x = (x - 최소) / (최대 - 최소)
#   = (x - np.mix(x) / (np.max(x) - np.min(x)))
x = (x - np.min(x)) / (np.max(x) - np.min(x))
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)

#2. 모델 구성
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


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=8,
          validation_split=0.2, verbose=1)

#4  평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 실습 1
# loss, mae :  3393.61083984375 47.12013244628906
# RMSE :  58.2547106883262
# mse :  3393.611317380587
# R2 :  0.47710474684639814

# 실습 2
# loss, mae :  3397.3935546875 47.2345085144043
# RMSE :  58.28716414270366
# mse :  3397.3935037984793
# R2 :  0.4765219790690959
