# 2개의 모델을 하나는 LSTM, 하나는 Dense로
# 앙상블 구현!!!
# 29_1 번과 성능 비교

import numpy as np
from numpy import array

# 1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,70], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110], [100,110,120],
              [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])  # (3,) -> (1, 3) -> (1, 3, 1)
                                         #Dense     #LSTM  

print(x1.shape)         # (13, 3)
print(x2.shape)         # (13, 3)
print(y.shape)          # (13,) 
print(x1_predict.shape) # (3,) 
print(x2_predict.shape) # (3,)

#### 실습 : 앙상블 모델을 만드시오.

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, shuffle = False, train_size = 0.8
)

# x1 = x1.reshape(13, 3, 1)
# x2 = x2.reshape(13, 3, 1)
# x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
x1_predict = x1_predict.reshape(1, 3)
x2_predict = x2_predict.reshape(1, 3)

# 2. 모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# 모델 1
input1 = Input(shape=(3))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(input1)

# 모델 2
input2 = Input(shape=(3, 1))
dense2 = LSTM(100, activation='relu')(input2)
dense2 = Dense(200, activation='relu')(dense2)
dense2 = Dense(200, activation='relu')(dense2)
dense2 = Dense(200, activation='relu')(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(300)(merge1)
middle1 = Dense(300)(merge1)
middle1 = Dense(300)(merge1)
middle1 = Dense(300)(merge1)
middle1 = Dense(300)(merge1)

# 모델 분기 1
output1 = Dense(300)(middle1)
output1 = Dense(100)(output1)
output1 = Dense(100)(output1)
output1 = Dense(1)(output1)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=output1)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='mse', patience=30, mode='auto')
model.fit([x1, x2], y, epochs=2000, batch_size=1, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# 4. 평가, 예측
result = model.evaluate([x1, x2], y, batch_size=1)
print("result : ", result)

y_predict = model.predict([x1_predict, x2_predict])
print("y_predict : ", y_predict)

# LSTM_ensemble
# result :  12.404455184936523
# y_predict :  [[85.959]]

# LSTM과 Dense ensemble
# result :  32.986637115478516
# y_predict :  [[72.30502]]