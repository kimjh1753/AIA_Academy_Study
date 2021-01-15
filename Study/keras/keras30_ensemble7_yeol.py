# 열이 다른 앙상블 모델에 대해 공부!!!

import numpy as np
from numpy import array

# 1. 데이터
x1 = np.array([[1,2], [2,3], [3,4], [4,5],
              [5,6], [6,7], [7,8], [8,9],
              [9,10], [10,11], 
              [20,30], [30,40], [40,50]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,70], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110], [100,110,120],
              [2,3,4], [3,4,5], [4,5,6]])
y1 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,70], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110], [100,110,120],
              [2,3,4], [3,4,5], [4,5,6]])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65])
x2_predict = array([65,75,85])  # (3,) -> (1, 3) -> (1, 3, 1)
                                         #Dense     #LSTM  

print(x1.shape)          # (13, 2)
print(x2.shape)          # (13, 3)
print(y1.shape)          # (13, 3) 
print(y2.shape)          # (13,) 
print(x1_predict.shape)  # (2,) 
print(x2_predict.shape)  # (3,)


#### 실습 : 앙상블 모델을 만드시오.

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle = False, train_size = 0.8
)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle = False, train_size = 0.8
)

# x1 = x1.reshape(13, 2, 1)
# x2 = x2.reshape(13, 3, 1)
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
x1_predict = x1_predict.reshape(1, 2, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

# 2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

# 모델 1
input1 = Input(shape=(2, 1))
dense1 = LSTM(100, activation='relu')(input1)
dense1 = Dense(20, activation='relu')(dense1)
dense1 = Dense(20, activation='relu')(dense1)
dense1 = Dense(20, activation='relu')(dense1)

# 모델 2
input2 = Input(shape=(3, 1))
dense2 = LSTM(130, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense2)
dense2 = Dense(20, activation='relu')(dense2)
dense2 = Dense(20, activation='relu')(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(30)(merge1)
middle1 = Dense(30)(merge1)


# 모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(100)(output1)
output1 = Dense(100)(output1)
output1 = Dense(3)(output1)

# 모델 분기 2
output2 = Dense(30)(middle1)
output2 = Dense(100)(output2)
output2 = Dense(100)(output2)
output2 = Dense(1)(output2)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit([x1, x2], [y1, y2], epochs=3000, batch_size=1, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# 4. 평가, 예측
result = model.evaluate([x1, x2], [y1, y2], batch_size=1)
print("result : ", result)

y_predict = model.predict([x1_predict, x2_predict])
print("y_predict : ", y_predict)

# LSTM_ensemble
# result :  12.404455184936523
# y_predict :  [[85.959]]

# ensemble6_heng
# ValueError: Data cardinality is ambiguous:
#   x sizes: 10, 13
#   y sizes: 10, 13

# result :  [28488.166015625, 28378.29296875, 109.87432098388672, 79.10661315917969, 5.041811943054199]
# y_predict :  [array([[529.68005, 569.1087 , 601.7425 ]], dtype=float32), array([[61.552765]], dtype=float32)]

