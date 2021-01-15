import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

data = np.load('../study/samsung/삼성전자.npy')

x = data[:, [0,1,2,3,4]]   # (2398, 14)
y = data[:, [3]]   # (2398, 14)

print(data.shape) # (2398, 14)
print(x.shape) # (2398, 5)
print(y.shape) # (2398, 1)

size = 20

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)
x_data = split_x(x, size)
print(x_data.shape) # (2379, 20, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, shuffle = False, random_state=311
)
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.8, shuffle=False, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)   
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

print(x_train.shape, x_test.shape, x_val.shape)  # (1534, 5, 1) (480, 5, 1) (384, 5, 1)

# 2~3. 모델 구성, (컴파일, 훈련)
from tensorflow.keras.models import Sequential, load_model

# 4. 평가, 예측
model = load_model('../study/samsung/keras_Samsung0114.h5')
result = model.evaluate(x_test, y_test)
print("로드체크포인트_loss : ", result[0])
print("로드체크포인트_accuracy : ", result[1])

x_predict = np.array([[88700,90000,88700,89700,0]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(1, 5, 1)
y_predict = model.predict(x_predict)
print(y_predict)
print("1월 15일 예상 값 : ", y_predict)

# 로드체크포인트_loss :  320810.78125
# 로드체크포인트_accuracy :  488.83770751953125
# 1월 14일 예상 값 :  [[89817.484]]

# 로드체크포인트_loss :  14374459.0
# 로드체크포인트_accuracy :  3728.546875
# 1월 15일 예상 값 :  [[90131.28]]

