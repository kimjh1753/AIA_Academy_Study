import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

x = np.load('../study/samsung/삼성전자_x.npy')
y = np.load('../study/samsung/삼성전자_y.npy')

print(x.shape) # (662, 5)
print(y.shape) # (662, 1)

size = 30

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)
x_data = split_x(x, size)
print(x_data.shape) # (657, 6, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state=311, shuffle = True
)
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)   
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
       
print(x_train.shape, x_test.shape, x_val.shape)  # (423, 5, 1) (133, 5, 1) (106, 5, 1)

from tensorflow.keras.models import Sequential, load_model

model = load_model('../study/samsung/keras_Samsung_01.h5')
result = model.evaluate(x_test, y_test)
print("로드체크포인트_loss : ", result[0])
print("로드체크포인트_accuracy : ", result[1])

x_predict = np.array([[89800,91200,89100,89700,-0.99, 34161101, 
                       4557102, -1781416, -2125136, -579352, -2190214]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(1, 11, 1)
y_predict = model.predict(x_predict)
print(y_predict)
print("1월 14일 예상 값 : ", y_predict)

# 1일마다
# 로드체크포인트_loss :  212548.1875
# 로드체크포인트_accuracy :  356.607421875
# 1월 14일 예상 값 :  [[88792.44]]

# 7일마다
# 로드체크포인트_loss :  320810.78125
# 로드체크포인트_accuracy :  488.83770751953125
# 1월 14일 예상 값 :  [[89817.484]]

# 30일마다
# 로드체크포인트_loss :  340003.4375
# 로드체크포인트_accuracy :  475.1623840332031
# 1월 14일 예상 값 :  [[87813.086]]