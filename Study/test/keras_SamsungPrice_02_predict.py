import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

x_data = np.load('../data/npy/삼성전자_x.npy', allow_pickle=True)
y_data = np.load('../data/npy/삼성전자_y.npy', allow_pickle=True)

print(x_data.shape) # (662, 5)
print(y_data.shape) # (662, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size = 0.8, random_state=66, shuffle = True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)   
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)        

print(x_train.shape, x_test.shape) # (529, 5, 1) (133, 5, 1)


from tensorflow.keras.models import Sequential, load_model

model = load_model('../data/h5/keras_Samsung_01.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", result[0])
print("accuracy : ", result[1])

y_predict = model.predict(x_test[-1:])
print(y_predict)

# loss :  28272.9296875
# accuracy :  0.0
# [[49392.93]]








