# hist를 이용하여 그래프를 그리시오.
# loss, val_loss, acc, val_acc

import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

# 1. 데이터
x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape)  # (178, 13)
print(y.shape)  # (178,)

# 실습, Dense

# 전처리 알아서 해 / MinMaxScaler, train_test_split
print(np.max(x[0]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# OneHotEncoding(tensorflow)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x.shape)            # (178, 13)
print(x_train.shape)      # (142, 13)
print(x_test.shape)       # (36, 13) 

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
model = Sequential()
model.add(Dense(13, activation='relu', input_shape=(13,)))  # input = 13
model.add(Dense(13, activation='relu'))    
model.add(Dense(26, activation='relu'))      
model.add(Dense(65, activation='relu'))      
model.add(Dense(13, activation='relu'))      
model.add(Dense(13, activation='relu'))      
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

hist = model.fit(x_train, y_train, epochs=2000, validation_split=0.2, 
                 verbose=1, batch_size=32, callbacks=[early_stopping])
print(hist)
print(hist.history.keys())

print(hist.history['loss'])

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val loss', 'train acc', 'val acc'])
plt.show()           

